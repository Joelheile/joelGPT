import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

# ai_gpt: the same idea as joel_gpt.py, rebuilt with what recent small-model research trains fastest.
# Main sources: nanochat (github.com/karpathy/nanochat), modded-nanogpt (github.com/KellerJordan/modded-nanogpt),
# parameter-golf (github.com/openai/parameter-golf), Muon (arxiv.org/abs/2502.16982),
# MobileLLM (arxiv.org/abs/2402.14905), Value Residual Learning (arxiv.org/abs/2410.17897).


# settings, about 22M weights
vocab_size = 4096
batch_size = 32
accumulation_steps = 2 # 2 x 32 chunks per update
block_size = 256 # tokens per chunk, about one story
n_embedding = 384
n_heads = 6
head_size = n_embedding // n_heads
n_layers = 8
hidden_size = 4 * n_embedding
value_embedding_layers = range(1, n_layers, 2)
logit_softcap = 15
learning_rate = 3e-3
warmup_steps = 200
cooldown_fraction = 0.45 # last 45% of steps decay to 0
eval_interval = 1000
device = "mps"
memory_fraction = 0.3 # leave gpu memory for other apps
gpu_pause = 0.02 # seconds of rest per step so the mac stays usable
temperature = 0.8
top_k = 40


# data (bpe tokenizer, text encoded once to uint16 files)
def train_tokenizer():
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<|endoftext|>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tokenizer.train(["data/tinystories-test.txt"], trainer)
    tokenizer.save("data/ai_gpt-tokenizer.json")

def encode_file(text_path, token_path):
    tokenizer = Tokenizer.from_file("data/ai_gpt-tokenizer.json")
    with open(text_path) as text_file, open(token_path + ".tmp", "wb") as token_file:
        while lines := text_file.readlines(16_000_000): # 16 mb at a time, 2.2 gb never sits in ram
            for encoding in tokenizer.encode_batch_fast(lines):
                np.array(encoding.ids, dtype=np.uint16).tofile(token_file)
    os.replace(token_path + ".tmp", token_path)

if not os.path.exists("data/ai_gpt-tokenizer.json"):
    train_tokenizer()
for text_path, token_path in [("data/tinystories-test.txt", "data/ai_gpt-val.bin"), ("data/tinystories-train.txt", "data/ai_gpt-train.bin")]:
    if not os.path.exists(token_path):
        print(f"encoding {text_path}, once")
        encode_file(text_path, token_path)

tokenizer = Tokenizer.from_file("data/ai_gpt-tokenizer.json")
end_token = tokenizer.token_to_id("<|endoftext|>")
train_data = np.memmap("data/ai_gpt-train.bin", dtype=np.uint16, mode="r")
val_data = np.memmap("data/ai_gpt-val.bin", dtype=np.uint16, mode="r")
max_steps = len(train_data) // (batch_size * accumulation_steps * block_size) # one pass over the data

def get_batch(data):
    starts = np.random.randint(0, len(data) - block_size - 1, batch_size)
    chunks = data[starts[:, None] + np.arange(block_size + 1)]
    chunks = torch.from_numpy(chunks.astype(np.int64)).to(device)
    return chunks[:, :-1], chunks[:, 1:]


# model
def rms_norm(x):
    return F.rms_norm(x, (x.size(-1),))

## rope: rotate query and key by position instead of adding a position table
def rope_tables():
    speeds = 1 / 10000 ** (torch.arange(0, head_size, 2) / head_size)
    angles = torch.outer(torch.arange(block_size), speeds)
    return angles.cos(), angles.sin()

def rotate(x, cos, sin):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(n_embedding, 3 * n_embedding, bias=False)
        self.out = nn.Linear(n_embedding, n_embedding, bias=False)

    def forward(self, x, extra_value, cos, sin):
        batch, length, _ = x.shape
        query, key, value = self.qkv(x).view(batch, length, 3, n_heads, head_size).transpose(1, 3).unbind(2)
        if extra_value is not None: # value embedding, a second token table added to the values
            value = value + extra_value.view(batch, length, n_heads, head_size).transpose(1, 2)
        query, key = rms_norm(query), rms_norm(key) # qk norm keeps scores small
        query, key = rotate(query, cos, sin), rotate(key, cos, sin)
        out = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        return self.out(out.transpose(1, 2).reshape(batch, length, n_embedding))

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Linear(n_embedding, hidden_size, bias=False)
        self.down = nn.Linear(hidden_size, n_embedding, bias=False)

    def forward(self, x):
        return self.down(F.relu(self.up(x)).square())

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = Attention()
        self.feed_forward = FeedForward()

    def forward(self, x, extra_value, cos, sin):
        x = x + self.attention(rms_norm(x), extra_value, cos, sin)
        return x + self.feed_forward(rms_norm(x))

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_table = nn.Embedding(vocab_size, n_embedding)
        nn.init.normal_(self.token_table.weight, std=0.02)
        self.value_embeddings = nn.ModuleDict({str(layer): nn.Embedding(vocab_size, n_embedding) for layer in value_embedding_layers})
        for table in self.value_embeddings.values():
            nn.init.zeros_(table.weight)
        self.blocks = nn.ModuleList([Block() for _ in range(n_layers)])
        for block in self.blocks: # every block starts as a no-op
            nn.init.zeros_(block.attention.out.weight)
            nn.init.zeros_(block.feed_forward.down.weight)
        self.residual_scales = nn.Parameter(torch.linspace(1.15, 1.05, n_layers))
        self.embedding_scales = nn.Parameter(torch.linspace(0.2, 0.05, n_layers))
        cos, sin = rope_tables()
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, tokens):
        length = tokens.shape[1]
        cos, sin = self.cos[:length], self.sin[:length]
        x = x0 = rms_norm(self.token_table(tokens))
        for layer, block in enumerate(self.blocks):
            x = self.residual_scales[layer] * x + self.embedding_scales[layer] * x0 # mix the embedding back in
            extra_value = self.value_embeddings[str(layer)](tokens) if str(layer) in self.value_embeddings else None
            x = block(x, extra_value, cos, sin)
        logits = rms_norm(x) @ self.token_table.weight.T # tied, the token table is also the output layer
        return logit_softcap * torch.tanh(logits / logit_softcap)

def loss_of(model, x, y):
    return F.cross_entropy(model(x).view(-1, vocab_size), y.reshape(-1))


# optimizer: muon for block matrices, adamw for the rest, warmup-flat-cooldown schedule
def make_optimizers(model):
    matrices = [weights for weights in model.blocks.parameters() if weights.dim() == 2]
    others = [model.token_table.weight, model.residual_scales, model.embedding_scales, *model.value_embeddings.parameters()]
    muon = torch.optim.Muon(matrices, lr=learning_rate, adjust_lr_fn="match_rms_adamw")
    adamw = torch.optim.AdamW(others, lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.0, fused=True)
    return [muon, adamw]

def learning_rate_scale(step):
    return min((step + 1) / warmup_steps, 1, (max_steps - step) / (cooldown_fraction * max_steps))

def set_learning_rate(optimizers, scale):
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["lr"] = learning_rate * scale


# training
def validation_loss(model):
    with torch.no_grad():
        return sum(loss_of(model, *get_batch(val_data)).item() for _ in range(40)) / 40

def train():
    os.nice(10)
    torch.mps.set_per_process_memory_fraction(memory_fraction)
    model = GPT().to(device)
    compiled = torch.compile(model)
    optimizers = make_optimizers(model)
    print(f"{sum(weights.numel() for weights in model.parameters()) / 1e6:.1f}M parameters, {max_steps} steps, ctrl+c stops and keeps the weights")
    print("compiling, first progress line after 100 steps, about 2 min")

    train_losses = []
    validation_losses = []
    last_print = time.time()
    try:
        for step in range(max_steps):
            set_learning_rate(optimizers, learning_rate_scale(step))
            for _ in range(accumulation_steps):
                loss = loss_of(compiled, *get_batch(train_data))
                (loss / accumulation_steps).backward()
            for optimizer in optimizers:
                optimizer.step()
                optimizer.zero_grad()
            if gpu_pause:
                torch.mps.synchronize()
                time.sleep(gpu_pause)

            if step % eval_interval == 0:
                validation_losses.append([step, validation_loss(compiled)])
                print(f"validation loss {validation_losses[-1][1]:.3f}")

            if (step + 1) % 100 == 0:
                train_losses.append([step, loss.item()])
                step_time = (time.time() - last_print) / 100
                last_print = time.time()
                hours_left = step_time * (max_steps - step) / 3600
                print(f"step {step + 1}/{max_steps}  loss {train_losses[-1][1]:.3f}  {step_time * 1000:.0f} ms/step  {hours_left:.1f} h left")
                save(model, step, train_losses, validation_losses)

        validation_losses.append([max_steps, validation_loss(compiled)])
        print(f"done, final validation loss {validation_losses[-1][1]:.3f}")
    except KeyboardInterrupt:
        print(f"stopped at step {step}, weights saved to weights/ai_gpt.pt")
    save(model, step, train_losses, validation_losses)

def save(model, step, train_losses, validation_losses):
    checkpoint = {"model": model.state_dict(), "step": step, "max_steps": max_steps,
                  "train_losses": train_losses, "validation_losses": validation_losses}
    torch.save(checkpoint, "weights/ai_gpt.tmp")
    os.replace("weights/ai_gpt.tmp", "weights/ai_gpt.pt")

def load():
    checkpoint = torch.load("weights/ai_gpt.pt")
    model = GPT().to(device)
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint


# generation
@torch.no_grad()
def generate(model, prompt, n_tokens):
    numbers = torch.tensor([[end_token] + tokenizer.encode(prompt).ids], device=device)
    for _ in range(n_tokens):
        logits = model(numbers[:, -block_size:])[:, -1] / temperature
        top_logits, top_numbers = logits.topk(top_k)
        next_number = top_numbers.gather(1, torch.multinomial(F.softmax(top_logits, dim=-1), 1))
        if next_number.item() == end_token:
            break
        numbers = torch.cat([numbers, next_number], dim=1)
    return tokenizer.decode(numbers[0].tolist())


# python ai_gpt.py trains, python ai_gpt.py "Once upon a time" generates
if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(generate(load()[0], sys.argv[1], 400))
    else:
        train()
