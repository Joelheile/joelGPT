import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

# settings
vocab_size = 4096 # bpe tokens, about 4 chars each
batch_size = 32 # chunks per forward and backward pass
accumulation_steps = 2 # passes per update, 2 x 32 = 64 chunks, half the gpu memory of one 64 pass
block_size = 256 # tokens per chunk, about 1000 chars, most stories fit
n_embedding = 384 # numbers per token vector
n_heads = 6 # attention heads, 64 numbers each
head_size = n_embedding // n_heads
n_layers = 8 # transformer blocks stacked, depth helps small models more than width
hidden_size = 1024 # feed forward width, swiglu uses 8/3 * n_embedding
learning_rate = 3e-3 # peak, shared by muon and adamw
warmup_steps = 200 # ramp up from 0
cooldown_fraction = 0.3 # last 30% of the steps decay lr linearly to 0
eval_interval = 1000 # steps between validation runs
device = "mps" # m2 gpu, float32 is fastest here (bf16 and fp16 measured slower)
temperature = 0.8 # below 1 sharpens the guesses when generating
top_k = 40 # only sample from the 40 most likely next tokens, cuts off junk
memory_fraction = 0.3 # gpu memory cap, about 3.8 gb of 16, training needs 3.5, other apps keep the rest
gpu_pause = 0.02 # seconds the gpu rests after each step so the screen stays smooth, 0 for full speed


# data (bpe tokenizer trained once, every text file encoded once to uint16)
## tokenizer
def train_tokenizer():
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False) # bytes, any text works
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<|endoftext|>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tokenizer.train(["tinystories-test.txt"], trainer) # 21 mb is plenty to find the most common merges
    tokenizer.save("tokenizer.json")

## encode text file to token file, in chunks so 2.2 gb never sits in ram
def encode_file(text_path, token_path):
    tokenizer = Tokenizer.from_file("tokenizer.json")
    with open(text_path) as text_file, open(token_path + ".tmp", "wb") as token_file:
        while lines := text_file.readlines(16_000_000): # about 16 mb per chunk
            for encoding in tokenizer.encode_batch_fast(lines): # rust, all cores
                np.array(encoding.ids, dtype=np.uint16).tofile(token_file)
    os.replace(token_path + ".tmp", token_path) # only complete files count

if not os.path.exists("tokenizer.json"):
    train_tokenizer()
for text_path, token_path in [("tinystories-test.txt", "val.bin"), ("tinystories-train.txt", "train.bin")]:
    if not os.path.exists(token_path):
        print(f"encoding {text_path}, once")
        encode_file(text_path, token_path)

tokenizer = Tokenizer.from_file("tokenizer.json")
end_token = tokenizer.token_to_id("<|endoftext|>") # between stories
train_data = np.memmap("train.bin", dtype=np.uint16, mode="r") # reads from disk on demand
val_data = np.memmap("val.bin", dtype=np.uint16, mode="r")
max_steps = len(train_data) // (batch_size * accumulation_steps * block_size) # one pass over train data

def get_batch(data):
    starts = np.random.randint(0, len(data) - block_size - 1, batch_size) # random start positions, every token gets trained
    chunks = data[starts[:, None] + np.arange(block_size + 1)] # one gather, one extra token for the target
    chunks = torch.from_numpy(chunks.astype(np.int64)).to(device)
    return chunks[:, :-1], chunks[:, 1:] # x inputs, y next token for each x


# model (llama style: rope, rms norm, swiglu, qk norm, tied embeddings, no biases)
## rope (rotates query and key by position, attention then sees relative distance)
def rope_tables():
    speeds = 1 / 10000 ** (torch.arange(0, head_size, 2) / head_size) # one rotation speed per number pair
    angles = torch.outer(torch.arange(block_size), speeds) # (block_size, head_size / 2)
    return angles.cos(), angles.sin()

def rotate(x, cos, sin):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(n_embedding, 3 * n_embedding, bias=False) # query, key, value for every head in one matmul
        self.out = nn.Linear(n_embedding, n_embedding, bias=False) # mixes heads

    def forward(self, x, cos, sin):
        batch, length, _ = x.shape
        query, key, value = self.qkv(x).view(batch, length, 3, n_heads, head_size).transpose(1, 3).unbind(2) # each (batch, heads, length, head_size)
        query, key = F.rms_norm(query, (head_size,)), F.rms_norm(key, (head_size,)) # qk norm, scores stay small
        query, key = rotate(query, cos, sin), rotate(key, cos, sin)
        out = F.scaled_dot_product_attention(query, key, value, is_causal=True) # scores, mask, softmax, mix in one kernel
        return self.out(out.transpose(1, 2).reshape(batch, length, n_embedding)) # glue heads, mix

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up = nn.Linear(n_embedding, 2 * hidden_size, bias=False) # gate and up in one matmul
        self.down = nn.Linear(hidden_size, n_embedding, bias=False)

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up) # swiglu, gate decides how much of up passes

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.RMSNorm(n_embedding)
        self.attention = Attention()
        self.norm2 = nn.RMSNorm(n_embedding)
        self.feed_forward = FeedForward()

    def forward(self, x, cos, sin):
        x = x + self.attention(self.norm1(x), cos, sin) # residual, keep x and add update
        return x + self.feed_forward(self.norm2(x))

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_table = nn.Embedding(vocab_size, n_embedding)
        nn.init.normal_(self.token_table.weight, std=0.02) # small, it is also the output layer
        self.blocks = nn.ModuleList([Block() for _ in range(n_layers)])
        for block in self.blocks: # every block starts as a no-op and learns its update, trains faster
            nn.init.zeros_(block.attention.out.weight)
            nn.init.zeros_(block.feed_forward.down.weight)
        self.norm_final = nn.RMSNorm(n_embedding)
        cos, sin = rope_tables()
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x):
        length = x.shape[1]
        x = self.token_table(x)
        for block in self.blocks:
            x = block(x, self.cos[:length], self.sin[:length])
        return self.norm_final(x) @ self.token_table.weight.T # tied, the token table also scores every next token

def loss_of(model, x, y):
    return F.cross_entropy(model(x).view(-1, vocab_size), y.reshape(-1))


# optimizer (muon for block matrices, adamw for token table and norms)
def make_optimizers(model):
    matrices = [weights for weights in model.blocks.parameters() if weights.dim() == 2]
    others = [model.token_table.weight, model.norm_final.weight] + [weights for weights in model.blocks.parameters() if weights.dim() < 2]
    muon = torch.optim.Muon(matrices, lr=learning_rate, adjust_lr_fn="match_rms_adamw") # orthogonalized momentum, same lr scale as adamw
    adamw = torch.optim.AdamW(others, lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.0, fused=True)
    return [muon, adamw]

def learning_rate_scale(step):
    return min((step + 1) / warmup_steps, 1, (max_steps - step) / (cooldown_fraction * max_steps)) # warmup, flat, cooldown

def set_learning_rate(optimizers, scale):
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["lr"] = learning_rate * scale


# training loop (batch, forward, loss, backward, update, val loss, save)
def validation_loss(model):
    with torch.no_grad():
        return sum(loss_of(model, *get_batch(val_data)).item() for _ in range(40)) / 40 # 40 x 32 chunks

def train():
    os.nice(10) # lower cpu priority, the apps you work in come first
    torch.mps.set_per_process_memory_fraction(memory_fraction)
    model = GPT().to(device)
    compiled = torch.compile(model) # fuses small gpu ops, measured 1114 -> 605 ms per forward and backward on the m2 pro
    optimizers = make_optimizers(model)
    print(f"{sum(weights.numel() for weights in model.parameters()) / 1e6:.1f}M parameters, {max_steps} steps, ctrl+c stops and keeps the weights")
    print("compiling, first progress line after 100 steps, about 2 min")

    train_losses = [] # [step, loss] pairs, visualizer draws them
    validation_losses = []
    last_print = time.time()
    try:
        for step in range(max_steps):
            set_learning_rate(optimizers, learning_rate_scale(step))
            for _ in range(accumulation_steps): # gradients add up over the passes
                loss = loss_of(compiled, *get_batch(train_data))
                (loss / accumulation_steps).backward()
            for optimizer in optimizers:
                optimizer.step()
                optimizer.zero_grad()
            if gpu_pause:
                torch.mps.synchronize() # wait for this step, then leave the gpu free for the screen
                time.sleep(gpu_pause)

            if step % eval_interval == 0:
                validation_losses.append([step, validation_loss(compiled)])
                print(f"validation loss {validation_losses[-1][1]:.3f}")

            if (step + 1) % 100 == 0: # .item() waits for the gpu, so only now and then, progress and checkpoint
                train_losses.append([step, loss.item()])
                step_time = (time.time() - last_print) / 100 # measured on this mac, the first one includes the compile
                last_print = time.time()
                hours_left = step_time * (max_steps - step) / 3600
                print(f"step {step + 1}/{max_steps}  loss {train_losses[-1][1]:.3f}  {step_time * 1000:.0f} ms/step  {hours_left:.1f} h left")
                save(model, step, train_losses, validation_losses)

        validation_losses.append([max_steps, validation_loss(compiled)])
        print(f"done, final validation loss {validation_losses[-1][1]:.3f}")
    except KeyboardInterrupt:
        print(f"stopped at step {step}, weights saved to ai-gpt.pt")
    save(model, step, train_losses, validation_losses)

def save(model, step, train_losses, validation_losses):
    checkpoint = {"model": model.state_dict(), "step": step, "max_steps": max_steps,
                  "train_losses": train_losses, "validation_losses": validation_losses}
    torch.save(checkpoint, "ai-gpt.tmp")
    os.replace("ai-gpt.tmp", "ai-gpt.pt") # swap in one go, visualizer never reads half a file

def load():
    checkpoint = torch.load("ai-gpt.pt")
    model = GPT().to(device)
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint


# generate (prompt, predict next token, sample, append, stop at story end)
@torch.no_grad()
def generate(model, prompt, n_tokens):
    numbers = torch.tensor([[end_token] + tokenizer.encode(prompt).ids], device=device) # end token first, like a new story in training
    for _ in range(n_tokens):
        logits = model(numbers[:, -block_size:])[:, -1] / temperature # last block_size tokens, last row predicts next token
        top_logits, top_numbers = logits.topk(top_k)
        next_number = top_numbers.gather(1, torch.multinomial(F.softmax(top_logits, dim=-1), 1)) # pick among the top k
        if next_number.item() == end_token:
            break
        numbers = torch.cat([numbers, next_number], dim=1)
    return tokenizer.decode(numbers[0].tolist())


# run (python gpt.py trains, python gpt.py "Once upon a time" generates)
if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(generate(load()[0], sys.argv[1], 400))
    else:
        train()
