import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

# gpt: the advanced model. Same idea as micro_gpt.py: embed tokens, run attention blocks,
# score every next token, cross entropy loss, gradient steps. Each part below is rebuilt with
# what small-model research from 2021 to 2026 found trains fastest per hour on one machine.
# Every section says what it does, what changed from micro_gpt.py, why, and which source the
# choice comes from, so you can rebuild the model one step at a time.
#
# Sources, cited as [n] below:
# [1]  nanochat, Karpathy, github.com/karpathy/nanochat (gpt.py, scripts/base_train.py)
# [2]  modded-nanogpt speedrun, github.com/KellerJordan/modded-nanogpt
# [3]  parameter-golf baseline, github.com/openai/parameter-golf
# [4]  TinyStories, arxiv.org/abs/2305.07759
# [5]  Primer, squared relu, arxiv.org/abs/2109.08668
# [6]  Value Residual Learning, arxiv.org/abs/2410.17897
# [7]  Gemma 2 logit soft-capping, huggingface.co/blog/gemma2
# [8]  Muon is Scalable for LLM Training, arxiv.org/abs/2502.16982
# [9]  torch.optim.Muon docs, docs.pytorch.org/docs/2.14/generated/torch.optim.Muon.html
# [10] Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations, arxiv.org/abs/2405.18392
# [11] Understanding Warmup-Stable-Decay Learning Rates, arxiv.org/abs/2410.05192
# [12] Training Dynamics of the Cooldown Stage in WSD, arxiv.org/abs/2508.01483
# [13] OLMo 2, arxiv.org/abs/2501.00656
# [14] MobileLLM, arxiv.org/abs/2402.14905
# [15] Scaling Laws with Vocabulary, arxiv.org/abs/2407.13623
# [16] Chinchilla, arxiv.org/abs/2203.15556
# [17] Critical Batch Size Revisited, arxiv.org/abs/2505.23971
# [18] How Does Critical Batch Size Scale in Pre-training, arxiv.org/abs/2410.21676
# [19] Analysing the Impact of Sequence Composition on LM Pre-Training, arxiv.org/abs/2402.13991
# [20] SkyLadder, context window scheduling, arxiv.org/abs/2503.15450
# [21] SmolLM3, huggingface.co/blog/smollm3
#
# Considered and left out, with the reason:
# - Document masking, so attention never looks across <|endoftext|> [19]. It helps, but on the
#   mac gpu it needs an explicit mask, and [19] measured explicit masks at 32 to 52% slower.
# - Sliding window attention and growing the context during training [2] [20]. They pay off for
#   long contexts. 256 tokens already holds a whole story here.
# - Untied output layer [2] [1]. The speedrun untied, later retied, then untied at 2/3 of
#   training. [14] measured tying as almost free for small models. Tied stays, it saves 1.6M weights.
# - fp8, bf16 weights, custom kernels, smear, multi token prediction [2]. They need cuda or extra
#   machinery for small gains. bf16 measured slower than float32 on this m2 pro.
# - Muon variants (Polar Express, NorMuon) [2]. Not in torch. The built-in torch Muon [9] is used.


# settings
## size: 8 layers of 384 numbers, about 22M weights, 14M of them in the blocks.
## Why: [4] shows 1M to 33M weight models write coherent TinyStories, and [14] shows small
## models gain more from depth than width. A full pass is 559M tokens, about 40 tokens per
## block weight, above the 20 that [16] calls compute optimal, so the model is not too big for the data.
## Changed from micro_gpt.py: 91 chars become 4096 bpe tokens, 64 chars of context become 256
## tokens (about 1000 chars, a whole story), 4 layers of 128 become 8 layers of 384.
vocab_size = 4096 # bpe tokens, about 3.8 chars each. [15]: small models want small vocabularies
batch_size = 32 # chunks per forward and backward pass
accumulation_steps = 2 # passes per update, 2 x 32 = 64 chunks, half the gpu memory of one 64 pass
block_size = 256 # tokens per chunk, about 1000 chars, most stories fit
n_embedding = 384 # numbers per token vector
n_heads = 6 # attention heads, 64 numbers each
head_size = n_embedding // n_heads
n_layers = 8 # transformer blocks stacked, depth helps small models more than width [14]
hidden_size = 4 * n_embedding # feed forward width, 1536. relu² uses 2 matrices, so this matches swiglu's 3 matrices of 1024
value_embedding_layers = range(1, n_layers, 2) # layers 1, 3, 5, 7 get a value embedding, like [1]
logit_softcap = 15 # output scores stay between -15 and 15, the value [2] and [1] settled on
## schedule: 16k tokens per update. [17] and [18] find small batches use tokens most efficiently
## early in training, and 64 x 256 is small. Changed from micro_gpt.py: one fixed lr of 1e-3 with
## adam becomes warmup, flat, cooldown with muon plus adamw.
learning_rate = 3e-3 # peak, shared by muon and adamw
warmup_steps = 200 # ramp up from 0, avoids huge first updates while adam's averages are empty
cooldown_fraction = 0.45 # last 45% of the steps decay lr linearly to 0, [2] tuned 0.45, [10] [11] [12] explain why
eval_interval = 1000 # steps between validation runs
## machine: settings for the m2 pro while you keep working, not part of the model
device = "mps" # m2 gpu, float32 is fastest here (bf16 and fp16 measured slower)
memory_fraction = 0.3 # gpu memory cap, about 3.8 gb of 16, training needs 3.5, other apps keep the rest
gpu_pause = 0.02 # seconds the gpu rests after each step so the screen stays smooth, 0 for full speed
## generation
temperature = 0.8 # below 1 sharpens the guesses when generating
top_k = 40 # only sample from the 40 most likely next tokens, cuts off junk


# data (bpe tokenizer trained once, every text file encoded once to uint16)
## What: byte pair encoding starts from single bytes and keeps merging the most common neighbor
## pair into a new token until there are 4096. " the", "Once" and " Lily" each become one token.
## Changed from micro_gpt.py: one number per char becomes one number per common chunk. Each
## step now covers about 3.8x more text, and the model does not have to learn spelling.
## Why 4096: [15] finds the best vocabulary grows with model size. Small TinyStories models in the
## wild use 4096 too, and a bigger vocabulary would make the tied output layer most of the model.
def train_tokenizer():
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False) # bytes, any text works
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<|endoftext|>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tokenizer.train(["data/tinystories-test.txt"], trainer) # 21 mb is plenty to find the most common merges
    tokenizer.save("data/gpt-tokenizer.json")

## encode text file to token file, in chunks so 2.2 gb never sits in ram
def encode_file(text_path, token_path):
    tokenizer = Tokenizer.from_file("data/gpt-tokenizer.json")
    with open(text_path) as text_file, open(token_path + ".tmp", "wb") as token_file:
        while lines := text_file.readlines(16_000_000): # about 16 mb per chunk
            for encoding in tokenizer.encode_batch_fast(lines): # rust, all cores
                np.array(encoding.ids, dtype=np.uint16).tofile(token_file)
    os.replace(token_path + ".tmp", token_path) # only complete files count

if not os.path.exists("data/gpt-tokenizer.json"):
    train_tokenizer()
for text_path, token_path in [("data/tinystories-test.txt", "data/gpt-val.bin"), ("data/tinystories-train.txt", "data/gpt-train.bin")]:
    if not os.path.exists(token_path):
        print(f"encoding {text_path}, once")
        encode_file(text_path, token_path)

tokenizer = Tokenizer.from_file("data/gpt-tokenizer.json")
end_token = tokenizer.token_to_id("<|endoftext|>") # between stories
train_data = np.memmap("data/gpt-train.bin", dtype=np.uint16, mode="r") # reads from disk on demand
val_data = np.memmap("data/gpt-val.bin", dtype=np.uint16, mode="r")
max_steps = len(train_data) // (batch_size * accumulation_steps * block_size) # one pass over train data

## batches: random start positions, same as micro_gpt.py.
## Why not start every chunk at a story: [2] aligns chunks with <|endoftext|>, and this model tried
## it. It trained well on "Once upon a time" but broke on any prompt that does not start a story.
## Random starts teach the model to continue text from any point.
def get_batch(data):
    starts = np.random.randint(0, len(data) - block_size - 1, batch_size)
    chunks = data[starts[:, None] + np.arange(block_size + 1)] # one gather, one extra token for the target
    chunks = torch.from_numpy(chunks.astype(np.int64)).to(device)
    return chunks[:, :-1], chunks[:, 1:] # x inputs, y next token for each x


# model
## norm: rms norm without learned weights.
## Changed from micro_gpt.py: layer_norm subtracts the mean and has gamma and beta. rms norm only
## divides by the root mean square, which is cheaper and works as well. [1] and [3] drop the learned
## scale too, the matrices after the norm can learn any scale they need.
def rms_norm(x):
    return F.rms_norm(x, (x.size(-1),))

## rope: rotates query and key by their position, so attention sees relative distance.
## Changed from micro_gpt.py: position_table adds a learned vector per position to the token.
## rope has no weights, works for any distance up to block_size, and is what [1] [2] [3] [21] use.
def rope_tables():
    speeds = 1 / 10000 ** (torch.arange(0, head_size, 2) / head_size) # one rotation speed per number pair
    angles = torch.outer(torch.arange(block_size), speeds) # (block_size, head_size / 2)
    return angles.cos(), angles.sin()

def rotate(x, cos, sin):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

## attention: same queries, keys, values and causal mask as micro_gpt.py, with three changes.
## 1. One matmul makes query, key and value for all heads, and scaled_dot_product_attention does
##    scores, mask, softmax and mix in one gpu kernel. Same math as micro_gpt.py's loop over heads.
## 2. qk norm: query and key go through rms norm before the dot product, so scores cannot blow up.
##    [2] [1] [3] use it, [13] credits it for stable training.
## 3. value embedding: on layers 1, 3, 5, 7 a second token table adds straight into the values.
##    [6] found that giving later layers direct access to token values saves 10 to 20% of the
##    training data, and it was one of the bigger speedrun gains in [2]. It starts at zero, so
##    training begins as a plain transformer and learns how much to use it.
class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(n_embedding, 3 * n_embedding, bias=False) # query, key, value for every head in one matmul
        self.out = nn.Linear(n_embedding, n_embedding, bias=False) # mixes heads

    def forward(self, x, extra_value, cos, sin):
        batch, length, _ = x.shape
        query, key, value = self.qkv(x).view(batch, length, 3, n_heads, head_size).transpose(1, 3).unbind(2) # each (batch, heads, length, head_size)
        if extra_value is not None:
            value = value + extra_value.view(batch, length, n_heads, head_size).transpose(1, 2)
        query, key = rms_norm(query), rms_norm(key) # qk norm, scores stay small
        query, key = rotate(query, cos, sin), rotate(key, cos, sin)
        out = F.scaled_dot_product_attention(query, key, value, is_causal=True) # scores, mask, softmax, mix in one kernel
        return self.out(out.transpose(1, 2).reshape(batch, length, n_embedding)) # glue heads, mix

## feed forward: widen, relu squared, shrink.
## Changed from micro_gpt.py: relu becomes relu(x)², and the biases are gone.
## [5] found squaring relu one of the two changes that cut training cost the most. [2] [1] [3] all
## switched to it. It replaces swiglu, which the first version of this file used: same weights,
## one matrix fewer to multiply, and simpler to write.
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Linear(n_embedding, hidden_size, bias=False)
        self.down = nn.Linear(hidden_size, n_embedding, bias=False)

    def forward(self, x):
        return self.down(F.relu(self.up(x)).square())

## block: norm, attention, residual, norm, feed forward, residual. Same order as micro_gpt.py.
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = Attention()
        self.feed_forward = FeedForward()

    def forward(self, x, extra_value, cos, sin):
        x = x + self.attention(rms_norm(x), extra_value, cos, sin) # residual, keep x and add update
        return x + self.feed_forward(rms_norm(x))

## gpt: token table, blocks, final norm, scores for every next token.
## Changes from micro_gpt.py, top to bottom:
## - The embedding goes through rms norm before the first block, so every token enters with the same size [1].
## - x0 mixing: before every block, the stream is rescaled and a bit of the normed embedding x0 is
##   added back. Deep layers can still see which token they sit on. [1] calls these resid_lambdas and
##   x0_lambdas, [3] calls it resid_mix, [2] calls it embedding skip connections. Both scales are
##   learned, the start values are the ones from [1].
## - Tied embeddings: the token table is also the output layer, so vocab_weights is gone.
## - Zero init: the last matrix of every attention and feed forward starts at 0, so every block
##   starts as a no-op and learns its update. [2] [1] [3] all do this.
## - Logit softcap: scores pass through 15 * tanh(score / 15). Big scores bend smoothly toward 15
##   instead of growing forever, which keeps the model from getting overconfident early [7] [2] [1].
class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_table = nn.Embedding(vocab_size, n_embedding)
        nn.init.normal_(self.token_table.weight, std=0.02) # small, it is also the output layer
        self.value_embeddings = nn.ModuleDict({str(layer): nn.Embedding(vocab_size, n_embedding) for layer in value_embedding_layers})
        for table in self.value_embeddings.values():
            nn.init.zeros_(table.weight) # adds nothing at first, learns its part
        self.blocks = nn.ModuleList([Block() for _ in range(n_layers)])
        for block in self.blocks: # every block starts as a no-op and learns its update, trains faster
            nn.init.zeros_(block.attention.out.weight)
            nn.init.zeros_(block.feed_forward.down.weight)
        self.residual_scales = nn.Parameter(torch.linspace(1.15, 1.05, n_layers)) # stream scale per layer, start values from [1]
        self.embedding_scales = nn.Parameter(torch.linspace(0.2, 0.05, n_layers)) # x0 share per layer, more in early layers
        cos, sin = rope_tables()
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, tokens):
        length = tokens.shape[1]
        cos, sin = self.cos[:length], self.sin[:length]
        x = x0 = rms_norm(self.token_table(tokens))
        for layer, block in enumerate(self.blocks):
            x = self.residual_scales[layer] * x + self.embedding_scales[layer] * x0 # x0 mixing
            extra_value = self.value_embeddings[str(layer)](tokens) if str(layer) in self.value_embeddings else None
            x = block(x, extra_value, cos, sin)
        logits = rms_norm(x) @ self.token_table.weight.T # tied, the token table also scores every next token
        return logit_softcap * torch.tanh(logits / logit_softcap)

def loss_of(model, x, y):
    return F.cross_entropy(model(x).view(-1, vocab_size), y.reshape(-1))


# optimizer: muon for block matrices, adamw for everything else.
## Changed from micro_gpt.py: one adam for all weights becomes two optimizers.
## Muon keeps adam-style momentum, then orthogonalizes each matrix update with a few Newton-Schulz
## steps, so every direction of the matrix moves by a similar amount. [2] introduced it for small
## language models and measured about 1.5x better sample efficiency. [8] showed it scales to large
## models when it has weight decay and its update size is matched to adamw. torch.optim.Muon [9]
## does both: weight_decay=0.1 by default, and adjust_lr_fn="match_rms_adamw" lets muon share adamw's lr.
## Muon only works on 2d matrices inside the blocks. Token tables and the scalar mixes use adamw,
## with betas (0.9, 0.95) and no weight decay like [1] [2].
def make_optimizers(model):
    matrices = [weights for weights in model.blocks.parameters() if weights.dim() == 2]
    others = [model.token_table.weight, model.residual_scales, model.embedding_scales, *model.value_embeddings.parameters()]
    muon = torch.optim.Muon(matrices, lr=learning_rate, adjust_lr_fn="match_rms_adamw") # orthogonalized momentum, same lr scale as adamw
    adamw = torch.optim.AdamW(others, lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.0, fused=True)
    return [muon, adamw]

## lr schedule: warmup, flat, cooldown, called warmup-stable-decay.
## The flat part makes fast progress at a high lr, the cooldown at the end lets the weights settle
## into the bottom of the valley, and most of the loss drop shows up there [10] [11] [12].
## [10] found this matches or beats a cosine schedule. [2] tuned the cooldown to the last 45%.
def learning_rate_scale(step):
    return min((step + 1) / warmup_steps, 1, (max_steps - step) / (cooldown_fraction * max_steps)) # warmup, flat, cooldown

def set_learning_rate(optimizers, scale):
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["lr"] = learning_rate * scale


# training loop (batch, forward, loss, backward, update, val loss, save)
## Same steps as micro_gpt.py's train. New: gradients add up over 2 passes of 32 chunks before an
## update (half the gpu memory of one pass of 64), torch.compile fuses small gpu ops (1114 -> 605 ms
## per forward and backward measured on this m2 pro), and the loop sleeps briefly after each step so
## the mac stays usable.
def validation_loss(model):
    with torch.no_grad():
        return sum(loss_of(model, *get_batch(val_data)).item() for _ in range(40)) / 40 # 40 x 32 chunks

def train():
    os.nice(10) # lower cpu priority, the apps you work in come first
    torch.mps.set_per_process_memory_fraction(memory_fraction)
    model = GPT().to(device)
    compiled = torch.compile(model)
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
        print(f"stopped at step {step}, weights saved to weights/gpt.pt")
    save(model, step, train_losses, validation_losses)

def save(model, step, train_losses, validation_losses):
    checkpoint = {"model": model.state_dict(), "step": step, "max_steps": max_steps,
                  "train_losses": train_losses, "validation_losses": validation_losses}
    torch.save(checkpoint, "weights/gpt.tmp")
    os.replace("weights/gpt.tmp", "weights/gpt.pt") # swap in one go, visualizer never reads half a file

def load():
    checkpoint = torch.load("weights/gpt.pt")
    model = GPT().to(device)
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint


# generate (prompt, predict next token, sample, append, stop at story end)
## Changed from micro_gpt.py: the prompt starts with <|endoftext|> like a new story in training,
## temperature 0.8 sharpens the guesses, only the 40 most likely tokens can be picked, and
## generation stops when the model ends the story.
@torch.no_grad()
def generate(model, prompt, n_tokens):
    numbers = torch.tensor([[end_token] + tokenizer.encode(prompt).ids], device=device)
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
