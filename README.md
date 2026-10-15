# joelGPT

I built a GPT from scratch in numpy to understand how LLMs work. This repo has the code. The article [LLMs from first principles](https://joels.tech/llm-first-principles) explains every part with pictures, animations and the math.

There are three small language models, all trained on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset. Each one builds on the one before:

| Model         | File               | Built by           | Tokens          | Runs on   | Weights                     |
| ------------- | ------------------ | ------------------ | --------------- | --------- | --------------------------- |
| micro-gpt     | `micro_gpt.py`     | me, from scratch   | 91 characters   | numpy CPU | `weights/micro_gpt.pkl`     |
| micro-gpt-bpe | `micro_gpt_bpe.py` | AI, from micro-gpt | 1024 BPE tokens | numpy CPU | `weights/micro_gpt_bpe.pkl` |
| gpt           | `gpt.py`           | AI, research based | 4096 BPE tokens | Mac GPU   | `weights/gpt.pt`            |

- **micro-gpt** is the model I wrote by hand in numpy: embeddings, attention, layer norm, transformer blocks, the full backward pass and the Adam optimizer. It reads one character at a time and trains on the 21 MB test file.
- **micro-gpt-bpe** is the same code with three changes: a 1024-token BPE tokenizer, the full 2.2 GB train file, and a context of 128 tokens (about 400 characters) instead of 64 characters.
- **gpt** is a PyTorch model built from 2021 to 2026 research on fast small-model training. The comments in `gpt.py` explain every decision, what changed from micro-gpt and which paper or project it comes from, so you can rebuild it step by step.

## Why

I wanted to understand how an LLM works. [This GreenCode video](https://www.youtube.com/watch?v=vKdKtxpfvIc) got me started. The theory comes from the 3Blue1Brown and StatQuest videos on neural networks and transformers.

## How I built it

I wrote micro-gpt myself with videos and tutorials: the embedding layer, attention, multi-head attention, layer norm and the transformer block. For backpropagation I had some help from AI. Claude Opus was my trainer the whole way. It gave hints and explanations, and I wrote the model code.

AI wrote micro-gpt-bpe, gpt and the training visualizer (`visualize.py` and `visualizer/`).

## What gpt does differently

| Part         | micro-gpt                     | gpt                                                              |
| ------------ | ----------------------------- | ---------------------------------------------------------------- |
| Tokens       | 91 characters                 | 4096 BPE tokens, about 3.8 characters each                       |
| Context      | 64 characters                 | 256 tokens, about 1000 characters, a whole story                 |
| Size         | 4 layers of 128, 0.8M weights | 8 layers of 384, 22M weights                                     |
| Positions    | learned position table        | RoPE, rotates query and key by position                          |
| Norm         | layer norm with gamma, beta   | RMS norm without weights, also on query and key (QK norm)        |
| Feed forward | ReLU                          | ReLU squared                                                     |
| Extra paths  | none                          | value embeddings on odd layers, embedding mixed into every layer |
| Output       | separate vocab_weights        | tied to the token table, scores capped at 15                     |
| Init         | small random everywhere       | block output matrices start at zero                              |
| Optimizer    | Adam, fixed learning rate     | Muon for block matrices, AdamW for the rest                      |
| Schedule     | constant                      | warmup, flat, linear cooldown over the last 45%                  |
| Speed        | numpy loops                   | torch.compile on the Mac GPU                                     |

## Folders

| Path                                 | What it holds                                         |
| ------------------------------------ | ----------------------------------------------------- |
| `data/tinystories-*.txt`             | Raw stories, train and test                           |
| `data/micro_gpt_bpe-*`, `data/gpt-*` | Tokenizers and encoded data, created on the first run |
| `weights/`                           | Trained weights, one file per model                   |
| `visualize.py`, `visualizer/`        | Local web page that follows training live             |

## Setup

```
pip install torch==2.14.1 numpy tokenizers
```

The stories are not in the repo. Download them from Hugging Face into `data/`:

```
curl -L -o data/tinystories-train.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
curl -L -o data/tinystories-test.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
```

gpt needs PyTorch 2.14 for the Muon optimizer. The micro models only need numpy, and micro-gpt-bpe also needs tokenizers.

## Train and generate

Every model works the same way. `python <file>` trains one full pass and saves the weights every 100 steps. `python <file> "prompt"` generates a story. Ctrl+C stops training and keeps the weights. `caffeinate -i` keeps the Mac awake.

| Model         | Train                                   | Time on an M2 Pro |
| ------------- | --------------------------------------- | ----------------- |
| micro-gpt     | `python micro_gpt.py`                   | about 30 minutes  |
| micro-gpt-bpe | `caffeinate -i python micro_gpt_bpe.py` | about 24 hours    |
| gpt           | `caffeinate -i python gpt.py`           | about 7 hours     |

```
python gpt.py "Once upon a time"
```

The first run of micro-gpt-bpe or gpt trains its tokenizer and encodes the 2.2 GB train file, which takes about 5 to 10 minutes.

Training runs at low CPU priority, so you can keep working. gpt also caps its GPU memory at about 3.8 GB and pauses the GPU briefly after each step.

## Watch training

Start the visualizer in a second terminal with the model name, then open http://localhost:8000:

```
python visualize.py micro_gpt
python visualize.py micro_gpt_bpe
python visualize.py gpt
```

For the micro models, the page shows every array of the forward and backward pass. For gpt, it shows the loss curves and a generated story.
