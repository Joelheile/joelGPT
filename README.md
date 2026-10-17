# joelGPT

A few days ago a tweet asked what the most advanced technical concept is that you can actually explain. I tried LLMs in my head and got stuck pretty fast. So after watching [GreenCode](https://www.youtube.com/watch?v=s9w3gtgvNSU) build his own LLM, I built my own GPT in NumPy. The full story, with the math and a lot of pictures, is in my post [LLMs from first principles](https://joels.tech/llm-first-principles).

Everything here trains on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), short stories written with the vocabulary of a 3 to 4 year old. Even tiny models learn to write little stories with it, which is why I picked it.

## What's in here

`joel_gpt.py` is the one I wrote myself. Embeddings, attention, LayerNorm, the backward pass and Adam, all in plain NumPy. It reads one letter at a time and has about 0.8 million weights. I used Claude to guide me through building it, but read papers and watched explainer videos and articles on topics that I got stuck at (like understanding attention in depth and backpropagation)

`joel_gpt_advanced.py` is my model with a BPE tokenizer, all 2.2 GB of stories and a longer context. I used AI to show me what an optimized version of my NumPy model would look like.

`ai_gpt.py` is a bigger PyTorch model that AI wrote for me. I wanted to see how a model would look like if I let Claude decide how to build it from scratch in best version possible.

I used AI to build a small live visualization (`visualize.py`), which shows every array inside the model while it trains.


## How to run it yourself

You only need Python with NumPy. The AI written model `ai_gpt.py` also needs PyTorch 2.14 for the Muon optimizer, and the BPE models need tokenizers.

```
pip install numpy tokenizers torch==2.14.1
```

You also need to install the story text files to train it, which aren't in the repo and download them into `data/`:

```
curl -L -o data/tinystories-train.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
curl -L -o data/tinystories-test.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
```

Then you can train the model with 

```
python joel_gpt.py
```
and run it with a prompt of your choice once it's done
```
python joel_gpt.py "Once upon a time"
```

Training saves to `weights/` every 100 steps and Ctrl+C stops it without losing anything.
On my M2 Pro, `joel_gpt.py` takes about 30 minutes, `ai_gpt.py` about 7 hours and `joel_gpt_advanced.py` about 24 hours. For the long ones I use `caffeinate -i` so the Mac doesn't fall asleep. 

To watch it learn, start the visualizer in a second terminal and open http://localhost:8000:

```
python visualize.py joel_gpt
```

Hope you can learn with it and can use my examples and explainers to build your own model :)
