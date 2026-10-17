# joelGPT

A few days ago a tweet asked what the most advanced technical concept is that you can actually explain. I tried LLMs in my head and got stuck pretty fast. So after watching [GreenCode](https://www.youtube.com/watch?v=s9w3gtgvNSU) build his own LLM, I built my own GPT in NumPy. The full story, with the math and a lot of pictures, is in my post [LLMs from first principles](https://joels.tech/llm-first-principles).

Everything here trains on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), short stories written with the vocabulary of a 3 to 4 year old. Even tiny models learn to write little stories with it, which is why I picked it.

## What's in here

`joel_gpt.py` is the one I wrote myself. Embeddings, attention, LayerNorm, the backward pass and Adam, all in plain NumPy. It reads one letter at a time and has about 0.8 million weights. Claude was my tutor for this one: it gave me hints and explained things, and I wrote the code. Backprop was the part where I needed the most help.

`joel_gpt_advanced.py` is my model with a BPE tokenizer, all 2.2 GB of stories and a longer context. AI made these changes on top of my code.

`ai_gpt.py` is a bigger PyTorch model that AI wrote for me, with the newer tricks people use to train small models fast. It runs on the Mac GPU and the sources are at the top of the file.

AI also wrote `visualize.py`, which shows every array inside the model while it trains.

## What they write

After 30 minutes of training, mine writes this:

```
Once upon a time, in a small house, there lived a tall girl named Sue. Sue wanted to help her. Sue loved to go home back inside, time to go home in you, and they always shared the meess.
```

It spells most words right and keeps the name Sue, but "meess" isn't a word and the sentence loses the plot halfway through.

The AI model after 7 hours:

```
Once upon a time, there was a little boy named Tim. Tim loved to draw with his pencil. One day, he went outside to draw a big sun on the ground. He was very happy.
```

That one came from an [earlier version of ai_gpt.py](https://github.com/Joelheile/joelGPT/blob/8ddc528/gpt.py) with about 16 million weights. The current file adds RoPE, Muon and value embeddings, and I haven't trained it yet.

## Run it

You need Python with NumPy. `ai_gpt.py` also needs PyTorch 2.14 for the Muon optimizer, and the BPE models need tokenizers.

```
pip install numpy tokenizers torch==2.14.1
```

The stories aren't in the repo, so download them into `data/` first:

```
curl -L -o data/tinystories-train.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
curl -L -o data/tinystories-test.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
```

Then train, and give it a prompt once it's done:

```
python joel_gpt.py
python joel_gpt.py "Once upon a time"
```

Training saves to `weights/` every 100 steps and Ctrl+C stops it without losing anything. On my M2 Pro, `joel_gpt.py` takes about 30 minutes, `ai_gpt.py` about 7 hours and `joel_gpt_advanced.py` about 24 hours. For the long ones I use `caffeinate -i` so the Mac doesn't fall asleep. Their first run also spends 5 to 10 minutes preparing the stories.

To watch it learn, start the visualizer in a second terminal and open http://localhost:8000:

```
python visualize.py joel_gpt
```

For my two models it shows the forward and backward pass. For `ai_gpt` it only shows the loss and a story.
