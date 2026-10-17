# joelGPT

I use LLMs every day, but a while ago I noticed I couldn't actually explain how they work. So I built my own GPT from scratch in NumPy. This repo has the code. The full explanation, with the math and a lot of pictures, is in my post [LLMs from first principles](https://joels.tech/llm-first-principles).

All models are trained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), short stories written with the vocabulary of a 3 to 4 year old. It's small and simple, so even a tiny model learns to write real stories with it.

## The models

- **joel_gpt.py** is the one I wrote by hand. Embeddings, attention, LayerNorm, the transformer blocks, the whole backward pass and Adam, all in NumPy. It reads one character at a time and has about 0.8 million weights.
- **joel_gpt_advanced.py** is the same model with a BPE tokenizer (1,024 tokens), all 2.2 GB of stories and a longer context. AI built it on top of my code.
- **ai_gpt.py** is a bigger PyTorch model that AI wrote for me, based on recent research on training small models fast. It runs on the Mac GPU. The sources are at the top of the file.

Here is what my NumPy model writes after 30 minutes of training:

```
Once upon a time, in a small house, there lived a tall girl named Sue. Sue wanted to help her. Sue loved to go home back inside, time to go home in you, and they always shared the meess.
```

And here is the AI model after 7 hours:

```
Once upon a time, there was a little boy named Tim. Tim loved to draw with his pencil. One day, he went outside to draw a big sun on the ground. He was very happy.
```

Not bad for mine, but you can see why people train bigger models :)

These stories come from an [earlier version of ai_gpt.py](https://github.com/Joelheile/joelGPT/blob/8ddc528/gpt.py) with about 16 million weights. The current file has a few more tricks (RoPE, Muon, value embeddings) and I haven't trained it yet.

## How I built it

After watching [GreenCode](https://www.youtube.com/watch?v=s9w3gtgvNSU) build his own LLM, I wanted to do the same. The theory came from the 3Blue1Brown and StatQuest videos on neural networks and transformers.

I wrote joel_gpt.py myself. For backpropagation I had some help from AI. Claude was my tutor the whole way: it gave me hints and explanations, and I wrote the code. AI wrote the other two models and the training visualizer.

## Run it

You need Python with NumPy. ai_gpt.py also needs PyTorch 2.14 for the Muon optimizer, and the BPE models need tokenizers.

```
pip install numpy tokenizers torch==2.14.1
```

The stories aren't in the repo, so download them into `data/` first:

```
curl -L -o data/tinystories-train.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
curl -L -o data/tinystories-test.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
```

Then train a model, and give it a prompt once it's done:

```
python joel_gpt.py
python joel_gpt.py "Once upon a time"
```

Training saves the weights to `weights/` every 100 steps, and Ctrl+C stops it without losing them. On my M2 Pro, joel_gpt.py takes about 30 minutes, ai_gpt.py about 7 hours and joel_gpt_advanced.py about 24 hours. For the long ones, run them with `caffeinate -i` so the Mac doesn't fall asleep. Their first run also spends 5 to 10 minutes preparing the stories.

## Watch it learn

I really wanted to see what happens inside while it trains. Start the visualizer in a second terminal and open http://localhost:8000:

```
python visualize.py joel_gpt
```

For my two models it shows every array of the forward and backward pass. For ai_gpt it shows the loss and a story it writes.
