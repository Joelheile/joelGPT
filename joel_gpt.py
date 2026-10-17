import numpy as np
import os
import pickle
import sys
import time

# settings
batch_size = 32 # chunks per batch
block_size = 64 # chars per chunk
n_embedding = 128 # numbers per char vector
n_heads = 4 # attention heads
head_size = n_embedding // n_heads # numbers per head
n_layers = 4 # transformer blocks stacked


# data (text to numbers, train/val split, batches)
with open("data/tinystories-test.txt") as file:
    text = file.read()

## tokenizer (one number per char)
characters = sorted(set(text))
vocab_size = len(characters)
char_to_number = {c: i for i, c in enumerate(characters)}

def encode(text):
    return [char_to_number[c] for c in text]

def decode(numbers):
    return "".join(characters[n] for n in numbers)

## train/val split
data = np.array(encode(text))
train_split = int(len(data) * 0.9)
train_data = data[:train_split]
val_data = data[train_split:]

## batching
def get_batch(data): # data is train_data or val_data
    starts = np.random.randint(0, len(data) - block_size, batch_size) # random start positions
    x = np.stack([data[i:i + block_size] for i in starts]) # input, block_size chars from i
    y = np.stack([data[i + 1:i + 1 + block_size] for i in starts]) # target, next char for each x
    return x, y


# helpers
## flat, (batch, block, n) to (batch * block, n), one row per char
def flat(a):
    return a.reshape(-1, a.shape[-1])

## softmax (rows to weights that add up to 1)
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True)) # minus max, no overflow
    row_sums = np.sum(exp_x, axis=-1, keepdims=True) # one sum per row
    return exp_x / row_sums

## layer normalization (each char vector to mean 0, spread 1)
def normalize(x):
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.sqrt(np.var(x, axis=-1, keepdims=True) + 1e-5) # standard deviation, typical distance from mean
    return (x - mean) / std, std

def layer_norm(x, gamma, beta):
    normalized, std = normalize(x)
    return gamma * normalized + beta

def layer_norm_backward(d_out, x, gamma):
    normalized, std = normalize(flat(x))
    d_gamma = np.sum(d_out * normalized, axis=0) # sum rows
    d_beta = np.sum(d_out, axis=0)
    d_norm = d_out * gamma # back through gamma
    d_x = (d_norm
           - np.mean(d_norm, axis=-1, keepdims=True)
           - normalized * np.mean(d_norm * normalized, axis=-1, keepdims=True)
           ) / std # back through mean and std
    return d_x, d_gamma, d_beta


# embeddings (token table + position table, look up rows)
token_table = np.random.randn(vocab_size, n_embedding) * 0.02 # small values, stable training
position_table = np.random.randn(block_size, n_embedding) * 0.02

def embed(x):
    positions = np.arange(x.shape[1]) # 0 to length-1, prompt can be shorter than block_size
    return token_table[x] + position_table[positions] # token + position in one vector

def embed_backward(d_embedded, x):
    d_token_table = np.zeros_like(token_table)
    np.add.at(d_token_table, x.reshape(-1), d_embedded) # each char adds to its row
    d_position_table = np.sum(d_embedded.reshape(-1, block_size, n_embedding), axis=0) # sum over batch
    return d_token_table, d_position_table


# attention (causal mask, softmax over scores, multi-head)
## query (what I look for), key (what I contain), value (what I hand over), one table per layer and head
query_weights = np.random.randn(n_layers, n_heads, n_embedding, head_size) * 0.02
key_weights = np.random.randn(n_layers, n_heads, n_embedding, head_size) * 0.02
value_weights = np.random.randn(n_layers, n_heads, n_embedding, head_size) * 0.02
output_weights = np.random.randn(n_layers, n_embedding, n_embedding) * 0.02 # mixes heads

def attention(x, layer, h):
    query = x @ query_weights[layer, h]
    key = x @ key_weights[layer, h]
    value = x @ value_weights[layer, h]

    scores = query @ key.mT / np.sqrt(head_size) # how much query looks at key, .mT flips last two axes
    mask = np.tril(np.ones(x.shape[1])) # only look back, not ahead
    scores = np.where(mask == 0, -np.inf, scores)

    weights = softmax(scores)
    return weights @ value, (query, key, value, weights) # weighted mix of values

def multi_head_attention(x, layer):
    head_outputs = []
    head_caches = []
    for h in range(n_heads):
        out, cache = attention(x, layer, h)
        head_outputs.append(out)
        head_caches.append(cache)

    heads = np.concatenate(head_outputs, axis=-1) # glue heads
    return heads @ output_weights[layer], (x, heads, head_caches) # mix heads

def multi_head_attention_backward(d_out, cache, layer):
    x, heads, head_caches = cache
    d_output_weights = flat(heads).T @ d_out
    d_heads = (d_out @ output_weights[layer].T).reshape(heads.shape)

    d_x = np.zeros_like(x)
    d_query_weights = np.zeros_like(query_weights[layer])
    d_key_weights = np.zeros_like(key_weights[layer])
    d_value_weights = np.zeros_like(value_weights[layer])
    for h in range(n_heads):
        query, key, value, weights = head_caches[h]
        d_head = d_heads[:, :, h * head_size:(h + 1) * head_size] # this head's slice

        d_value = weights.mT @ d_head
        d_weights = d_head @ value.mT
        d_scores = weights * (d_weights - np.sum(d_weights * weights, axis=-1, keepdims=True)) # softmax backward
        d_scores = d_scores / np.sqrt(head_size)
        d_query = d_scores @ key
        d_key = d_scores.mT @ query

        d_query_weights[h] = flat(x).T @ flat(d_query)
        d_key_weights[h] = flat(x).T @ flat(d_key)
        d_value_weights[h] = flat(x).T @ flat(d_value)
        d_x += d_query @ query_weights[layer, h].T + d_key @ key_weights[layer, h].T + d_value @ value_weights[layer, h].T # every head adds

    return flat(d_x), d_query_weights, d_key_weights, d_value_weights, d_output_weights


# feed forward (widen, relu, shrink)
W1 = np.random.randn(n_layers, n_embedding, 4 * n_embedding) * 0.02 # one table per layer
b1 = np.zeros((n_layers, 4 * n_embedding))
W2 = np.random.randn(n_layers, 4 * n_embedding, n_embedding) * 0.02
b2 = np.zeros((n_layers, n_embedding))

def feed_forward(x, layer):
    hidden = x @ W1[layer] + b1[layer]
    relu = np.maximum(0, hidden) # negatives to 0
    return relu @ W2[layer] + b2[layer], (x, hidden, relu)

def feed_forward_backward(d_out, cache, layer):
    x, hidden, relu = cache
    d_W2 = flat(relu).T @ d_out
    d_b2 = np.sum(d_out, axis=0)
    d_hidden = (d_out @ W2[layer].T) * (flat(hidden) > 0) # relu, gradient only where hidden was positive
    d_W1 = flat(x).T @ d_hidden
    d_b1 = np.sum(d_hidden, axis=0)
    return d_hidden @ W1[layer].T, d_W1, d_b1, d_W2, d_b2


# transformer block (layer norm, attention, feed forward, residuals)
gamma1 = np.ones((n_layers, n_embedding)) # before attention
beta1 = np.zeros((n_layers, n_embedding))
gamma2 = np.ones((n_layers, n_embedding)) # before feed forward
beta2 = np.zeros((n_layers, n_embedding))

def transformer_block(x, layer):
    attention_out, attention_cache = multi_head_attention(layer_norm(x, gamma1[layer], beta1[layer]), layer)
    mid = x + attention_out # residual, keep x and add update
    feed_forward_out, feed_forward_cache = feed_forward(layer_norm(mid, gamma2[layer], beta2[layer]), layer)
    return mid + feed_forward_out, (x, mid, attention_cache, feed_forward_cache)


# output (final layer norm, linear layer to vocab_size scores)
gamma_final = np.ones(n_embedding)
beta_final = np.zeros(n_embedding)
vocab_weights = np.random.randn(n_embedding, vocab_size) * 0.02


# parameters (everything the model learns, same order as backward pass returns)
parameters = [token_table, position_table, gamma1, beta1,
              query_weights, key_weights, value_weights, output_weights,
              gamma2, beta2, W1, b1, W2, b2,
              gamma_final, beta_final, vocab_weights]


# forward pass (embed, blocks, final norm, logits, loss)
def predict(x): # no y needed, also used for generation
    block_out = embed(x)
    block_caches = []
    for layer in range(n_layers): # output of one block is input of next
        block_out, block_cache = transformer_block(block_out, layer)
        block_caches.append(block_cache)

    final_norm = layer_norm(block_out, gamma_final, beta_final)
    probabilities = flat(softmax(final_norm @ vocab_weights)) # one row per char
    return probabilities, final_norm, block_out, block_caches

def forward_pass(x, y):
    probabilities, final_norm, block_out, block_caches = predict(x)

    correct_probabilities = probabilities[np.arange(len(probabilities)), y.reshape(-1)] # probability of right char per row
    loss = -np.mean(np.log(correct_probabilities)) # how wrong predictions are
    return loss, (probabilities, final_norm, block_out, block_caches)


# backward pass (same steps as forward, last first)
def backward_pass(x, y, cache):
    probabilities, final_norm, block_out, block_caches = cache

    ## loss and output layer
    one_hot = np.eye(vocab_size)[y.reshape(-1)] # 1 at correct next char
    d_logits = (probabilities - one_hot) / len(probabilities)
    d_vocab_weights = flat(final_norm).T @ d_logits
    d_block_out, d_gamma_final, d_beta_final = layer_norm_backward(d_logits @ vocab_weights.T, block_out, gamma_final)

    ## blocks, last layer first, one slice per layer
    d_gamma1 = np.zeros_like(gamma1)
    d_beta1 = np.zeros_like(beta1)
    d_query_weights = np.zeros_like(query_weights)
    d_key_weights = np.zeros_like(key_weights)
    d_value_weights = np.zeros_like(value_weights)
    d_output_weights = np.zeros_like(output_weights)
    d_gamma2 = np.zeros_like(gamma2)
    d_beta2 = np.zeros_like(beta2)
    d_W1 = np.zeros_like(W1)
    d_b1 = np.zeros_like(b1)
    d_W2 = np.zeros_like(W2)
    d_b2 = np.zeros_like(b2)

    for layer in reversed(range(n_layers)):
        block_in, mid, attention_cache, feed_forward_cache = block_caches[layer]

        ### feed forward and layer norm 2
        d_feed_forward_in, d_W1[layer], d_b1[layer], d_W2[layer], d_b2[layer] = feed_forward_backward(d_block_out, feed_forward_cache, layer)
        d_norm2, d_gamma2[layer], d_beta2[layer] = layer_norm_backward(d_feed_forward_in, mid, gamma2[layer])
        d_mid = d_block_out + d_norm2 # residual, both paths add up

        ### attention and layer norm 1
        d_attention_in, d_query_weights[layer], d_key_weights[layer], d_value_weights[layer], d_output_weights[layer] = multi_head_attention_backward(d_mid, attention_cache, layer)
        d_norm1, d_gamma1[layer], d_beta1[layer] = layer_norm_backward(d_attention_in, block_in, gamma1[layer])
        d_block_out = d_mid + d_norm1 # residual, becomes d_block_out of the layer below

    ## embeddings
    d_token_table, d_position_table = embed_backward(d_block_out, x)

    return [d_token_table, d_position_table, d_gamma1, d_beta1,
            d_query_weights, d_key_weights, d_value_weights, d_output_weights,
            d_gamma2, d_beta2, d_W1, d_b1, d_W2, d_b2,
            d_gamma_final, d_beta_final, d_vocab_weights]


# optimizer (Adam, update weights)
learning_rate = 1e-3
optimizer_beta1 = 0.9 # how much m remembers
optimizer_beta2 = 0.999 # how much v remembers
epsilon = 1e-8 # prevents division by zero

m = [np.zeros_like(weights) for weights in parameters] # momentum, average gradient per weight
v = [np.zeros_like(weights) for weights in parameters] # average squared gradient per weight
step = 0

def adam_optimizer(gradients):
    global step
    step += 1
    for i, (weights, gradient) in enumerate(zip(parameters, gradients)):
        m[i] = optimizer_beta1 * m[i] + (1 - optimizer_beta1) * gradient
        v[i] = optimizer_beta2 * v[i] + (1 - optimizer_beta2) * gradient ** 2
        m_hat = m[i] / (1 - optimizer_beta1 ** step) # bias correction, m and v start at 0
        v_hat = v[i] / (1 - optimizer_beta2 ** step)
        weights -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon) # -= changes weights in place

# training loop (batch, forward, loss, backward, update, val loss, save weights)
max_steps = len(train_data) // (batch_size * block_size) # one pass over train data
eval_interval = 1000
eval_batches = 20 # val batches to average, one is too noisy

def validation_loss(): # forward only, no learning from val
    return np.mean([forward_pass(*get_batch(val_data))[0] for _ in range(eval_batches)])

def train():
    start = time.time()
    train_losses = [] # [step, loss] pairs, visualizer draws them
    validation_losses = []
    for step_number in range(max_steps):
        x, y = get_batch(train_data)
        loss, cache = forward_pass(x, y)
        adam_optimizer(backward_pass(x, y, cache))

        if step_number % eval_interval == 0:
            validation_losses.append([step_number, validation_loss()])
            print(f"validation loss {validation_losses[-1][1]:.3f}")

        if step_number % 100 == 0: # progress and checkpoint
            train_losses.append([step_number, loss])
            minutes_left = (time.time() - start) / (step_number + 1) * (max_steps - step_number) / 60
            print(f"step {step_number}/{max_steps}  loss {loss:.3f}  {minutes_left:.0f} min left")
            save(step_number, train_losses, validation_losses)

    save(max_steps, train_losses, validation_losses)

def save(step_number, train_losses, validation_losses):
    checkpoint = {"parameters": parameters, "step": step_number, "max_steps": max_steps,
                  "train_losses": train_losses, "validation_losses": validation_losses}
    with open("weights/joel_gpt.tmp", "wb") as file:
        pickle.dump(checkpoint, file)
    os.replace("weights/joel_gpt.tmp", "weights/joel_gpt.pkl") # swap in one go, visualizer never reads half a file

def load():
    with open("weights/joel_gpt.pkl", "rb") as file:
        checkpoint = pickle.load(file)
    for weights, saved in zip(parameters, checkpoint["parameters"]):
        weights[...] = saved # [...] writes into the existing table, like -= in adam
    return checkpoint


# generate (prompt, predict next char, sample, append, repeat)
def generate(prompt, n_chars):
    numbers = encode(prompt)
    for _ in range(n_chars):
        context = numbers[-block_size:] # last block_size chars, position table has no more rows
        probabilities = predict(np.array([context]))[0] # batch of one
        next_number = np.random.choice(vocab_size, p=probabilities[-1]) # last row predicts next char
        numbers.append(next_number)
    return decode(numbers)


# run (python joel_gpt.py trains, python joel_gpt.py "Once upon a time" generates)
if __name__ == "__main__":
    if len(sys.argv) > 1:
        load()
        print(generate(sys.argv[1], 500))
    else:
        train()
