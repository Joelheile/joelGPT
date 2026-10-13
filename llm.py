import numpy as np

# settings
batch_size = 4 # chunks per batch
block_size = 8 # chars per chunk
n_embedding = 16 # numbers per char vector
n_heads = 4 # attention heads
head_size = n_embedding // n_heads # numbers per head


# data (text to numbers, train/val split, batches)
## input
with open("tinystories-test.txt") as file:
    text = file.read()

## tokenizer
sorted_characters = sorted(set(text))
vocab_size = len(sorted_characters)

char_to_number = {}
number_to_char = {}
for i, character in enumerate(sorted_characters):
    char_to_number[character] = i
    number_to_char[i] = character

def encode(characters):
    encoded_chars = []
    for c in characters:
        encoded_chars.append(char_to_number[c])
    return encoded_chars

def decode(encoded_chars):
    decoded_text = []
    for e in encoded_chars:
        decoded_text.append(number_to_char[e])
    return "".join(decoded_text)

## train/val split
data = np.array(encode(text))
train_split = int(len(data) * 0.9)
train_data = data[:train_split]
val_data = data[train_split:]

## batching
def get_batch(split): # split is train or val data
    starts = np.random.randint(0, len(split) - block_size, batch_size) # random start positions

    result_x = []
    result_y = []
    for i in starts:
        result_x.append(split[i:i + block_size]) # input, block_size chars from i
        result_y.append(split[i+1:i+1 + block_size]) # target, next char for each x

    return np.stack(result_x), np.stack(result_y) # lists to tables (4, 8)


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
    positions = np.arange(block_size) # 0 to block_size-1
    return token_table[x] + position_table[positions] # token + position in one vector

def embed_backward(d_embedded, x):
    d_token_table = np.zeros_like(token_table)
    np.add.at(d_token_table, x.reshape(-1), d_embedded) # each char adds to its row
    d_position_table = np.sum(d_embedded.reshape(-1, block_size, n_embedding), axis=0) # sum over batch
    return d_token_table, d_position_table


# attention (causal mask, softmax over scores, multi-head)
## query (what I look for), key (what I contain), value (what I hand over), one table per head
query_weights = np.random.randn(n_heads, n_embedding, head_size) * 0.02
key_weights = np.random.randn(n_heads, n_embedding, head_size) * 0.02
value_weights = np.random.randn(n_heads, n_embedding, head_size) * 0.02
output_weights = np.random.randn(n_embedding, n_embedding) * 0.02 # mixes heads

def attention(x, h): # h is head number
    query = x @ query_weights[h] # @ is matrix multiplication
    key = x @ key_weights[h]
    value = x @ value_weights[h]

    scores = query @ key.mT / np.sqrt(head_size) # how much query looks at key, .mT flips last two axes
    mask = np.tril(np.ones(block_size)) # only look back, not ahead
    scores = np.where(mask == 0, -np.inf, scores)

    weights = softmax(scores)
    return weights @ value, (query, key, value, weights) # weighted mix of values

def multi_head_attention(x):
    head_outputs = []
    head_caches = []
    for h in range(n_heads):
        out, cache = attention(x, h)
        head_outputs.append(out)
        head_caches.append(cache)

    heads = np.concatenate(head_outputs, axis=-1) # glue heads
    return heads @ output_weights, (x, heads, head_caches) # mix heads

def multi_head_attention_backward(d_out, cache):
    x, heads, head_caches = cache
    d_output_weights = flat(heads).T @ d_out
    d_heads = (d_out @ output_weights.T).reshape(heads.shape)

    d_x = np.zeros_like(x)
    d_query_weights = np.zeros_like(query_weights)
    d_key_weights = np.zeros_like(key_weights)
    d_value_weights = np.zeros_like(value_weights)
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
        d_x += d_query @ query_weights[h].T + d_key @ key_weights[h].T + d_value @ value_weights[h].T # every head adds

    return flat(d_x), d_query_weights, d_key_weights, d_value_weights, d_output_weights


# feed forward (widen, relu, shrink)
W1 = np.random.randn(n_embedding, 4 * n_embedding) * 0.02
b1 = np.zeros(4 * n_embedding)
W2 = np.random.randn(4 * n_embedding, n_embedding) * 0.02
b2 = np.zeros(n_embedding)

def feed_forward(x):
    hidden = x @ W1 + b1
    relu = np.maximum(0, hidden) # negatives to 0
    return relu @ W2 + b2, (x, hidden, relu)

def feed_forward_backward(d_out, cache):
    x, hidden, relu = cache
    d_W2 = flat(relu).T @ d_out
    d_b2 = np.sum(d_out, axis=0)
    d_hidden = (d_out @ W2.T) * (flat(hidden) > 0) # relu, gradient only where hidden was positive
    d_W1 = flat(x).T @ d_hidden
    d_b1 = np.sum(d_hidden, axis=0)
    return d_hidden @ W1.T, d_W1, d_b1, d_W2, d_b2


# transformer block (layer norm, attention, feed forward, residuals)
gamma1 = np.ones(n_embedding) # before attention
beta1 = np.zeros(n_embedding)
gamma2 = np.ones(n_embedding) # before feed forward
beta2 = np.zeros(n_embedding)

def transformer_block(x):
    attention_out, attention_cache = multi_head_attention(layer_norm(x, gamma1, beta1))
    mid = x + attention_out # residual, keep x and add update
    feed_forward_out, feed_forward_cache = feed_forward(layer_norm(mid, gamma2, beta2))
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


# forward pass (embed, block, final norm, logits, loss)
def forward_pass(x, y):
    block_out, block_cache = transformer_block(embed(x))
    final_norm = layer_norm(block_out, gamma_final, beta_final)
    probabilities = flat(softmax(final_norm @ vocab_weights)) # one row per char

    correct_probabilities = probabilities[np.arange(len(probabilities)), y.reshape(-1)] # probability of right char per row
    loss = -np.mean(np.log(correct_probabilities)) # how wrong predictions are
    return loss, (probabilities, final_norm, block_out, block_cache)


# backward pass (same steps as forward, last first)
def backward_pass(x, y, cache):
    probabilities, final_norm, block_out, block_cache = cache
    block_in, mid, attention_cache, feed_forward_cache = block_cache

    ## loss and output layer
    one_hot = np.eye(vocab_size)[y.reshape(-1)] # 1 at correct next char
    d_logits = (probabilities - one_hot) / len(probabilities)
    d_vocab_weights = flat(final_norm).T @ d_logits
    d_block_out, d_gamma_final, d_beta_final = layer_norm_backward(d_logits @ vocab_weights.T, block_out, gamma_final)

    ## feed forward and layer norm 2
    d_feed_forward_in, d_W1, d_b1, d_W2, d_b2 = feed_forward_backward(d_block_out, feed_forward_cache)
    d_norm2, d_gamma2, d_beta2 = layer_norm_backward(d_feed_forward_in, mid, gamma2)
    d_mid = d_block_out + d_norm2 # residual, both paths add up

    ## attention and layer norm 1
    d_attention_in, d_query_weights, d_key_weights, d_value_weights, d_output_weights = multi_head_attention_backward(d_mid, attention_cache)
    d_norm1, d_gamma1, d_beta1 = layer_norm_backward(d_attention_in, block_in, gamma1)
    d_embedded = d_mid + d_norm1 # residual, both paths add up

    ## embeddings
    d_token_table, d_position_table = embed_backward(d_embedded, x)

    return [d_token_table, d_position_table, d_gamma1, d_beta1,
            d_query_weights, d_key_weights, d_value_weights, d_output_weights,
            d_gamma2, d_beta2, d_W1, d_b1, d_W2, d_b2,
            d_gamma_final, d_beta_final, d_vocab_weights]


# gradient check (nudge one weight, compare loss change with computed gradient)
def gradient_check(weights, computed_grad, x, y):
    index = [] # random position, one number per dimension
    for size in weights.shape:
        index.append(np.random.randint(size))
    index = tuple(index)

    original = weights[index]
    weights[index] = original + 1e-5
    loss_plus = forward_pass(x, y)[0]
    weights[index] = original - 1e-5
    loss_minus = forward_pass(x, y)[0]
    weights[index] = original # put it back

    print("numerical", (loss_plus - loss_minus) / 2e-5, "computed", computed_grad[index])


# run one batch
x, y = get_batch(train_data) # x inputs, y correct next chars
loss, cache = forward_pass(x, y)
print("loss", loss)

gradients = backward_pass(x, y, cache)
for weights, gradient in zip(parameters, gradients): # pairs each weight with its gradient
    gradient_check(weights, gradient, x, y)


# optimizer (Adam, update weights)


# training loop (batch, forward, loss, backward, update, val loss, save weights)

# generate (prompt, predict next char, sample, append, repeat)
