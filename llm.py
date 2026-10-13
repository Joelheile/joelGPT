import numpy as np
# data (text to numbers, train/val split, batches)
## input
file = open("tinystories-test.txt")
text=file.read()

## tokenizer
characters = list(text)

unique_characters = set(characters)
sorted_characters = sorted(unique_characters)
character_length = len(sorted_characters)

vocab_size= len(unique_characters)
print("vocab size",vocab_size)

charToNumber = {}
numberToChar = {}

for i, character in enumerate(sorted_characters):
    charToNumber[character] = i
    numberToChar[i] = character
print(charToNumber)


def encode(characters):
    encoded_chars = []
    for c in characters:
        encoded_chars.append(charToNumber[c])
    return encoded_chars

def decode(encoded_chars):
    decoded_text = []

    for e in encoded_chars:
        decoded_text.append(numberToChar[e])
    return "".join(decoded_text)



## train/val split
data = np.array(encode(characters))

train_split = len(data) * 0.9

train_data = data[:int(train_split)]
val_data = data[int(train_split):]

## batching
batch_size = 4
block_size = 8

def get_batch(split): # split is either train or val data; train on random batch in text
    starts = np.random.randint(0, len(split) - block_size, batch_size) # picks [block_size] random positions in text

    result_x = []
    result_y = []
    for i in starts:
        x = split[i:i + block_size]  # cuts section with block_size numbers starting at i
        y = split[i+1:i+1 + block_size] # one more as we need next character for input into model

        result_x.append(x)
        result_y.append(y)

    result_x = np.stack(result_x, axis=0) # turns each list into table with batch_size rows and block_size columns (4,8)
    result_y = np.stack(result_y, axis=0)

    return result_x, result_y

x, y = get_batch(train_data)
print(x.shape, y.shape)
print(x[0], y[0])


# embeddings (token table + position table, look up rows)
n_embd=16

token_table=np.random.randn(vocab_size, n_embd) * 0.02 # else values get too big and training unstable
position_table = np.random.randn(block_size, n_embd) * 0.02

def embed(x):
    positions = np.arange(block_size) # return spaced values in interval
    return token_table[x] + position_table[positions] # merges token and position into one vector

print(token_table.shape)


# attention (causal mask, softmax over scores, multi-head)
emb = embed(x)

## query (what am I looking for), key (what I contain; others compare their query), value (what I hand over if picked)

## softmax (each row of scores, positive weights that add up to 1)
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True)) # minus max to prevent overflow
    row_sums = np.sum(exp_x, axis=-1, keepdims=True) # one sum per row
    return exp_x / row_sums

# attention
def attention(emb, W_q, W_k, W_v):
    q = emb @ W_q # @ is matrix multiplication
    k = emb @ W_k
    v = emb @ W_v

    k_swapped = np.swapaxes(k,1,2) # switch index 1 and 2 (second and third element)
    scores = q @ k_swapped # result is how much the query looks at the key
    scores = scores / np.sqrt(q.shape[-1]) # keep result small so softmax doesnt become extreme, size of one head

    ## mask to make sure characters only look at previous, not at next
    mask = np.tril(np.ones(block_size))
    scores = np.where(mask == 0, -np.inf, scores) # condition, if_true, if_false

    weights = softmax(scores)

    output = weights @ v # multiply softmax weights with value matrix to calculate information on how much each character looks at other characters
    return output

n_head = 4 # number of heads
head_size = n_embd // n_head # numbers per head

## empty lists of q,k,v matrices
W_qs = []
W_ks = []
W_vs = []

W_o = np.random.randn(n_embd, n_embd) * 0.02 # mixes heads together



for i in range(n_head): # setup weights per head
     W_qs.append(np.random.randn(n_embd, head_size) * 0.02)
     W_ks.append(np.random.randn(n_embd, head_size) * 0.02)
     W_vs.append(np.random.randn(n_embd, head_size) * 0.02)

def multi_head_attention(emb):
    head_outputs = []

    for h in range (n_head): # run attention for every head
        result = attention(emb, W_qs[h], W_ks[h], W_vs[h])
        head_outputs.append(result) # collect head result

    result = np.concatenate(head_outputs, axis=-1) @ W_o # glue heads with concatenate, then multiply with output matrix
    return result


output= multi_head_attention(emb)


# transformer block (layer normalization, attention, feed forward, residuals)

gamma = np.ones(n_embd)
beta = np.zeros(n_embd)

def layer_normalization(x):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)

    normalized=(x-mean) / np.sqrt(variance + 1e-5) # transform numbers between 1 and 0
    result = gamma * normalized + beta
    return result

ln = layer_normalization(emb)






# output (final layer norm, linear layer to vocab_size scores)



# loss (softmax + cross-entropy)

# backward pass (gradient for every layer, gradient check)

# optimizer (Adam, update weights)

# training loop (batch, forward, loss, backward, update, val loss, save weights)

# generate (prompt, predict next char, sample, append, repeat)
