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

char_to_number = {}
number_to_char = {}

for i, character in enumerate(sorted_characters):
    char_to_number[character] = i
    number_to_char[i] = character
print(char_to_number)


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
data = np.array(encode(characters))

train_split = len(data) * 0.9

train_data = data[:int(train_split)]
val_data = data[int(train_split):]

## batching
batch_size = 4
block_size = 8

def get_batch(split): # split = train or val data
    starts = np.random.randint(0, len(split) - block_size, batch_size) # random start positions

    result_x = []
    result_y = []
    for i in starts:
        x = split[i:i + block_size] # block_size numbers from i
        y = split[i+1:i+1 + block_size] # shifted by one: next characters

        result_x.append(x)
        result_y.append(y)

    result_x = np.stack(result_x, axis=0) # list to table (4, 8)
    result_y = np.stack(result_y, axis=0)

    return result_x, result_y

x, y = get_batch(train_data)
print(x.shape, y.shape)
print(x[0], y[0])


# embeddings (token table + position table, look up rows)
n_embedding=16

token_table=np.random.randn(vocab_size, n_embedding) * 0.02 # small values, stable training
position_table = np.random.randn(block_size, n_embedding) * 0.02

def embed(x):
    positions = np.arange(block_size) # 0 to block_size-1
    return token_table[x] + position_table[positions] # token + position in one vector

print(token_table.shape)


# attention (causal mask, softmax over scores, multi-head)
embedded = embed(x)

## query (what I look for), key (what I contain), value (what I hand over)

## softmax (rows to weights that add up to 1)
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True)) # minus max, no overflow
    row_sums = np.sum(exp_x, axis=-1, keepdims=True) # one sum per row
    return exp_x / row_sums

# attention
def attention(embedded, query_weight, key_weight, value_weight):
    query = embedded @ query_weight # @ is matrix multiplication
    key = embedded @ key_weight
    value = embedded @ value_weight

    key_transposed = np.swapaxes(key,1,2) # swap last two axes
    scores = query @ key_transposed # how much query looks at key
    scores = scores / np.sqrt(query.shape[-1]) # keep small, softmax not extreme

    ## mask: only look back, not ahead
    mask = np.tril(np.ones(block_size))
    scores = np.where(mask == 0, -np.inf, scores) # condition, if_true, if_false

    weights = softmax(scores)

    output = weights @ value # weighted mix of values
    return output

n_heads = 4 # number of heads
head_size = n_embedding // n_heads # numbers per head

## weight lists, one per head
query_weights = []
key_weights = []
value_weights = []

output_weights = np.random.randn(n_embedding, n_embedding) * 0.02 # mixes heads



for i in range(n_heads): # setup weights per head
     query_weights.append(np.random.randn(n_embedding, head_size) * 0.02)
     key_weights.append(np.random.randn(n_embedding, head_size) * 0.02)
     value_weights.append(np.random.randn(n_embedding, head_size) * 0.02)

def multi_head_attention(embedded):
    head_outputs = []

    for h in range (n_heads): # run every head
        result = attention(embedded, query_weights[h], key_weights[h], value_weights[h])
        head_outputs.append(result) # collect result

    result = np.concatenate(head_outputs, axis=-1) @ output_weights # glue heads, mix
    return result


output= multi_head_attention(embedded)


# transformer block (layer normalization, attention, feed forward, residuals)

gamma1 = np.ones(n_embedding) # before attention
beta1 = np.zeros(n_embedding)
gamma2 = np.ones(n_embedding) # before feed forward
beta2 = np.zeros(n_embedding)

def layer_normalization(x, gamma, beta):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)

    normalized=(x-mean) / np.sqrt(variance + 1e-5) # spread with mean of 0, max 1
    result = gamma * normalized + beta
    return result

normalized_embedded = layer_normalization(embedded, gamma1, beta1)

## feed forward neural network with two layers, need random start points to nudge in different directions
W1 = np.random.randn(n_embedding, 4* n_embedding) * 0.02
b1 = np.zeros(4* n_embedding)
W2 = np.random.randn(4* n_embedding, n_embedding) * 0.02
b2 = np.zeros(n_embedding)

def feed_forward(x):
    hidden = x @ W1 + b1
    ReLU = np.maximum(0, hidden) # return function but 0 everything below 0
    return ReLU @ W2 + b2

print(feed_forward(normalized_embedded).shape)

def transformer_block(x):
    ## x gets added to result after multi head attention, values normalized before calculating

    x = x + multi_head_attention(layer_normalization(x, gamma1, beta1))
    x = x + feed_forward(layer_normalization(x, gamma2, beta2))

    return x

transformer_block(embedded)

# output (final layer normalization, linear layer to vocab_size scores)



# loss (softmax + cross-entropy)

# backward pass (gradient for every layer, gradient check)

# optimizer (Adam, update weights)

# training loop (batch, forward, loss, backward, update, val loss, save weights)

# generate (prompt, predict next char, sample, append, repeat)
