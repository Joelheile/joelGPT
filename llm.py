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

# transformer block (layer norm, attention, feed forward, residuals)

# output (final layer norm, linear layer to vocab_size scores)

# loss (softmax + cross-entropy)

# backward pass (gradient for every layer, gradient check)

# optimizer (Adam, update weights)

# training loop (batch, forward, loss, backward, update, val loss, save weights)

# generate (prompt, predict next char, sample, append, repeat)
