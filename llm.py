import numpy as np
# data
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

def get_batch(split):
    batch_size = 4
    block_size = 8
    starts = np.random.randint(0, len(split) - block_size, batch_size)

    result_x = []
    result_y = []
    for i in starts:
        x = split[i:i + block_size]
        y = split[i+1:i+1 + block_size]

        result_x.append(x)
        result_y.append(y)

    result_x = np.stack(result_x, axis=0)
    result_y = np.stack(result_y, axis=0)

    return result_x, result_y

x, y = get_batch(train_data)
print(x.shape, y.shape)
print(x[0], y[0])





## embedding layer

# model forward pass
## positional encoding
## layer normalization
## causal multi-headed attention
## feed forward neural network
## linearization
## softmax

# training
## calculating cross-entropy loss
## backward pass
## gradient check
## optimizer
## training loop

# testing and optimizing
