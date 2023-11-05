import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, GRU, Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import re
import gzip
import tensorflow as tf


data_text = []
sacred_texts = ['ac.txt.gz',
                'arp.txt.gz',
                'chinese_buddhism.txt.gz',
                'csj.txt.gz',
                'ebm.txt.gz',
                'mom.txt.gz',
                'salt.txt.gz',
                'twi.txt.gz',
                'yaq.txt.gz'
                ]
for text in sacred_texts:
    with gzip.open(f'data/{text}','rt') as f:
        cleaned = []
        lines = f.readlines()
        for line in lines[:1000]:
            line = re.sub(r'\([^)]*\)', '', line)
            line = re.sub(r'\[[^\]]*\]', '', line)
            line = line.strip()
            cleaned.append(line)
        data_text = ' '.join(cleaned)

# with open('data/bible.txt') as f:
#     lines = f.readlines()
#     cleaned = []
#     for line in lines[:100]:
#         line = line.split(' ')
#         line = ' '.join(line[1:])
#         cleaned.append(line)
#     data_text = ' '.join(cleaned)

def text_cleaner(text):
    # lower case text
    newString = text.lower()
    newString = re.sub(r"'s\b","",newString)
    # remove punctuations
    newString = re.sub("[^a-zA-Z]", " ", newString) 
    long_words=[]
    for i in newString.split():
        if len(i)>=3:                  
            long_words.append(i)
    return (" ".join(long_words)).strip()

data_new = text_cleaner(data_text)

def create_seq(text):
    length = 30
    sequences = list()
    for i in range(length, len(text)):
        # select sequence of tokens
        seq = text[i-length:i+1]
        sequences.append(seq)
    print('Total Sequences: %d' % len(sequences))
    return sequences

sequences = create_seq(data_new)

chars = sorted(list(set(data_new)))
mapping = dict((c, i) for i, c in enumerate(chars))

def encode_seq(seq):
    sequences = list()
    for line in seq:
        # integer encode line
        encoded_seq = [mapping[char] for char in line]
        # store
        sequences.append(encoded_seq)
    return sequences

# encode the sequences
sequences = encode_seq(sequences)

vocab = len(mapping)
sequences = np.array(sequences)
# create X and y
X, y = sequences[:,:-1], sequences[:,-1]
# one hot encode y
y = to_categorical(y, num_classes=vocab)
# create train and validation sets
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

print('Train shape:', X_tr.shape, 'Val shape:', X_val.shape)

def build_model():
    model = Sequential()
    model.add(Embedding(vocab, 50, input_length=30, trainable=True))
    model.add(GRU(150, recurrent_dropout=0.1, dropout=0.1))
    model.add(Dense(vocab, activation='softmax'))
    print(model.summary())
    model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')
    model.fit(X_tr, y_tr, epochs=100, verbose=1, validation_data=(X_val, y_val))
    model.save('./sacred_1')

# build_model()   
model = tf.keras.models.load_model('./sacred_1')

def generate_seq(model, mapping, seq_length, seed_text, n_chars):
    in_text = seed_text
    for _ in range(n_chars):
        encoded = [mapping[char] for char in in_text]
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        yhat = model.predict(encoded, verbose=0)
        out_char = ''
        y_prediction_classes = []
        for x in yhat:
            y_prediction_classes.append(np.argmax(x))
        y_ANN_prediction_classes_arr = np.array(y_prediction_classes)
        for char, index in mapping.items():
            if index == y_ANN_prediction_classes_arr:
                out_char = char
                break
        in_text += char
    return in_text

inp = 'the world'
print(len(inp))
print(generate_seq(model, mapping, 30, inp.lower(), 100))

def generate_response(input_text):
    return generate_seq(model, mapping, 30, input_text.lower(), 300)

while True:
    user_input = input("You: ")   
    if user_input.lower() == "exit":
        print("Goodbye!")
        break
    response = generate_response(user_input)
    print("Model: " + response)