import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, GRU, Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import re
import gzip
import tensorflow as tf
import nltk
from nltk.tokenize import sent_tokenize


data_text = ''
sacred_texts = [
                # 'ac.txt.gz', #
                # 'ajp.txt.gz',
                # 'arw.txt.gz',
                # 'aww.txt.gz',
                # 'cc.txt.gz',
                # 'ettt.txt.gz', # --
                'gork.txt.gz',
                # 'pr.txt.gz',
                # 'seil.txt.gz',
                # 'mom.txt.gz', #
                # 'salt.txt.gz',
                # 'wov.txt.gz',
                # 'yaq.txt.gz',
                # 'zfa.txt.gz'
                ]
for text in sacred_texts:
    try:
        with gzip.open(f'data/{text}','rt') as f:
            cleaned = []
            lines = f.readlines()
            for line in lines[5:600]:
                line = re.sub(r'\([^)]*\)', '', line)
                line = re.sub(r'\[[^\]]*\]', '', line)
                line = line.strip()
                cleaned.append(line)
            text_str = ' '.join(cleaned)
            data_text += ' '
            data_text += text_str
    except:
        print(f'{text} not found')

# with open('data/bible.txt') as f:
#     lines = f.readlines()
#     cleaned = []
#     stop = len(lines) // 6
#     for line in lines[:stop]:
#         line = line.split(' ')
#         line = ' '.join(line[1:])
#         cleaned.append(line)
#     text_str = ' '.join(cleaned)
#     data_text += ' '
#     data_text += text_str

def text_cleaner(text):
    # lower case text
    new_string = text.lower()
    # new_string = re.sub(r"'s\b","", new_string)
    # remove punctuations
    # new_string = re.sub("[^a-zA-Z]", " ", new_string) 
    new_string = re.sub("[^a-zA-Z0-9.!?,']", " ", new_string)
    long_words=[]
    for i in new_string.split():
        if len(i)>=3:                  
            long_words.append(i)
    return (" ".join(long_words)).strip()

data_new = text_cleaner(data_text)
with open('text_ex.txt', 'w') as k:
                k.write(data_new)

def create_seq(text):
    length = 30
    sequences = list()
    for i in range(length, len(text)):
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
        encoded_seq = [mapping[char] for char in line]
        sequences.append(encoded_seq)
    return sequences

sequences = encode_seq(sequences)

vocab = len(mapping)
print('vocab = ', len(mapping), 'chars = ', len(chars))
assert vocab == 42, f'vocab - len(mapping) should be equal to {len(chars)}'

sequences = np.array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]
y = to_categorical(y, num_classes=vocab)
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print('Train shape:', X_tr.shape, 'Val shape:', X_val.shape)

def build_model(name):
    model = Sequential()
    model.add(Embedding(vocab, 50, input_length=30, trainable=True))
    model.add(GRU(150, recurrent_dropout=0.1, dropout=0.1))
    model.add(Dense(vocab, activation='softmax'))
    checkpoint = ModelCheckpoint(f'./models/{name}_best.h5', 
                                 monitor='val_acc', 
                                 save_best_only=True, 
                                 mode='max', 
                                 verbose=1)
    callbacks_list = [checkpoint] 
    print(model.summary())
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer=optimizer)
    try:
        model = load_model(f'./models/{name}_best.h5')
        print(f"Loaded pre-existing model from ./models/{name}_best.h5")
    except (OSError, IOError):
        print(f"No pre-existing model found at ./models/{name}_best.h5. Training a new model.")
    model.fit(X_tr, y_tr, 
              epochs=30, 
              verbose=1, 
              validation_data=(X_val, y_val), 
              callbacks=callbacks_list)

# build_model("sacred_3")
model = tf.keras.models.load_model('./models/sacred_3_best.h5')

def generate_seq(model, mapping, seq_length, seed_text, n_chars, temperature=1.2):
    in_text = seed_text
    for _ in range(n_chars):
        encoded = [mapping[char] for char in in_text]
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        yhat = model.predict(encoded, verbose=0)
        yhat_temp = yhat**temperature / np.sum(yhat**temperature)
        next_index = np.random.choice(len(mapping), p=yhat_temp[0])
        out_char = [char for char, index in mapping.items() if index == next_index][0]
        in_text += out_char
        if out_char in ('.', '!', '?') or len(in_text) >= n_chars:
            break
    in_text = sent_tokenize(in_text)
    in_text_str = ' '.join(in_text) 
    return in_text_str

inp = 'the world'
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