from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import numpy as np
import re

data_text = []
with open('data/bible.txt') as f:
    lines = f.readlines()
    data_text = ' '.join(lines[:20])

def text_cleaner(text):
    newString = text.lower()
    newString = re.sub(r"'s\b","",newString)
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
        seq = text[i-length:i+1]
        sequences.append(seq)
    # print('Total Sequences: %d' % len(sequences))
    return sequences

sequences = create_seq(data_new)

chars = sorted(list(set(data_new)))
mapping = dict((c, i) for i, c in enumerate(chars))

model = tf.keras.models.load_model('./bible_1')

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

inp = 'the world is'
print(len(inp))
print(generate_seq(model, mapping, 30, inp.lower(), 12))

def generate_response(input_text):
    return generate_seq(model, mapping, 30, input_text.lower(), 50)

# while True:
#     user_input = input("You: ")   
#     if user_input.lower() == "exit":
#         print("Goodbye!")
#         break
#     response = generate_response(user_input)
#     print("Model: " + response)