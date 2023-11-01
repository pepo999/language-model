# code courtesy of https://nlpforhackers.io/language-models/

import nltk
# nltk.download('reuters')
# nltk.download('punkt')
# nltk.download('nps_chat')
from nltk.corpus import reuters
# from nltk.corpus import nps_chat
from nltk import bigrams, trigrams
from collections import Counter, defaultdict
import random

from nltk.corpus.util import LazyCorpusLoader
from nltk.corpus.reader.plaintext import CategorizedPlaintextCorpusReader

# nps_chat_path = nltk.data.find('C:\\Users\\Pepo\\AppData\\Roaming\\nltk_data\\corpora\\nps_chat')
# nps_as_categorized_reader = CategorizedPlaintextCorpusReader(
#     root=nps_chat_path, fileids=r'(?!\.).*\.xml', cat_file='cats.txt'
# )

# Create a placeholder for model
model = defaultdict(lambda: defaultdict(lambda: 0))

print('reut sent: ', reuters.sents())

# Count frequency of co-occurance  
for sentence in reuters.sents()[:2]:
    for w1, w2, w3 in trigrams(sentence, pad_right=True, pad_left=True):
        model[(w1, w2)][w3] += 1
    
# for sentence in nps_as_categorized_reader.sents():
#     for w1, w2, w3 in trigrams(sentence, pad_right=True, pad_left=True):
#         print(w1, w2, w3)
#         model[(w1, w2)][w3] += 1

# for index, (key, value) in enumerate(model.items()):
#     if index < 10:
#         print(f"Dictionary {index + 1}:")
#         print("Key:", key)
#         print("Value:", dict(value))
#         print("\n")
        
# Let's transform the counts to probabilities
for w1_w2 in model:
    total_count = float(sum(model[w1_w2].values()))
    for w3 in model[w1_w2]:
        model[w1_w2][w3] /= total_count
        
print(dict(model["today", "the"]))
 
def sentence(): 
    text = ["today", "the"]
    sentence_finished = False
    while not sentence_finished:
    # select a random probability threshold  
        r = random.random()
        accumulator = .0

        for word in model[tuple(text[-2:])].keys():
            accumulator += model[tuple(text[-2:])][word]
            # select words that are above the probability threshold
            if accumulator >= r:
                text.append(word)
                break

        if text[-2:] == [None, None]:
            sentence_finished = True
        
    print (' '.join([t for t in text if t]))

sentence()