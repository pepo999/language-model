from nltk import bigrams, trigrams
from collections import Counter, defaultdict
import random
from pymongo import MongoClient
import time

client = MongoClient('mongodb+srv://pietroviglino999:rkoEZiZp6tzduEUZ@vpp4dbtest.yseft60.mongodb.net/admin?retryWrites=true&replicaSet=atlas-du9683-shard-0&readPreference=primary&srvServiceName=mongodb&connectTimeoutMS=10000&authSource=admin&authMechanism=SCRAM-SHA-1', 27017)
db = client['language_model']
collection = db['wiki']

wiki_list = list(collection.find({}, {"_id": 0, "title": 0}))
wiki_text = [x['text']for x in wiki_list]

# reut sent:  [['ASIAN', 'EXPORTERS', 'FEAR', 'DAMAGE', 'FROM', 'U', '.', 'S', '.-', 'JAPAN', 'RIFT', 'Mounting', 
# 'trade', 'friction', 'between', 'the', 'U', '.', 'S', '.', 'And', 'Japan', 'has', 'raised', 'fears', 'among', 'many', 'of', 'Asia', "'", 's', 'exporting', 'nations', 'that', 'the', 'row', 'could', 'inflict', 'far', '-', 'reaching', 'economic', 'damage', ',', 'businessmen', 'and', 'officials', 'said', '.'], ['They', 'told', 'Reuter', 'correspondents', 'in', 'Asian', 'capitals', 'a', 'U', '.', 'S', '.', 'Move', 'against', 'Japan', 'might', 'boost', 'protectionist', 'sentiment', 'in', 'the', 'U', '.', 'S', '.', 'And', 'lead', 'to', 'curbs', 'on', 'American', 'imports', 'of', 'their', 'products', '.'], ...]

wiki_data = []
for text in wiki_text:
    for sentence in text:
        sent = []
        split_s = sentence.split(' ')
        for word in split_s:
            sent.append(word)
        wiki_data.append(sent)

model = defaultdict(lambda: defaultdict(lambda: 0))
 
for sentence in wiki_data:
    for w1, w2, w3 in trigrams(sentence, pad_right=True, pad_left=True):
        model[(w1, w2)][w3] += 1

for w1_w2 in model:
    total_count = float(sum(model[w1_w2].values()))
    for w3 in model[w1_w2]:
        model[w1_w2][w3] /= total_count
 

def sentence(a, b, counter): 
    if counter == 10:
        return
    counter += 1
    text = [a, b]
    sentence_finished = False
    while not sentence_finished:
        r = random.random()
        accumulator = .0
        for word in model[tuple(text[-2:])].keys():
            accumulator += model[tuple(text[-2:])][word]
            if accumulator >= r:
                text.append(word)
                break
        if text[-2:] == [None, None]:
            sentence_finished = True       
    print (' '.join([t for t in text if t]))
    time.sleep(3)
    first_word = text[0]
    next_word = text[1] 
    if next_word is None:
        next_word = 'world'
    if first_word is None:
        first_word = 'the'
    sentence(first_word, next_word, counter)
    
counter = 0
sentence("to", "the", counter)