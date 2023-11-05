from nltk import bigrams, trigrams
from collections import defaultdict
import random
from pymongo import MongoClient
import time
import re
import gzip

client = MongoClient('mongodb+srv://pietroviglino999:rkoEZiZp6tzduEUZ@vpp4dbtest.yseft60.mongodb.net/admin?retryWrites=true&replicaSet=atlas-du9683-shard-0&readPreference=primary&srvServiceName=mongodb&connectTimeoutMS=10000&authSource=admin&authMechanism=SCRAM-SHA-1', 27017)
db = client['language_model']
wiki_coll = db['wiki']

wiki_list = list(wiki_coll.find({}, {"_id": 0, "title": 0}))
wiki_text = [x['text']for x in wiki_list]

wiki_data = []
for text in wiki_text:
    for sentence in text:
        sent = []
        split_s = sentence.split(' ')
        for word in split_s:
            sent.append(word)
        wiki_data.append(sent)
        
sacred_texts_gz = ['ac.txt.gz',
                'arp.txt.gz',
                'chinese_buddhism.txt.gz',
                'csj.txt.gz',
                'ebm.txt.gz',
                'mom.txt.gz',
                'salt.txt.gz',
                'twi.txt.gz',
                'yaq.txt.gz'
                ]
sacred = []
for text in sacred_texts_gz:
    with gzip.open(f'data/{text}','rt') as f:
        lines = f.readlines()
        for line in lines:
            line_s = []
            line = re.sub(r'\([^)]*\)', '', line)
            line = re.sub(r'\[[^\]]*\]', '', line)
            line = line.strip()
            line = line.split(' ')
            for word in line:
                line_s.append(word)
            sacred.append(line_s)
def bible():
    text = []
    with open('data/bible.txt') as f:
        lines = f.readlines()
        for line in lines:
            line_s = []
            line = line.replace('\n', '')
            line = line.split(' ')
            for word in line[1:]:
                line_s.append(word)
            text.append(line_s)
        return text

# bible_f = bible()s
model = defaultdict(lambda: defaultdict(lambda: 0))

def train(data): 
    for sentence in data:
        for w1, w2, w3 in trigrams(sentence, pad_right=True, pad_left=True):
            model[(w1, w2)][w3] += 1

    for w1_w2 in model:
        total_count = float(sum(model[w1_w2].values()))
        for w3 in model[w1_w2]:
            model[w1_w2][w3] /= total_count
 
def sentence(data, counter): 
    if counter == 50:
        return
    counter += 1
    random_pos = int(random.random() * len(data))
    text = [data[random_pos][0], data[random_pos][1]]
    sentence_finished = False
    while not sentence_finished:
        r = random.random() + 0.1
        accumulator = .0
        for word in model[tuple(text[-2:])].keys():
            accumulator += model[tuple(text[-2:])][word]
            if accumulator >= r:
                text.append(word)
                break
        if text[-2:] == [None, None]:
            sentence_finished = True    
    sentence_res =  ' '.join([t for t in text if t]) 
    print(sentence_res.capitalize())
    time.sleep(1)
    random_position = random.randint(0, len(text) - 1)
    first_word = text[random_position - 1]
    next_word = text[random_position] 
    if first_word is None:
        return
    if next_word is None:
        next_word = text[-1]
    sentence(data, counter)

# train(wiki_data)
# train(bible_f) 
train(sacred)

counter = 0
sentence(sacred, counter)