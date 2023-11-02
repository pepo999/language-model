import requests
from bs4 import BeautifulSoup
import random
from pymongo import MongoClient
import re
import time

client = MongoClient('mongodb+srv://pietroviglino999:rkoEZiZp6tzduEUZ@vpp4dbtest.yseft60.mongodb.net/admin?retryWrites=true&replicaSet=atlas-du9683-shard-0&readPreference=primary&srvServiceName=mongodb&connectTimeoutMS=10000&authSource=admin&authMechanism=SCRAM-SHA-1', 27017)
db = client['language_model']
collection = db['wiki']

def scrapeWikiArticle(url):
    try:
        response = requests.get(url=url,)
        soup = BeautifulSoup(response.content, 'html.parser')
        heading = soup.find(id="firstHeading")
        title = heading.text
        titles_db = list(collection.find({}, {"_id":0, "text": 0}))
        titles_list = [x['title'] for x in titles_db]
        if title not in titles_list and title != 'Web scraping' and  ':' not in title:   
            parafs = soup.find_all("p", recursive=True)
            text = []
            for p in parafs:
                p_content = p.text
                p_content = p_content.replace('\n', '')
                p_content = re.sub(r'\[[^\]]*\]', '', p_content)
                p_content = re.sub(r'\([^)]*\)', '', p_content)
                # p_list = re.split(r'(?<=[^A-Z]{2})\.\s', p_content)
                p_list = re.split(r'(?<![A-Z])\.\s', p_content)
                # p_list = p_content.split('. ')
                for sentence in p_list:
                    if sentence != '':
                        if sentence[-1] != '.':
                            sentence += '.'
                        text.append(sentence)
            collection.insert_one({"title": title, "text": text})
            print('added doc on db about ', title)
        allLinks = soup.find(id="bodyContent").find_all("a")
        random.shuffle(allLinks)
        linkToScrape = 0
        for link in allLinks:
            if link['href'].find("/wiki/") == -1: 
                continue
            linkToScrape = link
            break
        time.sleep(1)
        scrapeWikiArticle("https://en.wikipedia.org" + linkToScrape['href'])
    except Exception as e:
        print(e)
        scrapeWikiArticle("https://en.wikipedia.org/wiki/Web_scraping")

scrapeWikiArticle("https://en.wikipedia.org/wiki/Web_scraping")