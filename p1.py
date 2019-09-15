import numpy as np
import matplotlib.pyplot as plt
import pandas
import csv
import requests
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import sys
stdi,stdo,stde=sys.stdin,sys.stdout,sys.stderr
reload(sys)
sys.stdin,sys.stdout,sys.stderr=stdi,stdo,stde
sys.setdefaultencoding('utf8')
import math

# read the data
DATA_DIR = 'cw2-data/txt/'
txt_list = ['bbc-2016.txt','bbc-2017.txt','bbc-2018.txt','bbc-2019.txt','guardian-2016.txt','guardian-2017.txt','guardian-2018.txt','guardian-2019.txt','independent-2016.txt','independent-2017.txt','independent-2018.txt','independent-2019.txt','nytimes-2016.txt','nytimes-2017.txt','nytimes-2018.txt','nytimes-2019.txt','telegraph-2016.txt','telegraph-2017.txt','telegraph-2018.txt','telegraph-2019.txt']

file_list = []
blob_list = []
try:
    for file in txt_list:
        with open(DATA_DIR + file) as f:
            raw_file = f.read()
            blob = TextBlob(raw_file.decode('utf-8'))
            blob_list.append(blob)
except Exception as x:
    print('error> reading data file' + str(x))
    sys.exit

stopwords = requests.get('https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt').content.split('\n')
# remove the stop word
filtered_blob = []
for blob in blob_list:
    tmp_blob = TextBlob('')
    for w in blob.words.lower():
        if w not in stopwords:
            if w.isalnum()==True:
                tmp_blob += w + ' '
            elif w.isalpha() == True:
                tmp_blob += w + ' '
    filtered_blob.append(tmp_blob)

# polarity & subjectivity
sen = []
# polarity & subjectivity
for i in range(20):
    print(txt_list[i])
    sen.append(filtered_blob[i].sentiment)
    print(filtered_blob[i].sentiment)

# word count
for blob in filtered_blob:
    print(len(blob.words))

# most frequent term, frequency and how many times it appears
freq_words = []
tf = []

for blob in filtered_blob:
    c = len(blob.words)
    wc = blob.word_counts
    freq_word = sorted(wc,key=wc.__getitem__,reverse=True)[0]
    freq_words.append(freq_word)
    counts = wc[freq_word]
    frequency = float(counts)/float(c)
    tf.append(frequency)
    print(freq_word,frequency,counts)

# idf
idf = []
for word in freq_words:
    counter = 0
    for blob in filtered_blob:
        if word in blob.words:
            counter+=1
    counter = math.log((len(txt_list)-float(counter)+0.5)/(float(counter)+0.5))
    idf.append(counter)

# tf*idf
for i in range(20):
    print(float(tf[i])*float(idf[i]))
    print(counter)

years = ['2016','2017','2018','2019']
# polarities for each article
polarities = []
for i in range(20):
    polarities.append(sen[i][0])
plt.plot(years, polarities[:4], label='BBC',color = 'r')
plt.plot(years, polarities[4:8], label='Guardian', color = 'g')
plt.plot(years, polarities[8:12], label='Independent', color = 'y')
plt.plot(years, polarities[12:16], label='NYtimes',color = 'k')
plt.plot(years, polarities[16:20], label='Telegraph', color = 'b')

plt.legend()
plt.show()

#average the polarities
mean_polarities = []
for i in range(0,20,4):
    mean_polarities.append(np.mean(polarities[i:i+4]))

# subjectivities for each article
subjectivities = []
for i in range(20):
    subjectivities.append(sen[i][1])

mean_subjectivities = []
for i in range(0,20,4):
    mean_subjectivities.append(np.mean(subjectivities[i:i+4]))
plt.bar(['BBC', 'Guardian', 'Independent', 'NYtimes', 'Telegraph'], mean_polarities, color='k')
plt.ylabel('Average Polarity')
plt.xlabel('News Outlet')
plt.show()
