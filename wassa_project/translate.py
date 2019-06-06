import pandas as pd
import csv
from textblob import TextBlob

train_data = pd.read_csv('./wassa2018_data/train-v3.csv', header=None, sep='\t', quoting=3)
train_data.columns = ['sentiment', 'review']
test = pd.read_csv('./wassa2018_data/translation.csv', header=None, sep='\t', quoting=3)
print(len(test))
# en_blob = TextBlob(u'Simple is better than complex.')
# en_blob = TextBlob(u'@USERNAME A little anger that I am not invited for drinks anymore! :-(')

# en_blob = TextBlob(u'@USERNAME @USERNAME It\'s pretty [#TRIGGERWORD#] that there would even BE stock photos for an event like this.')
# # en_blob = TextBlob(en_blob)
# es_blob = en_blob.translate(to='fr')
#
# reen_blob = es_blob.translate(to='en')
# print(reen_blob)
# print(reen_blob.replace('[# TRIGGERWORD #]', '[#TRIGGERWORD#]'))

for i in range(len(train_data['review'])):
    data = train_data['review'][i]
    data = data.replace('un[#TRIGGERWORD#]', 'un'+train_data['sentiment'][i]).replace('[#TRIGGERWORD#]', train_data['sentiment'][i])
    data = data.replace('http://url.removed', 'url')
    en_blob = TextBlob(data)
    es_blob = en_blob.translate(to='fr')
    reen_blob = es_blob.translate(to='en')
    print(data)
    print(reen_blob.replace('[# TRIGGERWORD #]', '[#TRIGGERWORD#]'))