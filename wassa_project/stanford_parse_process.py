# -*- coding: utf-8 -*-
# @Time    : 2018/4/13 14:32
# @Author  : David
# @Email   : mingren4792@126.com
# @File    : mywassa_2018_cnn.py
# @Software: PyCharm

import pandas as pd
import importlib
from nltk.tokenize import TweetTokenizer
import numpy as np
import re
import os
import pickle
import gensim
from stanford_tokenizer_config import *
from gensim.models import KeyedVectors
from collections import defaultdict
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import pandas as pd

embedding_dim = 300

train_data = pd.read_csv("./wassa2018_data/train-v3.csv", header=None, sep='\t', quoting=3)
trial_data = pd.read_csv('./wassa2018_data/trial-v3.csv',  header=None, sep='\t', quoting=3)
trial_labels = pd.read_table('./wassa2018_data/trial-v3.labels', header=None)
train_data.columns = ['sentiment', 'review']
trial_data.columns = ['sentiment', 'review']

train_data["sentiment"] = train_data["sentiment"] .replace({'sad': 0,
                                                            'joy': 1, 'disgust': 2,
                                                            'surprise': 3, 'anger': 4,
                                                            'fear': 5
                                                      })
trial_data["sentiment"] = trial_data["sentiment"] .replace({'sad': 0,
                                                            'joy': 1, 'disgust': 2,
                                                            'surprise': 3, 'anger': 4,
                                                            'fear': 5
                                                      })


# pickle_file = os.path.join('pickle', 'tran_and_trial.pickle')
# train_921600, trial_9600 = pickle.load(open(pickle_file, 'rb'))
# train_921600 = pd.DataFrame(train_921600)
# trial_9600 = pd.DataFrame(trial_9600)
# # print(train_921600.values[0])
# # print(trial_9600.values[0])
# train_921600.columns = ['sentiment', 'review']
# trial_9600.columns = ['sentiment', 'review']

def load_pickle(train_pickle, trial_pickle):
    train_path = os.path.join('pickle', train_pickle)
    trial_path = os.path.join('pickle', trial_pickle)
    train_stanford = pickle.load(open(train_path, 'rb'))
    trial_stanford = pickle.load(open(trial_path, 'rb'))
    train_reviews = []
    trial_reviews = []
    for i in range(len(train_stanford)):
        review = ''
        for voc in range(len(train_stanford[i])):
            review += train_stanford[i][voc] + " "
        train_reviews.append(review)

    for i in range(len(trial_stanford)):
        review = ''
        for voc in range(len(trial_stanford[i])):
            review += trial_stanford[i][voc] + " "
        trial_reviews.append(review)

    return train_reviews, trial_reviews


def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    # string = string.replace(':)', ' smile ').replace(':(', ' smile ').replace(':-)', ' smile ')\
    #     .replace(':D', ' smile ').replace('=)', ' smile ').replace('😄', ' smile ').replace('☺', ' smile ')
    # string = string.replace('❤', ' like ').replace('<3', ' like ').replace('💕', ' like ').replace('😍', ' like ')
    # string = string.replace('🤗', ' happy ')
    # string = string.replace(':(', ' unhappy ').replace(':-(', ' unhappy ').replace('💔', ' unhappy ')\
    #     .replace('😕', 'unhappy ').replace('😤', ' unhappy ')
    # string = string.replace('😡', ' anger ').replace('🙃', ' anger ')
    # string = string.replace('😞', ' sadness ').replace('😓', ' sadness ').replace('😔', ' sadness ')
    # string = string.replace('😭', ' cry ')
    # string = string.replace('😂', ' smile to cry ')
    # string = string.replace('😱', '  terrified ')
    # string = string.replace('💀', ' skull ')
    # string = string.replace('👎🏽', ' dislike ').replace('😅', ' dislike ').replace('😒', ' scorn ')
    # string = string.replace('👏🏻', ' applause ')
    # string = string.replace('😈', ' grimace ')
    # string = string.replace('✊', ' fist ')
    # string = string.replace('🙄', ' roll eyes ')
    # string = string.replace('😳', ' panic ')
    # string = string.replace('😷', ' speechless ').replace('🤷', ' speachless ')
    # string = string.replace('😩', ' anguish ').replace('😒', ' anguish ').replace('😑', ' anguish ')
    # string = string.replace('😖', ' entanglement ')
    # string = string.replace('👌', ' ok ')
    string = string.replace('un[#TRIGGERWORD#]', 'not mood').replace('[#TRIGGERWORD#]', 'mood')
    # string = string.replace('[#TRIGGERWORD#]', ' ')
    string = string.replace('http://url.removed', ' url ').replace('@USERNAME', ' username ')

    # delete ascll
    string = re.sub('[^\x00-\x7f]', ' ', string)

    #letters only
    # string = re.sub("[^a-zA-Z]", " ", string)
    string = string.lower()


    words = stanford_tokenizer(string)

    # stops = set(stopwords.words("english"))
    # meaningful_words = [w for w in words if not w in stops]
    string = " ".join(words)
    return string


def tranform_label_to_num(sentiment):
    if 'sad'== sentiment:
        return 0
    elif 'joy'== sentiment:
        return 1
    elif 'disgust' == sentiment:
        return 2
    elif 'surprise' == sentiment:
        return 3
    elif 'anger' == sentiment:
        return 4
    elif 'fear' == sentiment:
        return 5



def build_data_train(train_data, trial_data, clean_string=True, train_ratio=0.9):
    """
    Loads data and split into train and test sets.
    """

    revs = []
    vocab = defaultdict(float)
    # Pre-process train data set
    for i in range(len(train_data['review'])):
        rev = train_data['review'][i]
        y = train_data['sentiment'][i]
        if clean_string:
            orig_rev = clean_str(rev)
        else:
            orig_rev = ' '.join(rev).lower()
        words = set(orig_rev.split())
        #  计算某个单词在文本中出现的次数
        for word in words:
            vocab[word] += 1
        datum = {'y': y,
                 'text': orig_rev,
                 'num_words': len(orig_rev.split()),
                 'split': 1}
        revs.append(datum)

    for i in range(len(trial_data['review'])):
        rev = trial_data['review'][i]
        y = tranform_label_to_num(trial_labels.values[i][0])
        if clean_string:
            orig_rev = clean_str(rev)
        else:
            orig_rev = ' '.join(rev).lower()
        words = set(orig_rev.split())
        for word in words:
            vocab[word] += 1
        datum = {'y': y,
                 'text': orig_rev,
                 'num_words': len(orig_rev.split()),
                 'split': -1}
        revs.append(datum)
    return revs, vocab


def load_bin_vec(model, vocab, k=embedding_dim):
    word_vecs = {}
    unk_words = 0

    for word in vocab.keys():
        try:
            word_vec = model[word]
            word_vecs[word] = word_vec
        except:
            unk_words = unk_words + 1

    print(unk_words)
    print(len(vocab))

    return word_vecs, unk_words


def get_W(word_vecs, k=embedding_dim):
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size + 2, k), dtype=np.float32)
    W[0] = np.zeros((embedding_dim, ))
    W[1] = np.random.uniform(-0.25, 0.25, k)
    i = 2
    for word in word_vecs:
        W[i] = word_vecs[word]

        word_idx_map[word] = i
        i = i + 1
    return W, word_idx_map



if __name__ == '__main__':
    print('build_data_train')
    revs, vocab = build_data_train(train_data, trial_data, clean_string=True)
    w2vec_file = 'G:/vector/glove_model.txt'
    max_l = np.max(pd.DataFrame(revs)['num_words'])
    print("load model")
    model = gensim.models.KeyedVectors.load_word2vec_format(w2vec_file, binary=False)
    # model = gensim.models.KeyedVectors.load_word2vec_format(w2vec_file, binary=True)
    # model = gensim.models.KeyedVectors.load_word2vec_format(w2vec_file, binary=False)
    print('load_bin_vec')
    w2v, unk_words = load_bin_vec(model, vocab)
    print('get w')
    W, word_idx_map = get_W(w2v)
    print('to pickle')
    data_processed = os.path.join('pickle', 'parse_staford_context_left_v3.pickle')
    pickle.dump([revs, W, word_idx_map, vocab, max_l], open(data_processed, 'wb'))

