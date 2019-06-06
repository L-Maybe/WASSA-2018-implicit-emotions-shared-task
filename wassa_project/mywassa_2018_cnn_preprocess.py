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

from gensim.models import KeyedVectors
from collections import defaultdict
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import pandas as pd

embedding_dim = 300

# train_data = pd.read_csv("./wassa2018_data/tran.csv")
#
# trial_data = pd.read_csv('./wassa2018_data/trial.csv')
# trial_labels = pd.read_table('./wassa2018_data/trial.labels')
# train_data.columns = ['sentiment', 'review']
# trial_data.columns = ['sentiment', 'review']
#
# train_data["sentiment"] = train_data["sentiment"] .replace({'sad': 0,
#                                                             'joy': 1, 'disgust': 2,
#                                                             'surprise': 3, 'anger': 4,
#                                                             'fear': 5
#                                                       })


pickle_file = os.path.join('pickle', 'tran_and_trial.pickle')
train_921600, trial_9600 = pickle.load(open(pickle_file, 'rb'))
train_921600 = pd.DataFrame(train_921600)
trial_9600 = pd.DataFrame(trial_9600)
# print(train_921600.values[0])
# print(trial_9600.values[0])
train_921600.columns = ['sentiment', 'review']
trial_9600.columns = ['sentiment', 'review']

def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = string.replace(':)', ' smile ').replace(':(', ' smile ').replace(':-)', ' smile ')\
        .replace(':D', ' smile ').replace('=)', ' smile ').replace('😄', ' smile ').replace('☺', ' smile ')
    string = string.replace('❤', ' like ').replace('<3', ' like ').replace('💕', ' like ').replace('😍', ' like ')
    string = string.replace('🤗', ' happy ')
    string = string.replace(':(', ' unhappy ').replace(':-(', ' unhappy ').replace('💔', ' unhappy ')\
        .replace('😕', 'unhappy ').replace('😤', ' unhappy ')
    string = string.replace('😡', ' anger ').replace('🙃', ' anger ')
    string = string.replace('😞', ' sadness ').replace('😓', ' sadness ').replace('😔', ' sadness ')
    string = string.replace('😭', ' cry ')
    string = string.replace('😂', ' smile to cry ')
    string = string.replace('😱', '  terrified ')
    string = string.replace('💀', ' skull ')
    string = string.replace('👎🏽', ' dislike ').replace('😅', ' dislike ').replace('😒', ' scorn ')
    string = string.replace('👏🏻', ' applause ')
    string = string.replace('😈', ' grimace ')
    string = string.replace('✊', ' fist ')
    string = string.replace('🙄', ' roll eyes ')
    string = string.replace('😳', ' panic ')
    string = string.replace('😷', ' speechless ').replace('🤷', ' speachless ')
    string = string.replace('😩', ' anguish ').replace('😒', ' anguish ').replace('😑', ' anguish ')
    string = string.replace('😖', ' entanglement ')
    string = string.replace('👌', ' ok ')
    # string = string.replace('[#TRIGGERWORD#]', ' ')
    # string = string.replace('http://url.removed', ' ')


    # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)

    # string = re.sub(r"\'s", " \'s", string)
    # string = re.sub(r"\'ve", " \'ve", string)
    # string = re.sub(r"n\'t", " n\'t", string)
    # string = re.sub(r"\'re", " \'re", string)
    # string = re.sub(r"\'d", " \'d", string)
    # string = re.sub(r"\'ll", " \'ll", string)
    # string = re.sub(r"\'s", " ", string)
    # string = re.sub(r"\'ve", " ", string)
    # string = re.sub(r"n\'t", " ", string)
    # string = re.sub(r"\'re", " ", string)
    # string = re.sub(r"\'d", " ", string)
    # string = re.sub(r"\'ll", " ", string)
    # string = re.sub(r",", " , ", string)
    # string = re.sub(r";", " ; ", string)
    # string = re.sub(r":", " : ", string)
    # string = re.sub(r"\.\.\.", ".", string)
    # string = re.sub(r"\.", " . ", string)
    # string = re.sub(r"!", " ! ", string)
    # string = re.sub(r"\(", " ( ", string)
    # string = re.sub(r"\)", " ) ", string)
    # string = re.sub(r"\?", " ? ", string)
    # string = re.sub(r"\{", " { ", string)
    # string = re.sub(r"\}", " } ", string)
    # string = re.sub(r"\[", " [ ", string)
    # string = re.sub(r"\]", " ] ", string)
    # string = re.sub(r"#", "", string)
    # string = re.sub(r"\~", " ~ ", string)
    # string = re.sub(r"\*", " * ", string)
    # # string = re.sub(r"\s{2,}", " ", string)
    # string = re.sub(r"\\n", " \\n ", string)


    string = string.lower()
    # print(words)
    words = TweetTokenizer().tokenize(string)

    new_words = []
    for word in words:
        # if word in (':-)', ':)', ':D', '=)', ';)'):
        #     new_words.append('smile')
        #     continue
        #
        # if word in ('❤'):
        #     new_words.append('love')
        #     continue
        #
        # if word in (':(', ':-(', '💔'):
        #     new_words.append('unhappy')
        #     continue
        #
        # if word in ('<3'):
        #     new_words.append('like')
        #     continue
        # if word in ('🤗', '😭'):
        #     new_words.append('anger')
        #     continue

        if word.startswith('@'):
            new_words.append('user')
        elif bool(re.search('.*[0-9]+.*', word)):
            new_words.append('number')
        else:
            new_words.append(word)

        # elif word.startswith('http'):
        #     new_words.append('<url>')
        # elif word.startswith('#'):
        #     new_words.append('<hashtag>')
        #     # new_words.append('hashtag')
        # elif bool(re.search('.*[0-9]+.*', word)):
        #     new_words.append('<number>')
        # else:
        #     new_words.append(word)

    string = " ".join(new_words)
    # string = string.replace(r'\u2019', "'").replace(r'\u002c', '')
    # print('pre:'+string)
    # try:
    #  new_words = tokenizer.tokenize(string)
    # except:
    #     print ('Error')
    # string = " ".join(new_words)
    # print(string)
    # letters_only = re.sub("[^a-zA-Z]", " ", string)
    # words = string.lower().split()
    # stops = set(stopwords.words("english"))
    # meaningful_words = [w for w in words if not w in stops]
    # return ( " ".join( meaningful_words ))
    return string


def build_data_train(train_data, trial_data, clean_string=True, train_ratio=0.94):
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
        if clean_string:
            orig_rev = clean_str(rev)
        else:
            orig_rev = ' '.join(rev).lower()
        words = set(orig_rev.split())
        for word in words:
            vocab[word] += 1
        datum = {'y': -1,
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
    revs, vocab = build_data_train(train_921600, trial_9600, clean_string=True)
    print(type(revs))
    print(type(vocab))
    w2vec_file = 'G:/vector/GoogleNews-vectors-negative300.bin'
    max_l = np.max(pd.DataFrame(revs)['num_words'])

    model = KeyedVectors.load_word2vec_format(w2vec_file, binary=True, encoding='utf-8')
    # model = gensim.models.KeyedVectors.load_word2vec_format(w2vec_file, binary=True)
    # model = gensim.models.KeyedVectors.load_word2vec_format(w2vec_file, binary=False)
    w2v, unk_words = load_bin_vec(model, vocab)

    W, word_idx_map = get_W(w2v)
    print(W.shape)
    data_processed = os.path.join('pickle', 'waasa2018_six_fill_target_data.pickle')
    pickle.dump([revs, W, word_idx_map, vocab, max_l], open(data_processed, 'wb'))

