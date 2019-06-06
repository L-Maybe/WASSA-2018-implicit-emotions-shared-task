# -*- coding: utf-8 -*-
# @Time    : 2018/4/13 14:32
# @Author  : David
# @Email   : mingren4792@126.com
# @File    : mywassa_2018_cnn.py
# @Software: PyCharm

import numpy as np
import re
import os
import pickle
import gensim

from gensim.models import KeyedVectors
from collections import defaultdict
from stanford_tokenizer_config import *
import pandas as pd

embedding_dim = 300

train_data = pd.read_csv("./wassa2018_data/train-v3.csv", header=None, sep='\t', quoting=3)
trial_data = pd.read_csv('./wassa2018_data/trial-v3.csv',  header=None, sep='\t', quoting=3)
trial_labels = pd.read_table('./wassa2018_data/trial-v3.labels', header=None)
train_data.columns = ['sentiment', 'review']
trial_data.columns = ['sentiment', 'review']


left_columns = ['sentiment', 'left']
right_columns = ['sentiment', 'right']
train_context_data = pd.read_csv('./wassa2018_data/train_context_v3.csv', sep='\t', quoting=3)
trial_context_data = pd.read_csv('./wassa2018_data/trial_context_v3.csv', sep='\t', quoting=3)

train_left_data = pd.DataFrame(train_context_data, columns=left_columns)
train_right_data = pd.DataFrame(train_context_data, columns=right_columns)

trial_left_data = pd.DataFrame(trial_context_data, columns=left_columns)
trial_right_data = pd.DataFrame(trial_context_data, columns=right_columns)


train_left_data["sentiment"] = train_left_data["sentiment"].replace({'sad': 0,
                                                          'joy': 1, 'disgust': 2,
                                                          'surprise': 3, 'anger': 4,
                                                          'fear': 5
                                                          })
train_right_data["sentiment"] = train_right_data["sentiment"].replace({'sad': 0,
                                                          'joy': 1, 'disgust': 2,
                                                          'surprise': 3, 'anger': 4,
                                                          'fear': 5
                                                          })

trial_left_data["sentiment"] = trial_left_data["sentiment"].replace({'sad': 0,
                                                          'joy': 1, 'disgust': 2,
                                                          'surprise': 3, 'anger': 4,
                                                          'fear': 5
                                                          })
trial_right_data["sentiment"] = trial_right_data["sentiment"].replace({'sad': 0,
                                                          'joy': 1, 'disgust': 2,
                                                          'surprise': 3, 'anger': 4,
                                                          'fear': 5
                                                          })




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
    string = string.replace('un[#TRIGGERWORD#]', 'not mood').replace('[#TRIGGERWORD#]', 'mood')
    # string = string.replace('[#TRIGGERWORD#]', ' ')
    string = string.replace('http://url.removed', ' url ').replace('@USERNAME', ' username ')

    # delete ascll
    string = re.sub('[^\x00-\x7f]', ' ', string)

    #letters only
    string = re.sub("[^a-zA-Z\'.!?]", " ", string)
    string = string.lower()


    words = stanford_tokenizer(string)

    # stops = set(stopwords.words("english"))
    # meaningful_words = [w for w in words if not w in stops]
    string = " ".join(words)
    return string




def build_data_train(train_data, trial_data, clean_string=True, train_ratio=0.9, token_review='review'):
    """
    Loads data and split into train and test sets.
    """
    revs = []
    vocab = defaultdict(float)
    # Pre-process train data set

    for i in range(len(trial_data[token_review])):
        print('第%d条，共%d条' %(i, len(trial_data[token_review])))
        rev = trial_data[token_review][i]
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

    for i in range(len(train_data[token_review])):
        print('第%d条，共%d条' % (i, len(train_data[token_review])))
        rev = train_data[token_review][i]
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
    print('left')
    revs_left, vocab_left = build_data_train(train_left_data, trial_left_data, clean_string=True, token_review='left')
    print('right')
    revs_right, vocab_right = build_data_train(train_right_data, trial_right_data, clean_string=True, token_review='right')

    w2vec_file = 'G:/vector/glove_model.txt'
    max_l_left = np.max(pd.DataFrame(revs_left)['num_words'])
    max_l_right = np.max(pd.DataFrame(revs_right)['num_words'])

    model = gensim.models.KeyedVectors.load_word2vec_format(w2vec_file, binary=False)

    print('start left')
    w2v, unk_words = load_bin_vec(model, vocab_left)
    W, word_idx_map = get_W(w2v)
    data_processed = os.path.join('pickle', 'context_left.pickle')
    pickle.dump([revs_left, W, word_idx_map, vocab_left, max_l_left], open(data_processed, 'wb'))
    print('end left')

    print('start right')
    w2v, unk_words = load_bin_vec(model, vocab_right)
    W, word_idx_map = get_W(w2v)
    data_processed = os.path.join('pickle', 'context_right.pickle')
    pickle.dump([revs_right, W, word_idx_map, vocab_right, max_l_right], open(data_processed, 'wb'))
    print('end right')

