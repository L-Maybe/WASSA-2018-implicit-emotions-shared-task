# -*- coding: utf-8 -*-
# @Time    : 2018/4/13 17:29
# @Author  : David
# @Email   : mingren4792@126.com
# @File    : wassa_train_cnn.py
# @Software: PyCharm

# max len = 56

import os
import sys
import logging
import pickle
import numpy as np
import pandas as pd

from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.preprocessing import sequence
from keras.layers import Input

from keras.utils import np_utils

# maxlen = 56
batch_size = 100
nb_epoch = 10
hidden_dim = 120

kernel_size = 3
nb_filter = 60

# test = pd.read_csv("./corpus/imdb/testData.tsv", header=0,
#                    delimiter="\t", quoting=3)


def get_idx_from_sent(sent, word_idx_map):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
        else:
            x.append(1)

    return x


def make_idx_data(revs, word_idx_map, maxlen=60):
    """
    Transforms sentences into a 2-d matrix.
    """
    X_train, X_test, X_dev, y_train, y_dev = [], [], [], [], []
    for rev in revs:
        #  return sentence vectors
        sent = get_idx_from_sent(rev['text'], word_idx_map)
        #  get y label
        y = rev['y']
        if rev['split'] == 1:
            X_train.append(sent)
            y_train.append(y)
        elif rev['split'] == 0:
            X_dev.append(sent)
            y_dev.append(y)
        elif rev['split'] == -1:
            X_test.append(sent)

    print(np.array(y_train).shape)
    X_train = sequence.pad_sequences(np.array(X_train), maxlen=maxlen)
    X_dev = sequence.pad_sequences(np.array(X_dev), maxlen=maxlen)
    X_test = sequence.pad_sequences(np.array(X_test), maxlen=maxlen)
    # X_valid = sequence.pad_sequences(np.array(X_valid), maxlen=maxlen)
    # one hot coding
    y_train = np_utils.to_categorical(np.array(y_train))
    y_dev = np_utils.to_categorical(np.array(y_dev))
    # y_valid = np.array(y_valid)

    return [X_train, X_test, X_dev, y_train, y_dev]


if __name__ == '__main__':
    pickle_file = os.path.join('pickle', 'waasa_CNN_data.pickle')

    revs, W, word_idx_map, vocab, maxlen = pickle.load(open(pickle_file, 'rb'))

    X_train, X_test, X_dev, y_train, y_dev = make_idx_data(revs, word_idx_map, maxlen=maxlen)

    # get a total of how many rows
    n_train_sample = X_train.shape[0]

    n_test_sample = X_test.shape[0]
    #  get word length
    len_sentence = X_train.shape[1]  # 200

    max_features = W.shape[0]

    num_features = W.shape[1]  # 400

    # Keras Model
    # this is the placeholder tensor for the input sequence
    sequence = Input(shape=(maxlen,), dtype='int32')
    # Setting trainable=False makes this coding layer not trainable.
    # 需要语义特征，我们大可把以及训练好的词向量权重直接扔到Embedding层中
    embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, weights=[W],
                         trainable=False)(sequence)
    embedded = Dropout(0.25)(embedded)

    # convolutional layer
    convolution = Convolution1D(filters=64,
                                kernel_size=kernel_size,
                                padding='valid',
                                activation='relu',
                                strides=1
                                )(embedded)

    convolution1 = Convolution1D(filters=128,
                                kernel_size=kernel_size,
                                padding='valid',
                                activation='relu',
                                strides=1
                                )(embedded)

    maxpooling = MaxPooling1D(pool_size=2)(convolution1)
    maxpooling = Flatten()(maxpooling)

    # We add a vanilla hidden layer:
    dense = Dense(70)(maxpooling)  # best: 120
    dense = Dropout(0.25)(dense)  # best: 0.25
    dense = Activation('relu')(dense)

    output = Dense(7, activation='softmax')(dense)
    model = Model(inputs=sequence, outputs=output)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    model.fit(X_train, y_train, validation_data=[X_dev, y_dev], batch_size=batch_size, epochs=nb_epoch, verbose=2)
    y_pred = model.predict(X_test, batch_size=batch_size)
    y_pred = np.argmax(y_pred, axis=1)

    result_output = pd.DataFrame(data={"sentiment": y_pred})

    # Use pandas to write the comma-separated output file
    result_output.to_csv("./result/cnn.csv", index=False, quoting=3)