from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import logging
import pickle
import numpy as np
import pandas as pd

from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, LSTM, GRU, Bidirectional, Input, Concatenate, Flatten
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from keras.utils import np_utils
from datetime import datetime
from time import time
import matplotlib.pyplot as plt

batch_size = 1024
nb_epoch = 200
hidden_dim = 120

Routings = 5
Num_capsule = 10
Dim_capsule = 32
model_info = 'SLGRU'
train_lexicon = pd.read_csv('./lexicon/train-v3.csv', sep=',')
trial_lexicon = pd.read_csv('./lexicon/trial-v3.csv', sep=',')


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
    X_train, X_trial, y_train, y_trial, lex_train, lex_trial = [], [], [], [], [], []
    for rev in revs:
        sent = get_idx_from_sent(rev['text'], word_idx_map)
        y = rev['y']

        if rev['split'] == 1:
            X_train.append(sent)
            y_train.append(y)

        elif rev['split'] == -1:
            X_trial.append(sent)
            y_trial.append(y)

    X_train = sequence.pad_sequences(np.array(X_train), maxlen=maxlen)
    X_trial = sequence.pad_sequences(np.array(X_trial), maxlen=maxlen)
    # X_valid = sequence.pad_sequences(np.array(X_valid), maxlen=maxlen)
    y_train = np_utils.to_categorical(np.array(y_train))
    y_trial = np_utils.to_categorical(np.array(y_trial))
    # y_valid = np.array(y_valid)

    lex_train = train_lexicon.values
    lex_trial = trial_lexicon.values
    lex_train = np.array(lex_train)
    lex_trial = np.array(lex_trial)

    return [X_train, X_trial, y_train, y_trial, lex_train, lex_trial]


def train(model, batch, epoch, x_train, y_train, x_trial, y_trial, lex_train, lex_trial):
    start = time()
    log_dir = datetime.now().strftime(model_info + '_%Y%m%d_%H%M')
    os.mkdir(log_dir)

    es = EarlyStopping(monitor='val_acc', patience=20)
    mc = ModelCheckpoint(log_dir + '\\' + model_info + '-EP{epoch:02d}-ACC{val_acc:.4f}.h5',
                         monitor='val_acc', save_best_only=True)
    tb = TensorBoard(log_dir=log_dir, histogram_freq=0)

    start = time()
    h = model.fit(x=[x_train, lex_train],
                  y=y_train,
                  batch_size=batch,
                  epochs=28,
                  validation_data=([x_trial, lex_trial], y_trial),
                  callbacks=[es, mc, tb])

    print('\n@ Total Time Spent: %.2f seconds' % (time() - start))
    acc, val_acc = h.history['acc'], h.history['val_acc']
    m_acc, m_val_acc = np.argmax(acc), np.argmax(val_acc)
    print("@ Best Training Accuracy: %.2f %% achieved at EP #%d." % (acc[m_acc] * 100, m_acc + 1))
    print("@ Best Testing Accuracy: %.2f %% achieved at EP #%d." % (val_acc[m_val_acc] * 100, m_val_acc + 1))
    return h


def accuracy_curve(h):
    acc, loss, val_acc, val_loss = h.history['acc'], h.history['loss'], h.history['val_acc'], h.history['val_loss']
    epoch = len(acc)
    plt.figure(figsize=(17, 5))
    plt.subplot(121)
    plt.plot(range(epoch), acc, label='Train')
    plt.plot(range(epoch), val_acc, label='Test')
    plt.title('Accuracy over ' + str(epoch) + ' Epochs', size=15)
    plt.legend()
    plt.grid(True)
    plt.subplot(122)
    plt.plot(range(epoch), loss, label='Train')
    plt.plot(range(epoch), val_loss, label='Test')
    plt.title('Loss over ' + str(epoch) + ' Epochs', size=15)
    plt.legend()
    plt.grid(True)

    plt.savefig('plot_kNN.png')
    plt.show()


if __name__ == '__main__':
    log_dir = datetime.now().strftime(model_info + '_%Y%m%d_%H%M')
    os.mkdir(log_dir)
    mc = ModelCheckpoint(log_dir + '\\' + model_info + '-EP{epoch:02d}-ACC{val_acc:.4f}.h5',
                         monitor='val_acc', save_best_only=True)
    pickle_file = os.path.join('pickle', 'parse_staford_no_letters_only_no_stopword_v3.pickle')

    revs, W, word_idx_map, vocab, maxlen = pickle.load(open(pickle_file, 'rb'))

    X_train, X_trial, y_train, y_trial, lex_train, lex_trial = make_idx_data(revs, word_idx_map, maxlen=maxlen)

    n_train_sample = X_train.shape[0]

    n_trial_sample = X_trial.shape[0]

    len_sentence = X_train.shape[1]  # 200

    max_features = W.shape[0]

    num_features = W.shape[1]  # 400

    sequence = Input(shape=(maxlen,), dtype='int32')
    lex_input = Input(shape=(43,), dtype='float32')

    embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, mask_zero=True,
                         weights=[W], trainable=False)(sequence)
    embedded = Dropout(0.25)(embedded)
    # stacked LSTM
    hidden = Bidirectional(LSTM(hidden_dim, recurrent_dropout=0.5, return_sequences=True))(embedded)
    hidden = Bidirectional(LSTM(hidden_dim, recurrent_dropout=0.5))(hidden)

    dense = Concatenate(axis=-1)([hidden, lex_input])
    output = Dense(6, activation='softmax')(dense)
    model = Model(inputs=[sequence, lex_input], outputs=output)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    model.fit(x=[X_train, lex_train],
              y=y_train,
              batch_size=1024,
              epochs=1,
              validation_data=([X_trial, lex_trial], y_trial),
              callbacks=[mc])

    # y_pred = model.predict([X_trial, lex_trial], batch_size=batch_size)


