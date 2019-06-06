# max len = 56
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import logging
import pickle
import numpy as np
import pandas as pd

from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, LSTM, GRU, Bidirectional, Input, Concatenate, MaxPooling1D, Flatten
from keras.preprocessing import sequence

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from keras.utils import np_utils
from datetime import datetime
from time import time
import matplotlib.pyplot as plt
from metrics import f1
from keras.utils import np_utils

model_info = 'SLGRU'


# maxlen = 56
batch_size = 830
nb_epoch = 19
hidden_dim = 120

train_lexicon = pd.read_csv('./lexicon/train-v3.csv')
train_lexicon = train_lexicon.values[:, 0:44]
train_lexicon = np.array(train_lexicon)

trial_lexicon = pd.read_csv('./lexicon/trial-v3.csv')
trial_lexicon = trial_lexicon.values[:, 0:44]
trial_lexicon = np.array(trial_lexicon)

test_lexicon = pd.read_csv('./lexicon/test_lexicon.csv')
test_lexicon = test_lexicon.values[:, 0:44]
test_lexicon = np.array(test_lexicon)
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


def train(model, batch, epoch, left_X_train,right_X_train,train_lexicon, left_y_train, left_X_test, right_X_test,trial_lexicon,left_y_trial):
    start = time()
    log_dir = datetime.now().strftime(model_info + '_%Y%m%d_%H%M')
    os.mkdir(log_dir)

    es = EarlyStopping(monitor='val_acc', patience=20)
    mc = ModelCheckpoint(log_dir + '\\' + model_info + '-EP{epoch:02d}-ACC{val_acc:.4f}.h5',
                         monitor='val_acc', save_best_only=True)
    tb = TensorBoard(log_dir=log_dir, histogram_freq=0)

    start = time()
    # h = model.fit(x=[x_train, lex_train],
    #               y=y_train,
    #               batch_size=batch,
    #               epochs=epoch,
    #               validation_data=([x_trial, lex_trial], y_trial),
    #               callbacks=[es, mc, tb])

    h = model.fit(x=[left_X_train, right_X_train, train_lexicon], y=left_y_train,
              validation_data=([left_X_test, right_X_test, trial_lexicon], left_y_trial), epochs=epoch, batch_size=batch,
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


def make_idx_data(revs, word_idx_map, maxlen=60):
    """
    Transforms sentences into a 2-d matrix.
    """
    X_train, X_trial, X_test, y_train, y_trial,  = [], [], [], [], []
    for rev in revs:
        sent = get_idx_from_sent(rev['text'], word_idx_map)
        y = rev['y']

        if rev['split'] == 1:
            X_train.append(sent)
            y_train.append(y)
        elif rev['split'] == -1:
            X_trial.append(sent)
            y_trial.append(y)
        elif rev['split'] == 2:
            X_test.append(sent)


    X_train = sequence.pad_sequences(np.array(X_train), maxlen=maxlen)
    X_trial = sequence.pad_sequences(np.array(X_trial), maxlen=maxlen)
    X_test = sequence.pad_sequences(np.array(X_test), maxlen=maxlen)
    # X_valid = sequence.pad_sequences(np.array(X_valid), maxlen=maxlen)
    if len(y_train) != 0:
        y_train = np_utils.to_categorical(np.array(y_train))
    if len(y_trial) != 0:
        y_trial = np_utils.to_categorical(np.array(y_trial))
    # y_dev = np_utils.to_categorical(np.array(y_dev))
    # y_valid = np.array(y_valid)

    return [X_train, X_trial, X_test, y_train, y_trial]


def get_test_data():
    left_pickle_file = os.path.join('pickle', 'test_context_left.pickle')

    left_revs, left_W, left_word_idx_map, left_vocab, left_maxlen = pickle.load(open(left_pickle_file, 'rb'))

    X_train_left, X_trial_left, X_test_left, y_train_left, y_trial_left = make_idx_data(left_revs,left_word_idx_map,maxlen=76)

    right_pickle_file = os.path.join('pickle', 'test_context_right.pickle')

    right_revs, right_W, right_word_idx_map, right_vocab, right_maxlen = pickle.load(open(right_pickle_file, 'rb'))

    X_train_right, X_trial_right, X_test_right, y_train_right, y_trial_right = make_idx_data(right_revs, right_word_idx_map, maxlen=100)
    return X_test_left, X_test_right


def context(num):
    left_X_test, right_X_test = get_test_data()

    left_pickle_file = os.path.join('pickle', 'context_left.pickle')

    left_revs, left_W, left_word_idx_map, left_vocab, left_maxlen = pickle.load(open(left_pickle_file, 'rb'))

    X_train_left, X_trial_left, X_test_left, y_train_left, y_trial_left = make_idx_data(left_revs, left_word_idx_map, maxlen=left_maxlen)

    left_max_features = left_W.shape[0]

    left_num_features = left_W.shape[1]  # 400

    # Keras Model
    left_sequence = Input(shape=(left_maxlen,), dtype='int32')

    left_embedded = Embedding(input_dim=left_max_features, output_dim=left_num_features, input_length=left_maxlen, mask_zero=True,
                         weights=[left_W], trainable=False)(left_sequence)
    left_embedded = Dropout(0.25)(left_embedded)
    left_hidden = Bidirectional(LSTM(hidden_dim, recurrent_dropout=0.3, return_sequences=True))(left_embedded)
    left_hidden = Bidirectional(LSTM(hidden_dim, recurrent_dropout=0.3))(left_hidden)



    right_pickle_file = os.path.join('pickle', 'context_right.pickle')

    right_revs, right_W, right_word_idx_map, right_vocab, right_maxlen = pickle.load(open(right_pickle_file, 'rb'))

    X_train_right, X_trial_right, X_test_right, y_train_right, y_trial_right= make_idx_data(right_revs, right_word_idx_map,
                                                                                    maxlen=right_maxlen)


    right_max_features = right_W.shape[0]

    right_num_features = right_W.shape[1]  # 400

    # Keras Model
    # this is the placeholder tensor for the input sequence
    right_sequence = Input(shape=(right_maxlen,), dtype='int32')

    right_embedded = Embedding(input_dim=right_max_features, output_dim=right_num_features, input_length=right_maxlen,
                         mask_zero=True,
                         weights=[right_W], trainable=False)(right_sequence)
    # embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, weights=[W], trainable=False) (sequence)
    right_embedded = Dropout(0.25)(right_embedded)
    right_hidden = Bidirectional(LSTM(hidden_dim, recurrent_dropout=0.45, return_sequences=True))(right_embedded)
    right_hidden = Bidirectional(LSTM(hidden_dim, recurrent_dropout=0.45))(right_hidden)

    lexicon_input = Input(shape=(43,), dtype='float32')

    x = Concatenate(axis=-1)([left_hidden, right_hidden, lexicon_input])
    dense = Dense(256, activation='relu')(x)
    dropout = Dropout(0.25)(dense)
    dense = Dense(256, activation='relu')(dropout)
    dense = Dense(256, activation='relu')(dense)
    output = Dense(6, activation='softmax')(dense)

    model = Model(inputs=[left_sequence, right_sequence, lexicon_input], outputs=output)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc',f1])

    model.fit(x=[X_train_left, X_train_right, train_lexicon], y=y_train_left, validation_data=([X_trial_left, X_trial_right, trial_lexicon],y_trial_right), batch_size=batch_size, epochs=nb_epoch)
    model.save('context_'+num+'.h5')

    y_pred = model.predict([left_X_test, right_X_test, test_lexicon],batch_size=batch_size)
    y_pred = np.argmax(y_pred, axis=1)

    result_output = pd.DataFrame(data={"sentiment": y_pred})


    result_output.to_csv("./result/multi_stacked-bi-context-lstm.csv", index=False, quoting=3)


if __name__ == '__main__':
    context('1')
    context('2')
    context('3')
    context('4')
    context('5')