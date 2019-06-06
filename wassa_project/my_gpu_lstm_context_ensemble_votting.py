# -*- coding: utf-8 -*-
# @Time    : 2018/4/23 21:42
# @Author  : David
# @Email   : mingren4792@126.com
# @File    : my_cnn_lstm_ensemble_votting.py
# @Software: PyCharm
from __future__ import print_function

import pickle
import numpy as np
import pandas as pd
import os
from keras.layers import Dense, Dropout, Bidirectional, Input, Concatenate, GRU
from keras.layers import LSTM, MaxPooling1D, Embedding
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Model
from keras.models import load_model
from keras.wrappers.scikit_learn import KerasClassifier
from voting_classifier import VotingClassifier
from metrics import f1


# batch_size = 256
# nb_epoch = 10  # trial set most accuracy epoch=13  val_acc=0.4736
hidden_dim = 120

train_lexicon = pd.read_csv('./lexicon/train-v3.csv', sep=',')
trial_lexicon = pd.read_csv('./lexicon/trial-v3.csv', sep=',')
test_lexicon = pd.read_csv('./lexicon/test_lexicon.csv', sep=',')

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


def make_idx_data(revs, word_idx_map, maxlen=60, is_split = True):
    """
    Transforms sentences into a 2-d matrix.
    """
    X_train, X_trial, X_test,y_train, y_trial,y_test, lex_train, lex_trial = [], [], [], [], [], [], [], []
    for rev in revs:
        sent = get_idx_from_sent(rev['text'], word_idx_map)
        y = rev['y']
        if is_split:
            if rev['split'] == 1:
                X_train.append(sent)
                y_train.append(y)

            elif rev['split'] == -1:
                X_trial.append(sent)
                y_trial.append(y)
        else:
            X_test.append(sent)
            y_test.append(-1)

    if is_split:
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
    else:
        X_test = sequence.pad_sequences(np.array(X_test), maxlen=117)
        lex_test = test_lexicon.values
        lex_test = np.array(lex_test)
        return [X_test, lex_test]


def stacked_bi_sltm(batch_size, epoch, num):
    x_test, lex_test = get_test_data()
    print(np.shape(x_test))
    X_train, X_trial, y_train, y_trial, lex_train, lex_trial, max_features, num_features, W, maxlen = get_data('parse_staford_no_letters_only_no_stopword_v3.pickle')
    sequence = Input(shape=(maxlen,), dtype='int32')
    lex_input = Input(shape=(43,), dtype='float32')

    embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, mask_zero=True,
                         weights=[W], trainable=False)(sequence)
    embedded = Dropout(0.25)(embedded)
    hidden = Bidirectional(LSTM(hidden_dim, recurrent_dropout=0.5, return_sequences=True))(embedded)
    hidden = Bidirectional(LSTM(hidden_dim, recurrent_dropout=0.5))(hidden)
    x = Concatenate(axis=-1)([hidden, lex_input])
    output = Dense(6, activation='softmax')(x)
    model = Model(inputs=[sequence, lex_input], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    # X_train = np.vstack([X_train, X_trial])
    # y_train = np.vstack([y_train, y_trial])
    # lexicon = np.vstack([lex_train, lex_trial])

    model.fit(x=[X_train, lex_train], y=y_train, batch_size=batch_size, epochs=epoch)
    model.save('lstm_28_model'+num+".h5")
    pre = model.predict([x_test, lex_test])
    return pre

def stacked_bi_gru(batch_size, epoch, num):
    x_test, lex_test = get_test_data()
    X_train, X_trial, y_train, y_trial, lex_train, lex_trial, max_features, num_features, W, maxlen = get_data('parse_staford_no_letters_only_no_stopword_v3.pickle')
    sequence = Input(shape=(maxlen,), dtype='int32')
    lex_input = Input(shape=(43,), dtype='float32')

    embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, mask_zero=True,
                         weights=[W], trainable=False)(sequence)
    embedded = Dropout(0.25)(embedded)
    hidden = Bidirectional(GRU(hidden_dim, recurrent_dropout=0.5, return_sequences=True))(embedded)
    hidden = Bidirectional(GRU(hidden_dim, recurrent_dropout=0.5))(hidden)
    x = Concatenate(axis=-1)([hidden, lex_input])
    output = Dense(6, activation='softmax')(x)
    model = Model(inputs=[sequence, lex_input], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    # X_train = np.vstack([X_train, X_trial])
    # y_train = np.vstack([y_train, y_trial])
    # lexicon = np.vstack([lex_train, lex_trial])

    model.fit(x=[X_train, lex_train], y=y_train, batch_size=batch_size, epochs=epoch)
    model.save('gru_26_model'+num+'.h5')
    pre = model.predict([x_test, lex_test])
    return pre

def get_data(processed_file):
    pickle_file = os.path.join('pickle', processed_file)

    revs, W, word_idx_map, vocab, maxlen = pickle.load(open(pickle_file, 'rb'))

    X_train, X_trial, y_train, y_trial, lex_train, lex_trial = make_idx_data(revs, word_idx_map, maxlen=maxlen, is_split=True)

    n_train_sample = X_train.shape[0]

    n_trial_sample = X_trial.shape[0]

    len_sentence = X_train.shape[1]  # 200

    max_features = W.shape[0]

    num_features = W.shape[1]  # 400
    return X_train, X_trial, y_train, y_trial, lex_train, lex_trial, max_features, num_features, W, maxlen

def get_test_data():
    pickle_file = os.path.join('pickle', 'test_data.pickle')
    revs, W, word_idx_map, vocab, max_l = pickle.load(open(pickle_file, 'rb'))
    x_test, lex_test= make_idx_data(revs, word_idx_map, maxlen=max_l, is_split=False)
    return x_test, lex_test



if __name__ == '__main__':

    # lstm_pre1 = stacked_bi_sltm(830, 28)
    # lstm_pre2 = stacked_bi_sltm(830, 28)
    # lstm_pre3 = stacked_bi_sltm(830, 28)
    # y_pre = lstm_pre1 + lstm_pre2 + lstm_pre3
    # y_pre = np.argmax(y_pre, axis=1)
    # result_output = pd.DataFrame(data={'sentiment': y_pre})
    # result_output.to_csv("./result/votting_submit_test.csv", index=False, quoting=3)

    sltm1 = load_model

    lstm_pre1 = stacked_bi_sltm(830, 28, '1')
    lstm_pre2 = stacked_bi_sltm(830, 28, '2')
    lstm_pre3 = stacked_bi_sltm(830, 28,'3')
    lstm_y_pre = lstm_pre1 + lstm_pre2 + lstm_pre3
    # y_pre =np.argmax(y_pre, axis=1)
    lstm_y_pre = lstm_y_pre/3


    gru_pre1 = stacked_bi_gru(830, 26, '1')
    gru_pre2 = stacked_bi_gru(830, 26, '2')
    gru_pre3 = stacked_bi_gru(830, 26, '3')
    gru_y_pre = gru_pre1 + gru_pre2 + gru_pre3
    gru_y_pre = gru_y_pre/3

    y_pre = gru_y_pre + lstm_y_pre
    y_pre = np.argmax(y_pre, axis=1)



    result_output = pd.DataFrame(data={'sentiment': y_pre})

    # Use pandas to write the comma-separated output file
    # result_output.to_csv("./result/bi-lstm.csv", index=False, quoting=3)

    result_output.to_csv("./result/votting_test.csv", index=False, quoting=3)
