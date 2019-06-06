import os
import sys
import logging
import pickle
import numpy as np
import pandas as pd

from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, LSTM, GRU, Bidirectional, Input, Concatenate
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from keras.utils import np_utils
from datetime import datetime
from time import time
import matplotlib.pyplot as plt
from metrics import f1

batch_size = 2048
nb_epoch = 200
hidden_dim = 120

kernel_size = 5
nb_filter = 60
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

def stacked_bi_lstm(batch, epoch):
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    logging.info('loading data...')
    # pickle_file = os.path.join('pickle', 'wassa_origin_tweet_glove.pickle3')
    pickle_file = os.path.join('pickle', 'parse_staford_no_letters_only_no_stopword_noreplace_v3.pickle')

    revs, W, word_idx_map, vocab, maxlen = pickle.load(open(pickle_file, 'rb'))
    logging.info('data loaded!')

    X_train, X_trial, y_train, y_trial, lex_train, lex_trial = make_idx_data(revs, word_idx_map, maxlen=maxlen)

    n_train_sample = X_train.shape[0]
    logging.info("n_train_sample [n_train_sample]: %d" % n_train_sample)

    n_trial_sample = X_trial.shape[0]
    logging.info("n_trial_sample [n_train_sample]: %d" % n_trial_sample)

    len_sentence = X_train.shape[1]  # 200
    logging.info("len_sentence [len_sentence]: %d" % len_sentence)

    max_features = W.shape[0]
    logging.info("num of word vector [max_features]: %d" % max_features)

    num_features = W.shape[1]  # 400
    logging.info("dimension of word vector [num_features]: %d" % num_features)

    sequence = Input(shape=(maxlen,), dtype='int32')
    lex_input = Input(shape=(43,), dtype='float32')

    embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, mask_zero=True,
                         weights=[W], trainable=False)(sequence)
    # embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, weights=[W], trainable=False) (sequence)
    embedded = Dropout(0.25)(embedded)

    # stacked LSTM
    hidden = Bidirectional(LSTM(hidden_dim // 2, recurrent_dropout=0.25, return_sequences=True))(embedded)
    hidden = Bidirectional(LSTM(hidden_dim // 2, recurrent_dropout=0.25))(hidden)

    dense = Concatenate(axis=-1)([hidden, lex_input])
    dense = Dense(256, activation='relu')(dense)
    # dropout = Dropout(0.25)(dense)
    # dense = Dense(128, activation='relu')(dropout)
    # dense = Dense(64, activation='relu')(dense)
    output = Dense(6, activation='softmax')(dense)
    model = Model(inputs=[sequence, lex_input], outputs=output)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', f1])
    model.fit(x=[X_train, lex_train],
              y=y_train,
              batch_size=batch,
              epochs=epoch,
              validation_data=([X_trial, lex_trial], y_trial))

    y_pred = model.predict([X_trial, lex_trial], batch_size=batch_size)
    return y_pred


def stacked_bi_lstm_context():

    left_pickle_file = os.path.join('pickle', 'context_left.pickle')

    left_revs, left_W, left_word_idx_map, left_vocab, left_maxlen = pickle.load(open(left_pickle_file, 'rb'))

    left_X_train, left_X_test, left_X_dev, left_y_train, left_y_dev = make_idx_data(left_revs, left_word_idx_map,
                                                                                    maxlen=left_maxlen)

    left_n_train_sample = left_X_train.shape[0]

    left_n_test_sample = left_X_test.shape[0]

    left_len_sentence = left_X_train.shape[1]  # 200

    left_max_features = left_W.shape[0]

    left_num_features = left_W.shape[1]  # 400

    # Keras Model
    # this is the placeholder tensor for the input sequence
    left_sequence = Input(shape=(left_maxlen,), dtype='int32')

    left_embedded = Embedding(input_dim=left_max_features, output_dim=left_num_features, input_length=left_maxlen,
                              mask_zero=True,
                              weights=[left_W], trainable=False)(left_sequence)
    left_embedded = Dropout(0.25)(left_embedded)
    left_hidden = Bidirectional(LSTM(hidden_dim, recurrent_dropout=0.25, return_sequences=True))(left_embedded)
    left_hidden = Bidirectional(LSTM(hidden_dim, recurrent_dropout=0.25))(left_hidden)

    right_pickle_file = os.path.join('pickle', 'context_right.pickle')

    right_revs, right_W, right_word_idx_map, right_vocab, right_maxlen = pickle.load(open(right_pickle_file, 'rb'))

    right_X_train, right_X_test, right_X_dev, right_y_train, right_y_dev = make_idx_data(right_revs, right_word_idx_map,
                                                                                         maxlen=right_maxlen)

    right_n_train_sample = right_X_train.shape[0]

    right_n_test_sample = right_X_test.shape[0]

    right_len_sentence = right_X_train.shape[1]  # 200

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
    right_hidden = Bidirectional(LSTM(hidden_dim, recurrent_dropout=0.25, return_sequences=True))(right_embedded)
    right_hidden = Bidirectional(LSTM(hidden_dim, recurrent_dropout=0.25))(right_hidden)

    x = Concatenate(axis=-1)([left_hidden, right_hidden])
    # x =Concatenate([left_flatten, right_flatten])
    dense = Dense(256, activation='relu')(x)
    dropout = Dropout(0.25)(dense)
    dense = Dense(256, activation='relu')(dropout)
    dense = Dense(256, activation='relu')(dense)
    output = Dense(6, activation='softmax')(dense)

    model = Model(inputs=[left_sequence, right_sequence], outputs=output)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    early_stopping = EarlyStopping(monitor='val_acc', patience=3, verbose=2)

    model.fit([left_X_train, right_X_train], left_y_train, validation_split=0.1, batch_size=batch_size, epochs=nb_epoch,
              verbose=1, callbacks=[early_stopping])

    test = np.hstack((left_X_test, right_X_test))

    y_pred = model.predict(test, batch_size=batch_size)
    return y_pred


def my_votting(pre1, pre2, pre3):
    result = pre1 + pre2 + pre3
    pre = np.argmax(result, axis=1)
    return pre

if __name__ == '__main__':
    bi_lstm_pre=stacked_bi_lstm()
    context_bi_lstm=stacked_bi_lstm_context()
