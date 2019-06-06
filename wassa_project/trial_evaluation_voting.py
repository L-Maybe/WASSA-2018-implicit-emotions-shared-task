
import os
import sys
import logging
import pickle
import numpy as np
import pandas as pd
from metrics import f1

from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, LSTM, GRU, Bidirectional, Input, Concatenate
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.models import load_model
from keras.utils import np_utils

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



def get_trial_context_data():
    left_pickle_file = os.path.join('pickle', 'context_left.pickle')

    left_revs, left_W, left_word_idx_map, left_vocab, left_maxlen = pickle.load(open(left_pickle_file, 'rb'))

    left_X_train, left_X_test, left_X_dev, left_y_train, left_y_dev, left_y_trial = make_idx_data(left_revs,
                                                                                                  left_word_idx_map,
                                                                                                  maxlen=left_maxlen)
    print(np.shape(left_X_train))

    right_pickle_file = os.path.join('pickle', 'context_right.pickle')

    right_revs, right_W, right_word_idx_map, right_vocab, right_maxlen = pickle.load(open(right_pickle_file, 'rb'))

    right_X_train, right_X_test, right_X_dev, right_y_train, right_y_dev, right_y_trial = make_idx_data(right_revs,
                                                                                                        right_word_idx_map,
                                                                                                        maxlen=right_maxlen)
    print(np.shape(right_X_train))
    lex_trial = trial_lexicon.values
    lex_trial = np.array(lex_trial)
    return left_X_test, right_X_test, lex_trial







if __name__ == '__main__':
    pickle_file = os.path.join('pickle', 'parse_staford_no_letters_only_no_stopword_v3.pickle')
    revs, W, word_idx_map, vocab, maxlen = pickle.load(open(pickle_file, 'rb'))
    X_train, X_trial, y_train, y_trial, lex_train, lex_trial = make_idx_data(revs, word_idx_map, maxlen=maxlen)

    left_X_test, right_X_test, lex_trial = get_trial_context_data()

    context1 = load_model('./context_1.h5', custom_objects={'f1': f1})
    context2 = load_model('./context_2.h5', custom_objects={'f1': f1})
    context3 = load_model('./context_3.h5', custom_objects={'f1': f1})

    context_gur1 = load_model('./context_gru1.h5', custom_objects={'f1': f1})
    context_gru2 = load_model('./context_gru2.h5', custom_objects={'f1': f1})
    context_gru3 = load_model('./context_gru3.h5', custom_objects={'f1': f1})

    lstm1 = load_model('./lstm_28_model1.h5')
    lstm2 = load_model('./lstm_28_model2.h5')
    lstm3 = load_model('./lstm_28_model3.h5')

    gru1 = load_model('./gru_26_model1.h5')
    gru2 = load_model('./gru_26_model2.h5')
    gru3 = load_model('./gru_26_model3.h5')

    context1_pre = context1.predict([left_X_test, right_X_test, lex_trial])
    context2_pre = context2.predict([left_X_test, right_X_test, lex_trial])
    context3_pre = context3.predict([left_X_test, right_X_test, lex_trial])

    context_gru1_pre = context_gur1.predict([left_X_test, right_X_test, lex_trial])
    context_gru2_pre = context_gru2.predict([left_X_test, right_X_test, lex_trial])
    context_gru3_pre = context_gru3.predict([left_X_test, right_X_test, lex_trial])

    lstm1_pre = lstm1.predict([X_trial, lex_trial])
    lstm2_pre = lstm3.predict([X_trial, lex_trial])
    lstm3_pre = lstm3.predict([X_trial, lex_trial])

    gru1_pre = gru1.predict([X_trial, lex_trial])
    gru2_pre = gru2.predict([X_trial, lex_trial])
    gru3_pre = gru3.predict([X_trial, lex_trial])

    lstm_pre = (lstm1_pre + lstm2_pre + lstm3_pre) / 3
    gru_pre = (gru1_pre + gru2_pre + gru3_pre) / 3
    context_pre = (context1_pre + context2_pre + context3_pre) / 3
    context_gru_pre = (context_gru1_pre+context_gru2_pre + context_gru3_pre) / 3

    pre = lstm_pre + gru_pre + context_pre + context_gru_pre
    pre = np.argmax(pre, axis=1)
    print(np.shape(pre))
    print(pre)
    print(type(pre))

    result_output = pd.DataFrame(data={"sentiment": pre})
    result_output.to_csv("./result/lstm_gru_context_context_gru_voting.csv", index=False, quoting=3)