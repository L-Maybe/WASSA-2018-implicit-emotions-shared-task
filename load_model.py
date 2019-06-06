import pandas as pd
import numpy as np
import pickle
import os
from keras.preprocessing import sequence
from keras.models import load_model
from keras.utils import np_utils
from capsule_net import Capsule
from metrics import f1

test_lexicon = pd.read_csv('./lexicon/test_lexicon.csv')

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
    X_train, X_trial, X_test, y_train, y_trial, lex_train, lex_trial, lex_test = [], [], [], [], [], [], [], []
    for rev in revs:
        sent = get_idx_from_sent(rev['text'], word_idx_map)
        y = rev['y']

        if rev['split'] == 1:
            X_train.append(sent)
            y_train.append(y)

        elif rev['split'] == 0:
            X_test.append(sent)

        elif rev['split'] == -1:
            X_trial.append(sent)
            y_trial.append(y)

    X_train = sequence.pad_sequences(np.array(X_train), maxlen=maxlen)
    X_trial = sequence.pad_sequences(np.array(X_trial), maxlen=maxlen)
    X_test = sequence.pad_sequences(np.array(X_test), maxlen=maxlen)
    y_train = np_utils.to_categorical(np.array(y_train))
    y_trial = np_utils.to_categorical(np.array(y_trial))

    # lex_train = train_lexicon.values
    # lex_trial = trial_lexicon.values
    lex_test = test_lexicon.values
    lex_train = np.array(lex_train)
    lex_trial = np.array(lex_trial)
    lex_test = np.array(lex_test)

    return [X_test, lex_test]


def make_context_idx_data(revs, word_idx_map, maxlen=60):
    """
    Transforms sentences into a 2-d matrix.
    """
    X_test,lex_trial = [], []
    for rev in revs:
        sent = get_idx_from_sent(rev['text'], word_idx_map)
        X_test.append(sent)

    X_test = sequence.pad_sequences(np.array(X_test), maxlen=maxlen)

    lex_test = test_lexicon.values
    lex_test = np.array(lex_test)

    return [X_test, lex_test]

def get_test_context_dataset():
    pickle_file = os.path.join('pickle', 'test_context_left.pickle')
    revs, W, word_idx_map, vocab, maxlen = pickle.load(open(pickle_file, 'rb'))
    X_left_context_test, lex_test = make_context_idx_data(revs, word_idx_map, maxlen=76)

    pickle_file = os.path.join('pickle', 'test_context_right.pickle')
    revs, W, word_idx_map, vocab, maxlen = pickle.load(open(pickle_file, 'rb'))
    X_right_context_test, lex_test = make_context_idx_data(revs, word_idx_map, maxlen=100)
    return X_left_context_test, X_right_context_test, lex_test

if __name__ == '__main__':
    pickle_file = os.path.join('pickle', 'test_trial_train.pickle')

    revs, W, word_idx_map, vocab, maxlen = pickle.load(open(pickle_file, 'rb'))

    X_test, lex_test = make_idx_data(revs, word_idx_map, maxlen=maxlen)

    # context1 = load_model('./context_1.h5', custom_objects={'f1': f1})
    # context2 = load_model('./context_2.h5', custom_objects={'f1': f1})
    # context3 = load_model('./context_3.h5', custom_objects={'f1': f1})
    #
    # stacked_bi_context_gru1 = load_model('./context_gru1.h5', custom_objects={'f1': f1})
    # stacked_bi_context_gru2 = load_model('./context_gru2.h5', custom_objects={'f1': f1})
    # stacked_bi_context_gru3 = load_model('./context_gru3.h5', custom_objects={'f1': f1})
    # Routings = 5
    # Num_capsule = 10
    # Dim_capsule = 32
    capsulenet1 = load_model('./capsulenet_gru_1.h5', custom_objects={'Capsule': Capsule})
    capsulenet2 = load_model('./capsulenet_gru_2.h5', custom_objects={'Capsule': Capsule})
    capsulenet3 = load_model('./capsulenet_gru_3.h5', custom_objects={'Capsule': Capsule})

    lstm1 = load_model('./lstm_1.h5')
    lstm2 = load_model('./lstm_2.h5')
    lstm3 = load_model('./lstm_3.h5')

    gru1 = load_model('./gru_1.h5')
    gru2 = load_model('./gru_2.h5')
    gru3 = load_model('./gru_3.h5')

    # context1_pre = context1.predict([X_left_context_test, X_right_context_test, lex_test])
    # context2_pre = context2.predict([X_left_context_test, X_right_context_test, lex_test])
    # context3_pre = context3.predict([X_left_context_test, X_right_context_test, lex_test])
    #
    # context_gru1_pre = stacked_bi_context_gru1.predict([X_left_context_test, X_right_context_test, lex_test])
    # context_gru2_pre = stacked_bi_context_gru2.predict([X_left_context_test, X_right_context_test, lex_test])
    # context_gru3_pre = stacked_bi_context_gru3.predict([X_left_context_test, X_right_context_test, lex_test])

    lstm1_pre = lstm1.predict([X_test, lex_test])
    lstm2_pre = lstm3.predict([X_test, lex_test])
    lstm3_pre = lstm3.predict([X_test, lex_test])
    #
    gru1_pre = gru1.predict([X_test, lex_test])
    gru2_pre = gru2.predict([X_test, lex_test])
    gru3_pre = gru3.predict([X_test, lex_test])

    capsulenet1_pre = capsulenet1.predict([X_test, lex_test])
    capsulenet2_pre = capsulenet2.predict([X_test, lex_test])
    capsulenet3_pre = capsulenet3.predict([X_test, lex_test])

    lstm_pre = (lstm1_pre + lstm2_pre + lstm3_pre) / 3
    gru_pre = (gru1_pre + gru2_pre + gru3_pre) / 3
    capsulenet_pre = (capsulenet1_pre + capsulenet2_pre + capsulenet3_pre) / 3
    # context_pre = (context1_pre + context2_pre + context3_pre) / 3
    # context_gru = (context_gru1_pre + context_gru2_pre + context_gru3_pre) / 3
    #
    pre = lstm_pre + gru_pre + capsulenet_pre
    pre = np.argmax(lstm1_pre, axis=1)
    result_output = pd.DataFrame(data={"sentiment": pre})
    result_output.to_csv("./result/lstm_gru_capsulenet_voting.csv", index=False, quoting=3)




    #
    # model = load_model('./context_1.h5')
    # pre = model.predict([X_left_context_test, X_right_context_test, lex_test])
    # print(pre)