import os
import pandas as pd
import numpy as np
import pickle
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import load_model
from metrics import f1
from sklearn.svm import SVC
from capsule_net import Capsule


trial_label = pd.read_table('./wassa2018_data/trial-v3.labels', header=None)

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


def context_make_idx_data(revs, word_idx_map, maxlen=60):
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
    if len(y_train) != 0:
        y_train = np_utils.to_categorical(np.array(y_train))
    if len(y_trial) != 0:
        y_trial = np_utils.to_categorical(np.array(y_trial))
    return [X_train, X_trial, X_test, y_train, y_trial]

def get_context_data():
    left_pickle_file = os.path.join('pickle', 'context_left.pickle')

    left_revs, left_W, left_word_idx_map, left_vocab, left_maxlen = pickle.load(open(left_pickle_file, 'rb'))

    X_train_left, X_trial_left, X_test_left, y_train_left, y_trial_left = context_make_idx_data(left_revs, left_word_idx_map,
                                                                                        maxlen=left_maxlen)

    right_pickle_file = os.path.join('pickle', 'context_right.pickle')

    right_revs, right_W, right_word_idx_map, right_vocab, right_maxlen = pickle.load(open(right_pickle_file, 'rb'))

    X_train_right, X_trial_right, X_test_right, y_train_right, y_trial_right = context_make_idx_data(right_revs,
                                                                                             right_word_idx_map,
                                                                                             maxlen=right_maxlen)
    return [X_trial_left, X_trial_right]


def get_test_context_data():
    left_pickle_file = os.path.join('pickle', 'test_context_left.pickle')

    left_revs, left_W, left_word_idx_map, left_vocab, left_maxlen = pickle.load(open(left_pickle_file, 'rb'))

    X_train_left, X_trial_left, X_test_left, y_train_left, y_trial_left = context_make_idx_data(left_revs,
                                                                                                left_word_idx_map,
                                                                                                maxlen=76)

    right_pickle_file = os.path.join('pickle', 'test_context_right.pickle')

    right_revs, right_W, right_word_idx_map, right_vocab, right_maxlen = pickle.load(open(right_pickle_file, 'rb'))

    X_train_right, X_trial_right, X_test_right, y_train_right, y_trial_right = context_make_idx_data(right_revs,
                                                                                                     right_word_idx_map,
                                                                                                     maxlen=100)
    return [X_test_left, X_test_right]




def make_idx_data(revs, word_idx_map, maxlen=60):
    """
    Transforms sentences into a 2-d matrix.
    """
    X_train, X_trial, X_test, y_train, y_trial = [], [], [], [], []
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

    lex_train = train_lexicon.values
    lex_trial = trial_lexicon.values
    lex_test = test_lexicon.values
    lex_train = np.array(lex_train)
    lex_trial = np.array(lex_trial)
    lex_test = np.array(lex_test)

    return [X_train, X_trial, X_test, y_train, y_trial]


def get_train_trial_test_data():
    pickle_file = os.path.join('pickle', 'test_trial_train.pickle')

    revs, W, word_idx_map, vocab, maxlen = pickle.load(open(pickle_file, 'rb'))

    X_train, X_trial, X_test, y_train, y_trial = make_idx_data(revs, word_idx_map, maxlen=maxlen)
    return [X_train, X_trial, X_test, y_train, y_trial]


def get_stacking_data():
    X_train, X_trial, X_test, y_train, y_trial = get_train_trial_test_data()
    X_trial_left, X_trial_right = get_context_data()
    X_test_left, X_test_right = get_test_context_data()
    print(np.shape(X_test))
    print(len(X_test_left))
    print(len(X_test_right))
    print(len(test_lexicon))

    stacked_bi_context1 = load_model('./context_1.h5', custom_objects={'f1': f1})
    stacked_bi_context2 = load_model('./context_2.h5', custom_objects={'f1': f1})
    stacked_bi_context3 = load_model('./context_3.h5', custom_objects={'f1': f1})
    stacked_bi_context4 = load_model('./context_4.h5', custom_objects={'f1': f1})
    stacked_bi_context5 = load_model('./context_5.h5', custom_objects={'f1': f1})


    capsulenet1 = load_model('./capsulenet_gru_1.h5', custom_objects={'Capsule': Capsule})
    capsulenet2 = load_model('./capsulenet_gru_2.h5', custom_objects={'Capsule': Capsule})
    capsulenet3 = load_model('./capsulenet_gru_3.h5', custom_objects={'Capsule': Capsule})
    capsulenet4 = load_model('./capsulenet_gru_4.h5', custom_objects={'Capsule': Capsule})
    capsulenet5 = load_model('./capsulenet_gru_5.h5', custom_objects={'Capsule': Capsule})

    lstm1 = load_model('./lstm_1.h5')
    lstm2 = load_model('./lstm_2.h5')
    lstm3 = load_model('./lstm_3.h5')
    lstm4 = load_model('./lstm_4.h5')
    lstm5 = load_model('./lstm_5.h5')

    gru1 = load_model('./gru_1.h5')
    gru2 = load_model('./gru_2.h5')
    gru3 = load_model('./gru_3.h5')
    gru4 = load_model('./gru_4.h5')
    gru5 = load_model('./gru_5.h5')

    stacked_bi_context_pre1 = stacked_bi_context1.predict([X_trial_left, X_trial_right, trial_lexicon.values])
    stacked_bi_context_pre2 = stacked_bi_context2.predict([X_trial_left, X_trial_right, trial_lexicon.values])
    stacked_bi_context_pre3 = stacked_bi_context3.predict([X_trial_left, X_trial_right, trial_lexicon.values])
    stacked_bi_context_pre4 = stacked_bi_context4.predict([X_trial_left, X_trial_right, trial_lexicon.values])
    stacked_bi_context_pre5 = stacked_bi_context5.predict([X_trial_left, X_trial_right, trial_lexicon.values])
    context_lstm = np.hstack([stacked_bi_context_pre1, stacked_bi_context_pre2, stacked_bi_context_pre3,
                              stacked_bi_context_pre4, stacked_bi_context_pre5])

    bi_lstm1_pre = lstm1.predict([X_trial, trial_lexicon])
    bi_lstm2_pre = lstm2.predict([X_trial, trial_lexicon])
    bi_lstm3_pre = lstm3.predict([X_trial, trial_lexicon])
    bi_lstm4_pre = lstm4.predict([X_trial, trial_lexicon])
    bi_lstm5_pre = lstm5.predict([X_trial, trial_lexicon])
    lstm = np.hstack([bi_lstm1_pre, bi_lstm2_pre, bi_lstm3_pre, bi_lstm4_pre, bi_lstm5_pre])

    bi_gru1_pre = gru1.predict([X_trial, trial_lexicon])
    bi_gru2_pre = gru2.predict([X_trial, trial_lexicon])
    bi_gru3_pre = gru3.predict([X_trial, trial_lexicon])
    bi_gru4_pre = gru4.predict([X_trial, trial_lexicon])
    bi_gru5_pre = gru5.predict([X_trial, trial_lexicon])
    gru = np.hstack([bi_gru1_pre, bi_gru2_pre, bi_gru3_pre, bi_gru4_pre, bi_gru5_pre])

    capsulenet1_pre = capsulenet1.predict([X_trial, trial_lexicon])
    capsulenet2_pre = capsulenet2.predict([X_trial, trial_lexicon])
    capsulenet3_pre = capsulenet3.predict([X_trial, trial_lexicon])
    capsulenet4_pre = capsulenet4.predict([X_trial, trial_lexicon])
    capsulenet5_pre = capsulenet5.predict([X_trial, trial_lexicon])
    capsulenet = np.hstack([capsulenet1_pre, capsulenet2_pre, capsulenet3_pre, capsulenet4_pre, capsulenet5_pre])

    data_processed = os.path.join('pickle', 'train_stacking_data.pickle')
    pickle.dump([context_lstm, lstm, gru, capsulenet, y_trial], open(data_processed, 'wb'))

    test_stacked_bi_context_pre1 = stacked_bi_context1.predict([X_test_left, X_test_right, test_lexicon.values])
    test_stacked_bi_context_pre2 = stacked_bi_context2.predict([X_test_left, X_test_right, test_lexicon.values])
    test_stacked_bi_context_pre3 = stacked_bi_context3.predict([X_test_left, X_test_right, test_lexicon.values])
    test_stacked_bi_context_pre4 = stacked_bi_context4.predict([X_test_left, X_test_right, test_lexicon.values])
    test_stacked_bi_context_pre5 = stacked_bi_context5.predict([X_test_left, X_test_right, test_lexicon.values])
    test_context_lstm = np.hstack([test_stacked_bi_context_pre1, test_stacked_bi_context_pre2, test_stacked_bi_context_pre3,
                              test_stacked_bi_context_pre4, test_stacked_bi_context_pre5])

    print(np.shape(test_context_lstm))

    test_bi_lstm1_pre = lstm1.predict([X_test, test_lexicon.values])
    test_bi_lstm2_pre = lstm2.predict([X_test, test_lexicon.values])
    test_bi_lstm3_pre = lstm3.predict([X_test, test_lexicon.values])
    test_bi_lstm4_pre = lstm4.predict([X_test, test_lexicon.values])
    test_bi_lstm5_pre = lstm5.predict([X_test, test_lexicon.values])
    test_lstm = np.hstack([test_bi_lstm1_pre, test_bi_lstm2_pre, test_bi_lstm3_pre, test_bi_lstm4_pre, test_bi_lstm5_pre])
    print(np.shape(test_lstm))

    test_bi_gru1_pre = gru1.predict([X_test, test_lexicon.values])
    test_bi_gru2_pre = gru2.predict([X_test, test_lexicon.values])
    test_bi_gru3_pre = gru3.predict([X_test, test_lexicon.values])
    test_bi_gru4_pre = gru4.predict([X_test, test_lexicon.values])
    test_bi_gru5_pre = gru5.predict([X_test, test_lexicon.values])
    test_gru = np.hstack([test_bi_gru1_pre, test_bi_gru2_pre, test_bi_gru3_pre, test_bi_gru4_pre, test_bi_gru5_pre])
    print(np.shape(test_gru))

    test_capsulenet1_pre = capsulenet1.predict([X_test, test_lexicon.values])
    test_capsulenet2_pre = capsulenet2.predict([X_test, test_lexicon.values])
    test_capsulenet3_pre = capsulenet3.predict([X_test, test_lexicon.values])
    test_capsulenet4_pre = capsulenet4.predict([X_test, test_lexicon.values])
    test_capsulenet5_pre = capsulenet5.predict([X_test, test_lexicon.values])
    test_capsulenet = np.hstack([test_capsulenet1_pre, test_capsulenet2_pre, test_capsulenet3_pre, test_capsulenet4_pre,
                            test_capsulenet5_pre])
    print(np.shape(test_capsulenet))
    data_processed = os.path.join('pickle', 'test_stacking_data.pickle')
    pickle.dump([test_context_lstm, test_lstm, test_gru, test_capsulenet], open(data_processed, 'wb'))


if __name__ == '__main__':
    get_stacking_data()
    data_processed = os.path.join('pickle', 'train_stacking_data.pickle')
    context_lstm, lstm, gru, capsulenet, y_trial = pickle.load(open(data_processed, 'rb'))
    train_data = np.hstack([context_lstm, lstm, gru, capsulenet])

    data_processed = os.path.join('pickle', 'test_stacking_data.pickle')
    test_context_lstm, test_lstm, test_gru, test_capsulenet = pickle.load(open(data_processed, 'rb'))
    test_data = np.hstack([test_context_lstm, test_lstm, test_gru, test_capsulenet])

    # C Penalty parameter C of the error term
    # kernel Specifies the kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ (default=rbf)
    clf = SVC()
    save_file = os.path.join('result', 'svm.csv')
    clf.fit(train_data, trial_label.values)
    y_pred = clf.predict(test_data)
    result_output = pd.DataFrame(data={"sentiment": y_pred})
    result_output.to_csv(save_file, index=False, quoting=3)