from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import logging
import pickle
import numpy as np
import pandas as pd

from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, LSTM, GRU, Bidirectional, Input, Concatenate, TimeDistributed
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from keras.utils import np_utils
from datetime import datetime
from time import time
import matplotlib.pyplot as plt
from metrics import f1

batch_size = 830
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
                  epochs=epoch,
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
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    logging.info('loading data...')
    # pickle_file = os.path.join('pickle', 'wassa_origin_tweet_glove.pickle3')
    pickle_file = os.path.join('pickle', 'parse_staford_no_letters_only_no_stopword_v3.pickle')

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

    char_vocab_size = char_maxlen
    char_embedding_dim = 25
    char_hidden_dim = 120

    # Keras Model
    # this is the placeholder tensor for the input sequence
    sequence = Input(shape=(maxlen,), dtype='int32')
    lex_input = Input(shape=(43,), dtype='float32')
    char_input = Input(shape=(maxlen, char_maxlen, ), dtype='int32')

    embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, mask_zero=True,
                         weights=[W], trainable=False)(sequence)
    char_embeddings = Embedding(input_dim=char_vocab_size,
                                output_dim=char_embedding_dim,
                                mask_zero=True,
                                name='char_embedding')(char_input)

    char_embeddings = TimeDistributed(Bidirectional(LSTM(char_hidden_dim)))(char_embeddings)
    embedded = Concatenate()([embedded, char_embeddings])

    # embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, weights=[W], trainable=False) (sequence)
    embedded = Dropout(0.25)(embedded)

    # LSTM
    # hidden = Bidirectional(LSTM(hidden_dim//2, recurrent_dropout=0.25)) (embedded)

    # GRU
    # hidden = Bidirectional(GRU(hidden_dim//2, recurrent_dropout=0.25)) (embedded)

    # stacked LSTM
    hidden = Bidirectional(LSTM(hidden_dim, recurrent_dropout=0.25, return_sequences=True))(embedded)
    hidden = Bidirectional(LSTM(hidden_dim, recurrent_dropout=0.25))(hidden)

    # stacked GRU
    # hidden = Bidirectional(GRU(hidden_dim//2, recurrent_dropout=0.25, return_sequences=True)) (embedded)
    # hidden = Bidirectional(GRU(hidden_dim//2, recurrent_dropout=0.25)) (hidden)

    dense = Concatenate(axis=-1)([hidden, lex_input])
    # dense = Dense(256,activation='relu')(dense)
    # dropout = Dropout(0.25)(dense)
    # dense = Dense(128, activation='relu')(dropout)
    # dense = Dense(64, activation='relu')(dense)
    output = Dense(6, activation='softmax')(dense)
    model = Model(inputs=[sequence, lex_input, char_input], outputs=output)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', f1])
    h = train(model, batch_size, nb_epoch, X_train, y_train, X_trial, y_trial, lex_train, lex_trial)

    # model.fit(X_train, y_train, validation_data=[X_trial, y_trial], batch_size=batch_size, epochs=nb_epoch, verbose=1)
    y_pred = model.predict([X_trial, lex_trial], batch_size=batch_size)
    y_pred = np.argmax(y_pred, axis=1)

    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

    print(precision_score(y_trial, y_pred, average='macro'))
    print(recall_score(y_trial, y_pred, average='macro'))
    print(accuracy_score(y_trial, y_pred))
    print(f1_score(y_trial, y_pred, average='macro'))

    accuracy_curve(h)