# max len = 56
from __future__ import print_function
from __future__ import absolute_import

import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, GRU, Bidirectional, \
    Input, Convolution1D, MaxPooling1D, Flatten
from keras.layers.merge import concatenate
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.callbacks import EarlyStopping

from mytool import *

# maxlen = 56
batch_size = 1024
nb_epoch = 40  # trial set most accuracy epoch=13  val_acc=0.4736
hidden_dim = 128
kernel_size = 5
nb_filter = 60


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
        sent = get_idx_from_sent(rev['text'], word_idx_map)
        y = rev['y']

        if rev['split'] == 1:
            X_train.append(sent)
            y_train.append(y)
        elif rev['split'] == 0:
            X_dev.append(sent)
            y_dev.append(y)
        elif rev['split'] == -1:
            X_test.append(sent)

    X_train = sequence.pad_sequences(np.array(X_train), maxlen=maxlen)
    X_dev = sequence.pad_sequences(np.array(X_dev), maxlen=maxlen)
    X_test = sequence.pad_sequences(np.array(X_test), maxlen=maxlen)
    # X_valid = sequence.pad_sequences(np.array(X_valid), maxlen=maxlen)
    y_train = np_utils.to_categorical(np.array(y_train))
    # y_dev = np_utils.to_categorical(np.array(y_dev))
    # y_valid = np.array(y_valid)

    return [X_train, X_test, X_dev, y_train, y_dev]



if __name__ == '__main__':
    pickle_file = os.path.join('pickle', 'waasa_CNN_glove_data.pickle')

    auxiliary_data = pd.read_csv('./wassa2018_data/tran_format_review_43.csv')

    auxiliary_data = auxiliary_data.values[:, 1:len(auxiliary_data) + 1]

    revs, W, word_idx_map, vocab, maxlen = pickle.load(open(pickle_file, 'rb'))

    X_train, X_test, X_dev, y_train, y_dev = make_idx_data(revs, word_idx_map, maxlen=maxlen)

    n_train_sample = X_train.shape[0]

    n_test_sample = X_test.shape[0]

    len_sentence = X_train.shape[1]     # 200

    max_features = W.shape[0]

    num_features = W.shape[1]               # 400
    print(type(X_train), type(auxiliary_data))

    model1 = Sequential()
    model1.add(Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, weights=[W], trainable=False))
    model1.add(Convolution1D(filters=128,
                            kernel_size=kernel_size,
                            padding='valid',
                            activation='relu',
                            strides=1
                            ))
    model1.add(MaxPooling1D(pool_size=2))
    model1.add(Flatten())

    model2 = Sequential()
    model2.add( Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, weights=[W], trainable=False))
    model2.add(Convolution1D(filters=128,
                             kernel_size=kernel_size,
                             padding='valid',
                             activation='relu',
                             strides=1
                             ))
    model2.add(MaxPooling1D(pool_size=2))
    model2.add(Flatten())


    modelall = Sequential()
    modelall.add(concatenate([model1, model2]))
    modelall.add(Dense(256, activation='relu'))
    modelall.add(Dense(128, activation='relu'))
    modelall.add(Dense(6, activation='softmax'))

    modelall.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    early_stopping = EarlyStopping(monitor='val_acc', patience=3, verbose=2)

    modelall.fit([X_train, auxiliary_data], y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1,
              callbacks=[early_stopping])
    y_pred = modelall.predict(X_test, batch_size=batch_size)
    y_pred = np.argmax(y_pred, axis=1)

    result_output = pd.DataFrame(data={'sentiment': y_pred})

    # Use pandas to write the comma-separated output file
    # result_output.to_csv("./result/bi-lstm.csv", index=False, quoting=3)

    result_output.to_csv("./result/cnn-lstm.csv", index=False, quoting=3)

