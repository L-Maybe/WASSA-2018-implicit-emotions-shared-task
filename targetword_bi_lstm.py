# max len = 56
from __future__ import print_function
from __future__ import absolute_import

import os
import pickle
import numpy as np
import pandas as pd

from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, LSTM, GRU, Bidirectional, Input, Flatten, Convolution1D
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
from keras.utils import np_utils

# maxlen = 56
batch_size = 830
nb_epoch = 40
hidden_dim = 120
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
    y_dev = np_utils.to_categorical(np.array(y_dev))
    # y_valid = np.array(y_valid)

    return [X_train, X_test, X_dev, y_train, y_dev]


if __name__ == '__main__':
    pickle_file = os.path.join('pickle', 'targetwords.pickle')

    revs, W, word_idx_map, vocab, maxlen = pickle.load(open(pickle_file, 'rb'))

    X_train, X_test, X_dev, y_train, y_dev = make_idx_data(revs, word_idx_map, maxlen=maxlen)

    n_train_sample = X_train.shape[0]

    n_test_sample = X_test.shape[0]

    len_sentence = X_train.shape[1]  # 200

    max_features = W.shape[0]

    num_features = W.shape[1]  # 400

    # Keras Model
    # this is the placeholder tensor for the input sequence
    sequence = Input(shape=(maxlen,), dtype='int32')

    embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, mask_zero=True,
                         weights=[W], trainable=False)(sequence)
    # embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, weights=[W], trainable=False) (sequence)
    embedded = Dropout(0.3)(embedded)

    # convolution = Convolution1D(filters=nb_filter,
    #                             kernel_size=kernel_size,
    #                             padding='valid',
    #                             activation='relu',
    #                             strides=1
    #                             )(embedded)

    # bi-lstm
    embedded = Bidirectional(LSTM(hidden_dim, recurrent_dropout=0.25, return_sequences=True))(embedded)
    enc = Bidirectional(LSTM(hidden_dim, recurrent_dropout=0.25))(embedded)

    # # gru
    # enc = Bidirectional(GRU(hidden_dim//2, recurrent_dropout=0.25))(convolution)

    # att = AttentionM()(enc)

    # fc1_dropout = Dropout(0.3)(enc)
    # flatten = Flatten()(fc1_dropout)
    fc1 = Dense(256, activation="relu")(enc)
    fc1_dropout = Dropout(0.3)(fc1)
    fc2 = Dense(256, activation="relu")(fc1_dropout)
    fc3 = Dense(256, activation="relu")(fc2)
    # fc2_dropout = Dropout(0.3)(fc1)

    output = Dense(6, activation='softmax')(fc3)
    model = Model(inputs=sequence, outputs=output)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    early_stopping = EarlyStopping(monitor='val_acc', patience=3, verbose=1)

    model.fit(X_train, y_train, validation_data=[X_dev, y_dev], batch_size=batch_size, epochs=nb_epoch, verbose=1,
              callbacks=[early_stopping])
    y_pred = model.predict(X_test, batch_size=batch_size)
    y_pred = np.argmax(y_pred, axis=1)

    result_output = pd.DataFrame(data={"sentiment": y_pred})

    # Use pandas to write the comma-separated output file
    # result_output.to_csv("./result/bi-lstm.csv", index=False, quoting=3)

    result_output.to_csv("./result/target-bi-lstm.csv", index=False, quoting=3)