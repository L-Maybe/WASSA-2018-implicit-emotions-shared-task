# max len = 56
from __future__ import print_function
from __future__ import absolute_import

import pickle
import numpy as np
import pandas as pd

from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, LSTM, GRU, Bidirectional, \
    Input, Convolution1D, MaxPooling1D, Flatten
import keras
from keras.layers.merge import concatenate
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.callbacks import EarlyStopping

from mytool import *

# maxlen = 56
batch_size = 830
nb_epoch = 20  # trial set most accuracy epoch=13  val_acc=0.4736
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
    # X_dev = sequence.pad_sequences(np.array(X_dev), maxlen=maxlen)
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

    # Keras Model
    # this is the placeholder tensor for the input sequence
    # input 75
    sequence = Input(shape=(maxlen, ), dtype='int32', name='first')

    # embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, mask_zero=True, weights=[W], trainable=False) (sequence)
    embedded = Embedding(input_dim=max_features, output_dim=num_features, input_length=maxlen, weights=[W], trainable=False) (sequence)
    embedded = Dropout(0.3)(embedded)

    # LSTM
    # hidden = LSTM(hidden_dim, recurrent_dropout=0.25) (embedded)
    convolution = Convolution1D(filters=256,
                            kernel_size=kernel_size,
                            padding='valid',
                            activation='relu',
                            strides=1
                            )(embedded)

    convolution1 = Convolution1D(filters=512,
                            kernel_size=kernel_size,
                            padding='valid',
                            activation='relu',
                            strides=1)(convolution)

    dropout = Dropout(0.3)(convolution1)
    maxpooling = MaxPooling1D(pool_size=2)(dropout)
    # flatten = Flatten()(maxpooling)
    lstm = LSTM(hidden_dim // 2, return_sequences=True)(maxpooling)
    maxpooling1 = MaxPooling1D(pool_size=2)(lstm)
    lstm1 = LSTM(hidden_dim, return_sequences=True)(maxpooling1)
    # maxpooling1 = MaxPooling1D(pool_size=2)(lstm1)
    flatten = Flatten()(lstm1)

    auxiliary_input = Input(shape=(43,), name='aux_input')  # 新加入的一个Input,5维度
    x = keras.layers.concatenate([flatten, auxiliary_input])  # 组合起来，对应起来
    print(x)
    dense = Dense(600, activation='relu')(x)
    dropout = Dropout(0.4)(dense)
    dense = Dense(500, activation='relu')(dropout)
    dropout = Dropout(0.4)(dense)
    dense = Dense(500, activation='relu')(dropout)

    # flatten = Flatten()(dense)
    #
    # auxiliary_input = Input(shape=(43,), name='auxiliary')
    # concat = concatenate([auxiliary_input, flatten])

    output = Dense(6, activation='softmax')(dense)
    model = Model(inputs=[sequence, auxiliary_input], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    early_stopping = EarlyStopping(monitor='val_acc', patience=3, verbose=2)


    model.fit([X_train, auxiliary_data[0:144404]], y_train, validation_split=0.06, batch_size=batch_size, epochs=nb_epoch, verbose=1,
              callbacks=[early_stopping])
    y_pred = model.predict(X_test, batch_size=batch_size)
    y_pred = np.argmax(y_pred, axis=1)

    result_output = pd.DataFrame(data={'sentiment': y_pred})

    # Use pandas to write the comma-separated output file
    # result_output.to_csv("./result/bi-lstm.csv", index=False, quoting=3)

    result_output.to_csv("./result/cnn-lstm.csv", index=False, quoting=3)

