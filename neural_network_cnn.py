# -*- coding: utf-8 -*-
# @Time    : 2018/3/9 23:06
# @Author  : David
# @Email   : mingren4792@126.com
# @File    : neural_network_cnn.py
# @Software: PyCharm

import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPool2D, Flatten
from keras.optimizers import Adam

np.random.seed(1377)  # for reproducibility
# download the mnist ro the path '~/.keras/datasets/'
# if it is the first time to be called
# x shape(60000  28X28), y shape(10000,)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(np.shape(x_train))
# data pre-processing
# -1为sample的个数， 1为通道（黑白）， 28*28像素
x_train = x_train.reshape(-1, 1, 28, 28)
x_test = x_test.reshape(-1, 1, 28, 28)
# One-hot encoding
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

# Another way to build your cnn
model = Sequential()

# Conv layer 1 output shape(32, 28, 28)
model.add(Convolution2D(
    filters=32,  # 过滤器个数
    kernel_size=(5, 5),  # filter的大小
    padding='same',  # padding method
    input_shape=(1,  # channels
                 28, 28)
))
model.add(Activation('relu'))

# Pooling layer 1 (max pooling) output shape(32, 14, 14)
model.add(MaxPool2D(
    pool_size=(2, 2),
    strides=(2, 2),
    padding='same'
))

# Conv layer 2 output shape(64, 14, 14)
model.add(Convolution2D(
    filters=64,
    kernel_size=(5, 5),
    padding='same'
))
model.add(Activation('relu'))

# Pooling layer2 (max pooling) output shape(64, 7, 7)
model.add(MaxPool2D(
    pool_size=(2, 2),
    padding='same'
))

# Fully connect layer1 input shape (64, 7, 7) = (3136), output shape (1024)
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))

# Full connected layer 2 to shape (10） for 10 classes
model.add(Dense(10))
model.add(Activation('softmax'))

# Another way to define your optimizer
adam = Adam(lr=1e-4)

# we add metrics to get more results you want to see
model.compile(
    optimizer=adam,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print('Training ===========')
# Another way to train the model
model.fit(x_train, y_train, epochs=20, batch_size=64)

print('\nTesting ==========')
# Evaluate the model with the metics we defined earlier
loss, accuracy = model.evaluate(x_test, y_test)

print('\ntest loss:', loss)
print('\ntest accuracy:', accuracy)