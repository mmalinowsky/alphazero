#!/usr/bin/python3

import os.path
import numpy as np

from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *



data_filepath = 'games/1000games.npy'
model_filepath = 'models/latest_model.h5'
planes = 6 * 2 + 1
batch_size = 128
epochs = 100
with open(data_filepath, 'rb') as f:
    X = np.load(f)
    Y = np.load(f)
samples = X.shape[0]
input_shape = (planes, 8, 8)
train_samples = int(samples * 0.9)
x_train = X[:train_samples]
y_train = Y[:train_samples]

x_test = X[train_samples:]
y_test = Y[train_samples:]

x_train = x_train.reshape(train_samples, planes, 8, 8)
y_train = y_train.reshape(train_samples, 64)

x_test = x_test.reshape(samples - train_samples, planes, 8, 8)
y_test = y_test.reshape(samples - train_samples, 64)


train =  [x_train, y_train]
test_data = [x_test, y_test]
np.random.seed(123)


model = keras.Sequential()

if os.path.exists(model_filepath):
    model = keras.models.load_model(model_filepath)
else:


    model.add(Conv2D(192, kernel_size=(5, 5), padding='same',
              activation='relu', input_shape=(planes, 8, 8)))
    model.add(Dropout(rate=0.6))
    for i in range(2,12):
        model.add(Conv2D(192, kernel_size=(3, 3), padding='same',
                  activation='relu'))
        model.add(Dropout(rate=0.6))
    model.add(Conv2D(1, kernel_size=(1, 1), padding='same',
              activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(rate=0.6))
    model.add(Dense(64, activation='softmax'))



model.summary()
if os.path.exists(model_filepath):
    model.load_weights(model_filepath)

    # model.compile(optimizer="adam", loss="categorical_crossentropy")


#lr_schedule = \
#    keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.05,
#        decay_steps=10000, decay_rate=0.9)
#optimizer = keras.optimizers.SGD(learning_rate=lr_schedule)

    # optimizer = keras.optimizers.Adam(learning_rate=0.05)

steps_per_epoch = 60000 // batch_size
validation_steps = 10000 // batch_size

#model.compile(optimizer=optimizer, steps_per_execution = 50, loss='categorical_crossentropy',
#              metrics=['accuracy'])
model.compile(optimizer='adam', steps_per_execution = 50, loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
    batch_size=batch_size,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=epochs,
    validation_data=test_data)

score = model.evaluate(x_test, y_test)
print ('Test loss:', score[0])
print ('Test accuracy:', score[1])

model.save(model_filepath)
