import os.path
import numpy as np
#import tensorflow.python.keras as keras
#from tensorflow.python.keras.models import Sequential
#from tensorflow.python.keras.layers import Dense, Input, Activation, Dropout
#from tensorflow.python.keras.layers import Conv2D, Flatten, MaxPooling2D
#from tensorflow.python.keras.layers.convolutional import ZeroPadding2D
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
data_filepath = 'games/ficsgamesdb_202001_standard2000_nomovetimes_145127_new_13planes.npy'
#data_filepath = 'games/adams_13planes.npy'
model_filepath = 'models/cnn_13p_2021'
planes = 6*2+1
batch_size = 64
epochs = 5 
with open(data_filepath, 'rb') as f:
    X = np.load(f)
    Y = np.load(f)
samples = X.shape[0]
input_shape = planes,8,8
train_samples = int(samples * 0.9)
x_train = X[:train_samples]
y_train = Y[:train_samples]


x_test = X[train_samples:]
y_test = Y[train_samples:]

x_train = x_train.reshape(train_samples, planes,8,8)
y_train = y_train.reshape(train_samples, 64)

x_test = x_test.reshape(samples-train_samples, planes,8,8)
y_test = y_test.reshape(samples-train_samples, 64)
#y_train = to_categorical(y_train)
#y_train = np.array([y_train])
np.random.seed(123)
#x_train = np.array([x_train])
#inputs = Input(shape=(8,8))
#y_train = np.array([y_train])
#x = Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(inputs)
#x = MaxPooling2D(pool_size=(3, 3))(x)
#x = Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)
#x = MaxPooling2D(pool_size=(3, 3))(x)
#x = Flatten()(inputs)
#x = Dense(128, activation="relu")(inputs)
#outputs = Dense(64, activation="softmax")(x)
#model = keras.Model(inputs, outputs)
model = keras.Sequential()

if os.path.exists(model_filepath):
	model = keras.models.load_model(model_filepath)
else:
#model.ZeroPadding2D(padding=3, input_shape=(8,8,1))
	model.add(Conv2D(192, kernel_size=(5,5), padding='same', activation='relu', input_shape=(planes,8,8)))
	model.add(Conv2D(192, kernel_size=(3,3), padding='same', activation='relu'))
	model.add(Conv2D(192, kernel_size=(3,3), padding='same', activation='relu'))
	model.add(Conv2D(192, kernel_size=(3,3), padding='same', activation='relu'))
	model.add(Flatten())
	model.add(Dense(512, activation='relu'))
	model.add(Dense(64, activation='softmax'))

	#model.add(Dense(512, activation='relu'))
	#model.add(Dense(64, activation='softmax'))
	#model.add(Dense(64, activation='softmax'))
	#model.add(Dense(256, activation='sigmoid', input_shape=(64,)))
	#model.add(Dense(64, activation='sigmoid'))
	#model.add(Activation('softmax'))

	model.summary()
	#model.compile(optimizer="adam", loss="categorical_crossentropy")
	lr_schedule = keras.optimizers.schedules.ExponentialDecay(
	initial_learning_rate=0.1,
	decay_steps=10000,
	decay_rate=0.9)
	optimizer = keras.optimizers.SGD(learning_rate=lr_schedule)
	model.compile(optimizer=optimizer, loss="categorical_crossentropy")
	#print("lr", keras.eval(model.optimizer.lr))
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

score = model.evaluate(x_test, y_test)
model.save(model_filepath)
print(score)


