import numpy as np
import tensorflow.python.keras as keras

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Input, Activation
from tensorflow.python.keras.layers import Conv2D, Flatten, MaxPooling2D
import generate_games



#game_generator = generate_games.GameGenerator()
#[X, Y] = game_generator.generate()

with open('games.npy', 'rb') as f:
    X = np.load(f)
    Y = np.load(f)
samples = X.shape[0]

train_samples = int(samples * 0.9)
x_train = X[:train_samples]
y_train = Y[:train_samples]


x_test = X[train_samples:]
y_test = Y[train_samples:]

x_train = x_train.reshape(train_samples, 64)
y_train = y_train.reshape(train_samples, 64)

x_test = x_test.reshape(samples-train_samples, 64)
y_test = y_test.reshape(samples-train_samples, 64)
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
model.add(Dense(256, activation='sigmoid', input_shape=(64,)))
model.add(Dense(64, activation='sigmoid'))
model.add(Activation('softmax'))

model.summary()
model.compile(optimizer="adam", loss="mean_squared_error")

history = model.fit(x_train, y_train, batch_size=32, epochs=10)

score = model.evaluate(x_test, y_test)
model.save("models/first")
print(score)

test_board = np.array([x_train[4]])
print(test_board)

move_prediction = model.predict(test_board)

rowMsg = ""
for c in range(8):
	for r in range(8):
		rowMsg += str(move_prediction[0][r+c*8])+" "
	print(rowMsg)
	rowMsg = ""