import time
import numpy as np
import tensorflow as tf
from tabulate import tabulate
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Flatten, Dense, Dropout

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = to_categorical(y_train), to_categorical(y_test)

model = Sequential([
            Flatten(input_shape=(32, 32, 3)),
            Dense(256, activation='relu'),
            Dense(128, activation='relu'),
            Dense(64, activation ='relu'),
            Dense(10, activation='softmax')])

result = [['Accuracy'], ['Loss'], ['Time Taken']]

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
st = time.time()
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))
et = time.time()
test_loss, test_accuracy = model.evaluate(x_test, y_test)
result[0].append(test_accuracy)
result[1].append(test_loss)
result[2].append(et-st)

model = Sequential([
            Flatten(input_shape=(32, 32, 3)),
            Dense(256, kernel_initializer='glorot_uniform', activation='relu'),
            Dense(128, kernel_initializer='glorot_uniform', activation='relu'),
            Dense(64, kernel_initializer='glorot_uniform', activation ='relu'),
            Dense(10, activation='softmax')])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
st = time.time()
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))
et = time.time()
test_loss, test_accuracy = model.evaluate(x_test, y_test)
result[0].append(test_accuracy)
result[1].append(test_loss)
result[2].append(et-st)

model = Sequential([
            Flatten(input_shape=(32, 32, 3)),
            Dense(256, kernel_initializer='he_uniform', activation='relu'),
            Dense(128, kernel_initializer='he_uniform', activation='relu'),
            Dense(64, kernel_initializer='he_uniform', activation ='relu'),
            Dense(10, activation='softmax')])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
st = time.time()
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))
et = time.time()
test_loss, test_accuracy = model.evaluate(x_test, y_test)
result[0].append(test_accuracy)
result[1].append(test_loss)
result[2].append(et-st)

model = Sequential([
            Flatten(input_shape=(32, 32, 3)),
            Dense(256, activation='relu'),
            Dropout(0.2),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(64, activation ='relu'),
            Dropout(0.2),
            Dense(10, activation='softmax')])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
st = time.time()
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))
et = time.time()
test_loss, test_accuracy = model.evaluate(x_test, y_test)
result[0].append(test_accuracy)
result[1].append(test_loss)
result[2].append(et-st)

model = Sequential([
            Flatten(input_shape=(32, 32, 3)),
            Dense(256, activation='relu', kernel_regularizer=regularizers.l1_l2(0.01)),
            Dense(128, activation='relu', kernel_regularizer=regularizers.l1_l2(0.01)),
            Dense(64, activation ='relu', kernel_regularizer=regularizers.l1_l2(0.01)),
            Dense(10, activation='softmax')])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
st = time.time()
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))
et = time.time()
test_loss, test_accuracy = model.evaluate(x_test, y_test)
result[0].append(test_accuracy)
result[1].append(test_loss)
result[2].append(et-st)

headers = ['Metrics', 'Baseline', 'Xavier', 'Kaiming', 'Dropout', 'L1 L2']
print(tabulate(result, headers=headers, floatfmt=".2f"))
max_accuracy_index = result[0][1:].index(max(result[0][1:]))
min_time_index = result[2][1:].index(min(result[2][1:]))
print(f"\nConfiguration with Maximum Accuracy ({max(result[0][1:]):.2f}): { headers[1:][max_accuracy_index]}")
print(f"Configuration with Minimum Convergence Time ({min(result[2][1:]):.2f}): { headers[1:][min_time_index]}")
