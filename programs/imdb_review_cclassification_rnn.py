import numpy as np
import pandas as pd
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocabulary = 10000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocabulary)
sequence_length = 200
x_train_padded = pad_sequences(x_train, maxlen=sequence_length)
x_test_padded = pad_sequences(x_test, maxlen=sequence_length)
model = Sequential([
    Embedding(input_dim=vocabulary, output_dim=128, input_length=sequence_length),
    LSTM(128),
    Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train_padded, y_train, epochs=5, batch_size=64, validation_split=0.2)

loss, accuracy = model.evaluate(x_test_padded, y_test)
print(f'Training Loss: {loss}')
print(f'Training Accuracy: {accuracy}')

test_sequence = np.reshape(x_test_padded[1], (1, -1))
prediction = model.predict(test_sequence)
if round(prediction[0][0]) == 1.0:
    print('Positive Review')
else:
    print('Negative Review')
