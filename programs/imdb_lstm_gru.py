import numpy as np
import pandas as pd
from tabulate import tabulate
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM, Dense, Embedding, GRU, SimpleRNN

vocabulary = 10000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocabulary)

sequence_length = 200
x_train_padded = pad_sequences(x_train, maxlen=sequence_length)
x_test_padded = pad_sequences(x_test, maxlen=sequence_length)

lstm_model = Sequential([
    Embedding(input_dim=vocabulary, output_dim=128, input_length=sequence_length),
    LSTM(128),
    Dense(1, activation='sigmoid')])
lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

gru_model = Sequential([
    Embedding(input_dim=vocabulary, output_dim=128, input_length=sequence_length),
    GRU(128),
    Dense(1, activation='sigmoid')])
gru_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

rnn_model = Sequential([
    Embedding(input_dim=vocabulary, output_dim=128, input_length=sequence_length),
    SimpleRNN(128),
    Dense(1, activation='sigmoid')])
rnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

lstm_history = lstm_model.fit(x_train_padded, y_train, epochs=5, batch_size=64, validation_split=0.2)
gru_history = gru_model.fit(x_train_padded, y_train, epochs=5, batch_size=64, validation_split=0.2)
rnn_history = rnn_model.fit(x_train_padded, y_train, epochs=5, batch_size=64, validation_split=0.2)

lstm_loss, lstm_accuracy = lstm_model.evaluate(x_test_padded, y_test)
gru_loss, gru_accuracy = gru_model.evaluate(x_test_padded, y_test)
rnn_loss, rnn_accuracy = rnn_model.evaluate(x_test_padded, y_test)

r = lambda x: round(x, 2)
data = [
    ["Accuracy", r(rnn_accuracy), r(lstm_accuracy), r(gru_accuracy)],
    ["Loss", r(rnn_loss), r(lstm_loss), r(gru_loss)],
]

headers = ["Metric", "RNN", "LSTM", "GRU"]
print(tabulate(data, headers=headers, tablefmt="pretty"))

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(lstm_history.history['accuracy'], label='LSTM Training Accuracy ')
plt.plot(lstm_history.history['val_accuracy'], label='LSTM Validation Accuracy')
plt.plot(gru_history.history['accuracy'], label='GRU Training Accuracy ')
plt.plot(gru_history.history['val_accuracy'], label='GRU Validation Accuracy')
plt.plot(rnn_history.history['accuracy'], label='RNN Training Accuracy ')
plt.plot(rnn_history.history['val_accuracy'], label='RNN Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(lstm_history.history['loss'], label='LSTM Training Loss ')
plt.plot(lstm_history.history['val_loss'], label='LSTM Validation Loss')
plt.plot(gru_history.history['loss'], label='GRU Training Loss ')
plt.plot(gru_history.history['val_loss'], label='GRU Validation Loss')
plt.plot(rnn_history.history['loss'], label='RNN Training Loss ')
plt.plot(rnn_history.history['val_loss'], label='RNN Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))

