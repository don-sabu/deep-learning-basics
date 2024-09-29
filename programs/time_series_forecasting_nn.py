import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

# Load and preprocess data
df = pd.read_csv('nifty.csv').iloc[::-1]
df.drop(['Open', 'High', 'Low'], axis=1, inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
df_chg = df.set_index('Date', drop=True)

split_date = pd.Timestamp('2017-12-27')
df1 = df_chg['Close']
train = df1.loc[:split_date]
test = df1.loc[split_date:]

# Plotting train and test data
plt.figure(figsize=(15, 8))
ax = train.plot()
test.plot(ax=ax)
plt.legend(['Train', 'Test'])
plt.title('Train and Test Data')
plt.show()

# Prepare training and testing data
train_processed = df_chg['Close'].loc[:split_date].values.reshape(-1, 1)
test_processed = df_chg['Close'].loc[split_date:].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(-1, 1))
train_sc = scaler.fit_transform(train_processed)
test_sc = scaler.transform(test_processed)

X_train = train_sc[:-1]
y_train = train_sc[1:]
X_test = test_sc[:-1]
y_test = test_sc[1:]

# Define and train the model
model = Sequential()
model.add(Dense(12, input_dim=1, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)
history = model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=1, callbacks=[early_stop], shuffle=False)

# Evaluate and predict
y_pred_test = model.predict(X_test)
loss = model.evaluate(X_test, y_test, batch_size=1)
print(f'Loss: {loss}')

# Inverse transform the predictions and true values
X_test_inv = scaler.inverse_transform(X_test)
y_pred_test_inv = scaler.inverse_transform(y_pred_test)
y_test_inv = scaler.inverse_transform(y_test)


test_dates = df_chg.loc[split_date:].index

# Plot predictions vs true values with dates
plt.figure(figsize=(15, 8))
plt.plot(test_dates[1:], y_test_inv, label='True')
plt.plot(test_dates[1:], y_pred_test_inv, label='Predicted')
plt.title("ANN's Prediction vs True Values")
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()



plt.figure(figsize=(15, 8))

# Plot Training Data
plt.plot(df_chg.index, df_chg['Close'], label='All Data', color='grey', alpha=0.5)

# Plot Train Data
plt.plot(train.index, train, label='Train', color='blue')

# Plot True and Predicted Values
plt.plot(test_dates[1:], y_test_inv, label='True Values', color='green', linestyle='--')
plt.plot(test_dates[1:], y_pred_test_inv, label='Predicted Values', color='red', linestyle='--')

# Add vertical line to indicate split date
plt.axvline(x=split_date, color='black', linestyle='--', label='Split Date')

# Add legends and labels
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.title("Train Data, True Values, and Predictions")
plt.show()
