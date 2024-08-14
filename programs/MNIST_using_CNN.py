import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout

(xtrain, ytrain), (xtest, ytest) = mnist.load_data()
xtrain, xtest = xtrain / 255.0, xtest / 255.0
ytrain, ytest = to_categorical(ytrain), to_categorical(ytest)

model = Sequential([
    Conv2D(8, (3, 3), activation='relu', input_shape = (28, 28, 1)),
    MaxPooling2D(strides=(2,2)),
    Conv2D(16, (3,3), activation='relu'),
    MaxPooling2D(strides=(2,2)),
    Conv2D(32, (3,3), activation='relu'),
    Flatten(),
    Dense(32, activation='relu'),
    Dropout(0.25),
    Dense(10, activation='softmax')])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(xtrain, ytrain, batch_size=64, epochs=10, validation_data=(xtest, ytest))

loss, accuracy = model.evaluate(xtest, ytest)
print(f"Accuracy: {round(accuracy,4)}")
print(f"Loss: {round(loss,4)}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(history.history['accuracy'], label='Training Accuracy')
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_title('Training and Validation Accuracy')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax2.plot(history.history['loss'], label='Training Loss')
ax2.plot(history.history['val_loss'], label='Validation Loss')
ax2.set_title('Training and Validation Loss')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.legend()
plt.tight_layout()
plt.show()
