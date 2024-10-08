import numpy as np
import tensorflow as tf
from tabulate import tabulate
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Flatten, Dense

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = to_categorical(y_train), to_categorical(y_test)

hidden_units_list = [(256, 128, 64), (512, 256, 128), (1024, 512, 256)]
activation_list = ['relu', 'tanh', 'sigmoid']

accuracy_dict = {}
for hidden_units in hidden_units_list:
    accuracy_dict[hidden_units] = []
    for activation in activation_list:
        model = Sequential([
            Flatten(input_shape=(32, 32, 3)),
            Dense(hidden_units[0], activation=activation),
            Dense(hidden_units[1], activation=activation),
            Dense(hidden_units[2], activation =activation),
            Dense(10, activation='softmax')])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))
        _, test_accuracy = model.evaluate(x_test, y_test)
        accuracy_dict[hidden_units].append(round(test_accuracy * 100, 4))

table = []
for hidden_units, accuracies in accuracy_dict.items():
    table.append([hidden_units] + accuracies)

headers = ["Hidden units"] + activation_list
print(tabulate(table, headers=headers, floatfmt=".2f"))

max_accuracy = 0
best_config = None
for hidden_units, accuracies in accuracy_dict.items():
    for activation, accuracy in zip(activation_list, accuracies):
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            best_config = (hidden_units, activation)
print("\nConfiguration with highest accuracy is")
print(f"Hidden units: {best_config[0]}")
print(f"Activation: {best_config[1]}")
print(f"Test accuracy: {max_accuracy:.2f}")

num_images = 3
sample_images = x_train[:num_images]
predictions = model.predict(sample_images)
def plot_probability_meter(predictions, image):
    class_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship",
    "truck"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 2))
    axs[0].imshow(image)
    axs[0].axis('off')
    axs[1].barh(class_labels, predictions[0], color='blue')
    axs[1].set_xlim([0, 1])
    plt.tight_layout()
    plt.show()
    
for i in range(num_images):
    plot_probability_meter(predictions[i:i+1], sample_images[i])
