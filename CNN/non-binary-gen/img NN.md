```python
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load MNIST dataset (as an example)
from tensorflow.keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess the data
X_train = X_train.reshape((X_train.shape[0], -1)) / 255.0
X_test = X_test.reshape((X_test.shape[0], -1)) / 255.0

# Convert labels to one-hot encoding
num_classes = 10
y_train_onehot = np.eye(num_classes)[y_train]
y_test_onehot = np.eye(num_classes)[y_test]

# Neural Network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.weights1 = np.random.rand(input_size, hidden_size)
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.rand(hidden_size, output_size)
        self.bias2 = np.zeros((1, output_size))
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.layer1 = self.sigmoid(np.dot(X, self.weights1) + self.bias1)
        self.output = self.sigmoid(np.dot(self.layer1, self.weights2) + self.bias2)
        return self.output

    def backward(self, X, y):
        loss = y - self.output

        output_delta = loss * self.sigmoid_derivative(self.output)
        layer1_loss = output_delta.dot(self.weights2.T)
        layer1_delta = layer1_loss * self.sigmoid_derivative(self.layer1)

        self.weights2 += self.layer1.T.dot(output_delta) * self.learning_rate
        self.bias2 += np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate
        self.weights1 += X.T.dot(layer1_delta) * self.learning_rate
        self.bias1 += np.sum(layer1_delta, axis=0, keepdims=True) * self.learning_rate

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y)

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

# Train the neural network
input_size = X_train.shape[1]
hidden_size = 64
output_size = num_classes
learning_rate = 0.01
epochs = 50

model = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)
model.train(X_train, y_train_onehot, epochs)

# Evaluate on the test set
predictions = model.predict(X_test)
accuracy = np.mean(predictions == y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save the model (weights and biases)
model_params = {
    "weights1": model.weights1,
    "bias1": model.bias1,
    "weights2": model.weights2,
    "bias2": model.bias2,
}

np.savez("simple_neural_network_model.npz", **model_params)
print("Model saved.")

# Visualize predictions
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(28, 28), cmap="gray")
    ax.set_title(f"Predicted: {predictions[i]}, Actual: {y_test[i]}")
    ax.axis("off")

plt.show()

```
Bing
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from PIL import Image
import os

# Load images and labels
def load_images(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename))
        if img is not None:
            images.append(np.array(img))
            labels.append(filename.split('_')[0])  # Assumes labels are the first part of the filename
    return images, labels

# Define the neural network
class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 4)
        self.weights2 = np.random.rand(4, 1)
        self.y = y
        self.output = np.zeros(y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        d_weights2 = np.dot(self.layer1.T, 2*(self.y - self.output) * sigmoid_derivative(self.output))
        d_weights1 = np.dot(self.input.T, np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1))

        self.weights1 += d_weights1
        self.weights2 += d_weights2

# Sigmoid function
def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

# Derivative of sigmoid
def sigmoid_derivative(x):
    return x * (1.0 - x)

# Load images
images, labels = load_images('path_to_your_images')

# Convert to NumPy arrays and normalize images
images = np.array(images) / 255.0
labels = np.array(labels)

# Flatten images
images = images.reshape((images.shape[0], -1))

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Create and train the neural network
nn = NeuralNetwork(X_train, y_train)
for i in range(1500):
    nn.feedforward()
    nn.backprop()

# Predictions for testing set
y_pred = nn.feedforward(X_test)

# Evaluate model
print(confusion_matrix(y_test, y_pred))

# Save model weights
np.save('model_weights1.npy', nn.weights1)
np.save('model_weights2.npy', nn.weights2)

```
BardAi
```

```