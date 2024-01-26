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
