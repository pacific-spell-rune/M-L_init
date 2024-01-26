import numpy as np
import matplotlib.pyplot as plt
import os

# ... (Previous code for model definition and forward pass)
# Load images from folders
def load_images(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img = plt.imread(os.path.join(folder_path, filename))
        img = img.flatten()  # Flatten to 1D array
        images.append(img)
    return np.array(images)

# Load training and testing data
X_train = load_images("training_images")
y_train = np.array([int(folder) for folder in os.listdir("training_images")])  # Assuming labels are folder names
X_test = load_images("testing_images")
y_test = np.array([int(folder) for folder in os.listdir("testing_images")])

# Training loop (multiple epochs and batches)
num_epochs = 10
batch_size = 32

for epoch in range(num_epochs):
    for i in range(0, len(X_train), batch_size):
        # Forward pass, backward pass, update weights (as in previous code)

# Evaluation
predictions = forward(X_test)
predicted_classes = np.argmax(predictions, axis=1)
accuracy = np.mean(predicted_classes == y_test)
print("Accuracy:", accuracy)

# Visualize results (example)
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(X_test[i].reshape(28, 28), cmap="gray")
    plt.title(f"Predicted: {predicted_classes[i]}, Actual: {y_test[i]}")
plt.show()

# Save the model
np.save("model_weights.npy", [W1, b1, W2, b2])
