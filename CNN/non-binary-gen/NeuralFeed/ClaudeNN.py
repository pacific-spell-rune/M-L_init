import numpy as np 
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

# Load data from folders 
def load_data(test_size=0.2):
    
    # List folders in data directory
    data_dir = "/data" 
    folders = os.listdir(data_dir)
    
    # Load images and labels into arrays
    X = [] # Images 
    y = [] # Labels
    for folder in folders:
        folder_path = os.path.join(data_dir, folder)
        images = os.listdir(folder_path)
        
        for image in images:
            img = os.path.join(folder_path, image)
            img = plt.imread(img) # Read image
            img = np.resize(img, (28,28)) # Resize
            
            X.append(img)
            y.append(folder) # Folder name as label
            
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Split test and train data
    test_size = int(test_size*X.shape[0])
    
    X_test = X[:test_size]
    X_train = X[test_size:]
    
    y_test = y[:test_size] 
    y_train = y[test_size:]
    
    return X_train, X_test, y_train, y_test
        
# Neural Network from scratch
class NeuralNetwork():
    
    def __init__(self, input_nodes, hidden_nodes, num_classes):
        
        # Parameters        
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes 
        self.num_classes = num_classes  
                
        #Weights        
        self.W1 = np.random.randn(self.input_nodes, self.hidden_nodes) 
        self.W2 = np.random.randn(self.hidden_nodes, self.num_classes)
        
    def relu(self, X):
        return np.maximum(0, X)
        
    def softmax(self, X):
        exp = np.exp(X - np.max(X))
        return exp/exp.sum(axis=1, keepdims=True)
    
    # Forward pass       
    def forward_pass(self, X):
        self.layer1 = self.relu(np.dot(X, self.W1))  
        layer2 = np.dot(self.layer1, self.W2) 
        layer2_act = self.softmax(layer2)  
        return layer2_act
       
    def loss(self, Ypred, Ytrue): 
        samples = len(Ypred)        
        Ypred_good = np.array([Ypred[i][Ytrue[i]] for i in range(samples)])
    
        loss = -np.sum(np.log(Ypred_good)) / samples
        return loss

    # Backpropagation      
    def backward_pass(self, X, Ypred, Ytrue):
                
        # Calculates gradients        
        self.dW2 = np.dot(self.layer1.T, (Ypred - Ytrue)) / X.shape[0]
        self.dW1 = np.dot(X.T, np.dot(Ypred - Ytrue, self.W2.T)*self.relu(np.dot(X, self.W1))* (self.relu(np.dot(X, self.W1)) > 0)) / X.shape[0]

    # Update weights            
    def update(self, learning_rate):        
        self.W1 = self.W1 - self.dW1 * learning_rate
        self.W2 = self.W2 - self.dW2 * learning_rate
        
    # Train         
    def train(self, X, Ytrue, epochs, lr):
    
        losses = []
        for epoch in range(epochs):
            # Forward pass  
            Ypred = self.forward_pass(X)
            
            # Loss
            loss = self.loss(Ypred, Ytrue)  
            losses.append(loss)
        
            # Backprop
            self.backward_pass(X, Ypred, Ytrue)  
        
            # Update weights        
            self.update(lr)
        
            if (epoch+1) % 10 == 0:
                print('Epoch %d, loss=%f' %((epoch+1), loss))
        
        plt.plot(losses) 
        plt.xlabel('Epochs')
        plt.ylabel('Loss')  
        plt.show()        
        
    # Prediction         
    def predict(self, X): 
        Ypred = self.forward_pass(X)
        class_preds = np.argmax(Ypred, axis=1)
        return class_preds
        
# Train model
def train(X, Y, model):
    model.train(X, Y, epochs=100, lr=0.01) 
    print("Model trained")
    
# Predict new data   
def predict(X, model):
    preds = model.predict(X)
    return preds

if __name__ == "__main__":

    # Load data
    X_train, X_test, Y_train, Y_test = load_data() 
    
    # Create model object
    input_nodes = 784 # Image pixels 
    hidden_nodes = 100
    num_classes = len(np.unique(Y_train))
    model = NeuralNetwork(input_nodes, hidden_nodes, num_classes)
    
    # Train   
    train(X_train, Y_train, model)
   
    # Predict test data
    predictions = predict(X_test, model)
    
    # Evaluate predictions          
    accuracy = accuracy_score(Y_test, predictions)
    precision = precision_score(Y_test, predictions, average='weighted') 
    recall = recall_score(Y_test, predictions, average='weighted')
    f1 = f1_score(Y_test, predictions, average='weighted')   
    
    print("Accuracy: %.2f%%" % (accuracy*100.0))
    print("Precision: %.2f%%" % (precision*100.0))
    print("Recall: %.2f%%" % (recall*100.0))
    print("F1 Score: %.2f%%" % (f1*100.0))
    
    # Save model
    model_file = "image_model.pkl"  
    pickle.dump(model, open(model_file, "wb"))