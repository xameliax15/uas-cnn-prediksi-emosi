"""
Artificial Neural Network with Backpropagation
Implementasi ANN dari scratch untuk klasifikasi emosi
"""

import numpy as np
import pickle
import json


class ANNClassifier:
    """
    Custom Artificial Neural Network Classifier with Backpropagation
    """
    
    def __init__(self, input_size, hidden_layers=[64, 32], output_size=5, 
                 learning_rate=0.01, activation='sigmoid'):
        """
        Initialize ANN
        
        Args:
            input_size: Jumlah fitur input
            hidden_layers: List jumlah neuron per hidden layer
            output_size: Jumlah kelas output
            learning_rate: Learning rate untuk backpropagation
            activation: Fungsi aktivasi ('sigmoid', 'relu', 'tanh')
        """
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.activation = activation
        
        # Inisialisasi weights dan biases
        self.weights = []
        self.biases = []
        
        # Layer architecture
        layer_sizes = [input_size] + hidden_layers + [output_size]
        
        # Xavier/He initialization untuk weights
        for i in range(len(layer_sizes) - 1):
            if activation == 'relu':
                # He initialization
                w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            else:
                # Xavier initialization
                w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(1.0 / layer_sizes[i])
            
            b = np.zeros((1, layer_sizes[i+1]))
            
            self.weights.append(w)
            self.biases.append(b)
        
        # Training history
        self.history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
    
    def _sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _sigmoid_derivative(self, x):
        """Derivative of sigmoid"""
        s = self._sigmoid(x)
        return s * (1 - s)
    
    def _relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def _relu_derivative(self, x):
        """Derivative of ReLU"""
        return (x > 0).astype(float)
    
    def _tanh(self, x):
        """Tanh activation function"""
        return np.tanh(x)
    
    def _tanh_derivative(self, x):
        """Derivative of tanh"""
        return 1 - np.tanh(x) ** 2
    
    def _softmax(self, x):
        """Softmax activation for output layer"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def _activate(self, x, derivative=False):
        """Apply activation function"""
        if derivative:
            if self.activation == 'sigmoid':
                return self._sigmoid_derivative(x)
            elif self.activation == 'relu':
                return self._relu_derivative(x)
            elif self.activation == 'tanh':
                return self._tanh_derivative(x)
        else:
            if self.activation == 'sigmoid':
                return self._sigmoid(x)
            elif self.activation == 'relu':
                return self._relu(x)
            elif self.activation == 'tanh':
                return self._tanh(x)
    
    def forward_propagation(self, X):
        """
        Forward propagation
        
        Returns:
            activations: List of activations for each layer
            z_values: List of z values (before activation) for each layer
        """
        activations = [X]
        z_values = []
        
        # Hidden layers
        for i in range(len(self.weights) - 1):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            z_values.append(z)
            a = self._activate(z)
            activations.append(a)
        
        # Output layer (softmax)
        z = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        z_values.append(z)
        a = self._softmax(z)
        activations.append(a)
        
        return activations, z_values
    
    def backward_propagation(self, X, y, activations, z_values):
        """
        Backward propagation (Backpropagation Algorithm)
        
        Returns:
            weight_gradients: Gradients for weights
            bias_gradients: Gradients for biases
        """
        m = X.shape[0]
        
        weight_gradients = [None] * len(self.weights)
        bias_gradients = [None] * len(self.biases)
        
        # Output layer error (Cross-entropy with softmax)
        delta = activations[-1] - y
        
        # Backpropagate through all layers
        for i in range(len(self.weights) - 1, -1, -1):
            # Compute gradients
            weight_gradients[i] = np.dot(activations[i].T, delta) / m
            bias_gradients[i] = np.sum(delta, axis=0, keepdims=True) / m
            
            # Propagate error to previous layer
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self._activate(z_values[i-1], derivative=True)
        
        return weight_gradients, bias_gradients
    
    def compute_loss(self, y_true, y_pred):
        """
        Compute cross-entropy loss
        """
        m = y_true.shape[0]
        # Add small epsilon to prevent log(0)
        log_likelihood = -np.log(y_pred[range(m), y_true.argmax(axis=1)] + 1e-10)
        loss = np.sum(log_likelihood) / m
        return loss
    
    def train_batch(self, X, y):
        """
        Train on a single batch
        """
        # Forward propagation
        activations, z_values = self.forward_propagation(X)
        
        # Backward propagation
        weight_gradients, bias_gradients = self.backward_propagation(X, y, activations, z_values)
        
        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * weight_gradients[i]
            self.biases[i] -= self.learning_rate * bias_gradients[i]
        
        # Compute loss
        loss = self.compute_loss(y, activations[-1])
        
        return loss
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32, verbose=True):
        """
        Train the neural network
        
        Args:
            X_train: Training features
            y_train: Training labels (one-hot encoded)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size for mini-batch gradient descent
            verbose: Print training progress
        """
        n_samples = X_train.shape[0]
        
        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            # Mini-batch training
            epoch_loss = 0
            n_batches = 0
            
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                batch_loss = self.train_batch(X_batch, y_batch)
                epoch_loss += batch_loss
                n_batches += 1
            
            # Average loss for epoch
            avg_loss = epoch_loss / n_batches
            
            # Training accuracy
            train_pred = self.predict(X_train)
            train_acc = np.mean(train_pred == y_train.argmax(axis=1))
            
            self.history['loss'].append(avg_loss)
            self.history['accuracy'].append(train_acc)
            
            # Validation metrics
            if X_val is not None and y_val is not None:
                val_activations, _ = self.forward_propagation(X_val)
                val_loss = self.compute_loss(y_val, val_activations[-1])
                val_pred = self.predict(X_val)
                val_acc = np.mean(val_pred == y_val.argmax(axis=1))
                
                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_acc)
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Acc: {train_acc:.4f}")
        
        return self.history
    
    def predict(self, X):
        """
        Predict class labels
        
        Returns:
            Predicted class indices
        """
        activations, _ = self.forward_propagation(X)
        return np.argmax(activations[-1], axis=1)
    
    def predict_proba(self, X):
        """
        Predict class probabilities
        
        Returns:
            Probability distribution for each class
        """
        activations, _ = self.forward_propagation(X)
        return activations[-1]
    
    def save_model(self, filepath):
        """
        Save model to file
        """
        model_data = {
            'weights': self.weights,
            'biases': self.biases,
            'input_size': self.input_size,
            'hidden_layers': self.hidden_layers,
            'output_size': self.output_size,
            'learning_rate': self.learning_rate,
            'activation': self.activation,
            'history': self.history
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath):
        """
        Load model from file
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.weights = model_data['weights']
        self.biases = model_data['biases']
        self.input_size = model_data['input_size']
        self.hidden_layers = model_data['hidden_layers']
        self.output_size = model_data['output_size']
        self.learning_rate = model_data['learning_rate']
        self.activation = model_data['activation']
        self.history = model_data['history']
    
    def get_config(self):
        """
        Get model configuration
        """
        return {
            'input_size': self.input_size,
            'hidden_layers': self.hidden_layers,
            'output_size': self.output_size,
            'learning_rate': self.learning_rate,
            'activation': self.activation,
            'total_parameters': sum(w.size for w in self.weights) + sum(b.size for b in self.biases)
        }
