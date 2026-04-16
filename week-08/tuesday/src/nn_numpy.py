import numpy as np

class SimpleNN:
    def __init__(self, input_size, hidden_sizes, output_size=1, learning_rate=0.01):
        """
        Initialize a 3-layer Neural Network (2 hidden layers).
        """
        self.lr = learning_rate
        
        # Xavier Initialization for weights
        self.W1 = np.random.randn(input_size, hidden_sizes[0]) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden_sizes[0]))
        
        self.W2 = np.random.randn(hidden_sizes[0], hidden_sizes[1]) * np.sqrt(2. / hidden_sizes[0])
        self.b2 = np.zeros((1, hidden_sizes[1]))
        
        self.W3 = np.random.randn(hidden_sizes[1], output_size) * np.sqrt(2. / hidden_sizes[1])
        self.b3 = np.zeros((1, output_size))

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def sigmoid_derivative(self, a):
        return a * (1 - a)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.relu(self.z2)
        
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = self.sigmoid(self.z3)
        return self.a3

    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        # Binary Cross Entropy
        loss = - (1/m) * np.sum(y_true * np.log(y_pred + 1e-15) + (1 - y_true) * np.log(1 - y_pred + 1e-15))
        return loss

    def backward(self, X, y_true, y_pred):
        m = y_true.shape[0]
        
        # dL/da3 * da3/dz3
        dz3 = y_pred - y_true
        dW3 = (1/m) * np.dot(self.a2.T, dz3)
        db3 = (1/m) * np.sum(dz3, axis=0, keepdims=True)
        
        # Propagate to second hidden layer
        da2 = np.dot(dz3, self.W3.T)
        dz2 = da2 * self.relu_derivative(self.z2)
        dW2 = (1/m) * np.dot(self.a1.T, dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        # Propagate to first hidden layer
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.relu_derivative(self.z1)
        dW1 = (1/m) * np.dot(X.T, dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        # Update weights and biases
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train(self, X, y, epochs=1000):
        history = []
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.compute_loss(y, y_pred)
            self.backward(X, y, y_pred)
            history.append(loss)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        return history
