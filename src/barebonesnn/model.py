import numpy as np


class MLP:
    def __init__(self, input_dims=784, hidden_dims=256, output_dims=10):
        # He Initialization
        # See https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/
        self.W1 = np.random.randn(hidden_dims, input_dims) * np.sqrt(2 / input_dims)
        self.b1 = np.random.randn(hidden_dims, 1) * np.sqrt(2 / input_dims)

        self.W2 = np.random.randn(output_dims, hidden_dims) * np.sqrt(2 / hidden_dims)
        self.b2 = np.random.randn(output_dims, 1) * np.sqrt(2 / hidden_dims)

    def forward(self, x):
        self.x = x
        self.a = self.W1 @ x + self.b1
        self.h = np.maximum(0, self.a)

        self.z = self.W2 @ self.h + self.b2

        # Prevent overflow by shifting maximum value to 0 so that e^z is stable:
        # Say z = [1001, 1002, 1003] -> z - max(z) = [0, 1, 2]
        # https://stackoverflow.com/questions/42599498/numerically-stable-softmax
        self.z = self.z - np.max(self.z)

        self.y_hat = np.exp(self.z) / (np.sum(np.exp(self.z)))

        return self.y_hat

    def back(self, y):
        # Thansk to Kevin Clark at Stanford, see:
        # https://web.stanford.edu/class/cs224n/readings/gradient-notes.pdf
        dz = self.y_hat - y

        dW2 = dz @ self.h.T
        db2 = dz

        dh = self.W2.T @ dz
        da = dh * (self.a > 0)

        dW1 = da @ self.x.T
        db1 = da

        return dW1, db1, dW2, db2

    def step(self, grads, lr):
        dw1, db1, dW2, db2 = grads

        self.W1 -= lr * dw1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    def save_weights(self, path):
        np.savez(path,W1=self.W1,b1=self.b1,W2=self.W2,b2=self.b2,)

    def load_weights(self, path):
        data = np.load(path)

        self.W1 = data["W1"]
        self.b1 = data["b1"]
        self.W2 = data["W2"]
        self.b2 = data["b2"]
