import numpy as np
from data import fetch_data
from model import MLP
from utils import accuracy

np.random.seed(1337)

LEARNING_RATE = 1e-3
EPOCHS = 10

mlp = MLP()
X_train, X_val, X_test, y_train, y_val, y_test = fetch_data()

for epoch in range(EPOCHS):
    indices = np.random.permutation(len(X_train))

    total_loss = 0.0
    correct = 0

    for i in indices:
        x = X_train[i]
        y = y_train[i]

        y_hat = mlp.forward(x)

        # Cross-entropy loss
        total_loss += -np.sum(y * np.log(y_hat + 1e-12))

        if np.argmax(y_hat) == np.argmax(y):
            correct += 1

        gradients = mlp.back(y)
        mlp.step(gradients, LEARNING_RATE)

    train_loss = total_loss / len(X_train)
    train_accuracy = correct / len(X_train)

    val_accuracy = accuracy(mlp, X_val, y_val)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"- loss: {train_loss:.4f} "
        f"- train: {train_accuracy:.4f} "
        f"- val: {val_accuracy:.4f}"
    )

# Only touch the test set after training is finished
test_accuracy = accuracy(mlp, X_test, y_test)

print(f"\nFinal test accuracy: {test_accuracy:.4f}")
mlp.save_weights("mlp_weights.npz")
