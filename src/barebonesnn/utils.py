import numpy as np


def accuracy(model, X, y):
    correct = 0

    for x, target in zip(X, y):
        y_hat = model.forward(x)

        prediction = np.argmax(y_hat)
        actual = np.argmax(target)

        if prediction == actual:
            correct += 1

    return correct / len(X)
