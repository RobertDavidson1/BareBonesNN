import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def fetch_data():
    X, y = fetch_openml("mnist_784", as_frame=False, return_X_y=True)

    # Normalise the pixel values
    # https://www.tensorflow.org/tutorials/quickstart/beginner
    X = X.astype(np.float32) / 255.0

    # Convert to one hot vectors
    # For each label, select the corresponding row from the identity matrix
    # If y = 3, then select row 3 = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y = np.eye(10)[y.astype(int)]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=10_000, random_state=1337, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=10_000,
        random_state=1337,
        stratify=y_train,
    )

    # Convert to column vectors
    X_train = X_train.reshape(-1, 784, 1)
    y_train = y_train.reshape(-1, 10, 1)

    X_val = X_val.reshape(-1, 784, 1)
    y_val = y_val.reshape(-1, 10, 1)

    X_test = X_test.reshape(-1, 784, 1)
    y_test = y_test.reshape(-1, 10, 1)

    return X_train, X_val, X_test, y_train, y_val, y_test
