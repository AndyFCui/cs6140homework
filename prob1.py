import numpy as np
import matplotlib.pyplot as plt

from prob1_model import Predictor, train_model, evaluate_model


def visualize_dataset(X, y):
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], 0.1, [
        "black" if e < 0 else "red"
    for e in y])
    plt.show()
    plt.close(fig)


def check_val_a(p):
    px = p[0]
    py = p[1]
    if -4 <= px <= -1 and 0 <= py <= 3:
        return 1.
    if -2 <= px <= 1 and -4 <= py <= -1:
        return 1.
    if 2 <= px <= 5 and -2 <= py <= 1:
        return 1.
    return -1.


def check_val_b(p):
    px = p[0]
    py = p[1]
    if -4 <= px <= -3 and 2 <= py <= 3:
        return 1.
    if -1 <= px <= 0 and -3 <= py <= -2:
        return 1.
    if 2 <= px <= 3 and -1 <= py <= 0:
        return 1.
    return -1.


def generate_dataset(n, check_val_fcn):
    X = np.random.uniform((-6, -4), (6, 4), size=(n, 2))
    y = np.array([check_val_fcn(X[i, :]) for i in range(n)])
    return X, y


def main():
    X_val, y_val = generate_dataset(250000, check_val_b)
    # visualize_dataset(X, y)

    for n in [250, 1000, 10000]:
        X, y = generate_dataset(n, check_val_b)
        for h1 in [1, 4, 12]:
            for h2 in [0, 3]:
                param = [h1, h2]
                model = Predictor(*param)
                model_name = f"n={n}, p=" + \
                    ", ".join(map(lambda x: str(x), param))
                train_model(model_name, model, X, y, X_val, y_val)


if __name__ == "__main__":
    main()
