import numpy as np
import matplotlib.pyplot as plt

# make data
N = 100

X = np.linspace(0, 6 * np.pi, N)
Y = np.sin(X)

plt.plot(X, Y)
plt.show()


def make_poly(X, deg):
    n = len(X)
    data = [np.ones(n)]
    for d in range(deg):
        data.append(X**(d + 1))
    # return transposition of column vector?
    return np.vstack(data).T


def fit(X, Y):
    return np.linalg.solve(X.T.dot(X), X.T.dot(Y))


def fit_and_display(X, Y, sample, deg):
    N = len(X)  # number of available points in X (should match Y)

    # build array of random ints no greater than N.  these will be the indices of the training data
    # this array will have elements = sample
    train_idx = np.random.choice(N, sample)

    # populate training X and Y datasets
    Xtrain = X[train_idx]
    Ytrain = Y[train_idx]

    plt.scatter(Xtrain, Ytrain)
    plt.show()

    # fit polynom to given degree based on training data
    Xtrain_poly = make_poly(Xtrain, deg)

    # w fit from standard linear regression solving format
    w = fit(Xtrain_poly, Ytrain)

    # display polynomial
    X_poly = make_poly(X, deg)
    Y_hat = X_poly.dot(w)

    plt.plot(X, Y, c='g')
    plt.plot(X, Y_hat, c='r')

    plt.scatter(Xtrain, Ytrain, c='b')

    plt.title('deg = %d' % deg)

    plt.show()


def get_mse(Y, Yhat):
    d = Y - Yhat
    return d.dot(d) / len(d)


def plot_train_vs_test_curves(X, Y, sample=20, max_deg=20):
    N = len(X)
    train_idx = np.random.choice(N, sample)
    Xtrain = X[train_idx]
    Ytrain = Y[train_idx]

    # use all non-training data as testing data
    test_idx = [idx for idx in range(N) if idx not in train_idx]
    Xtest = X[test_idx]
    Ytest = Y[test_idx]

    mse_trains = []
    mse_tests = []

    for deg in range(max_deg + 1):
        Xtrain_poly = make_poly(Xtrain, deg)
        w = fit(Xtrain_poly, Ytrain)
        Yhat_train = Xtrain_poly.dot(w)
        mse_train = get_mse(Ytrain, Yhat_train)

        Xtest_poly = make_poly(Xtest, deg)
        Yhat_test = Xtest_poly.dot(w)
        mse_test = get_mse(Ytest, Yhat_test)

        mse_trains.append(mse_train)
        mse_tests.append(mse_test)

    plt.plot(mse_trains, label="train mse")
    plt.plot(mse_tests, label="test mse")
    plt.legend()
    plt.show()

    plt.plot(mse_trains, label="train mse")
    plt.legend()
    plt.show()


for deg in (5, 6, 7, 8, 9):
    fit_and_display(X, Y, 10, deg)
plot_train_vs_test_curves(X, Y)
