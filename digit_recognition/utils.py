import numpy as np 
import pandas as pd 
from sklearn.utils import shuffle


def one_hot_encoder(data):
    # One-hot encoding
    unique_time = np.unique(data)
    #print(unique_time)
    one_hot = np.zeros((data.shape[0], len(unique_time)))
    for t in unique_time:
        one_hot[:,int(t)] = np.where(data==t, 1, 0)
        
    return one_hot

def softmax(a):
    exp_a = np.exp(a)
    return exp_a/exp_a.sum(axis=1, keepdims=True)


def sigmoid(a):
    return 1/(1+np.exp(-a))


def cross_entropy(Y, Y_hat):
    return -np.mean(Y*np.log(Y_hat))


def get_data(file_name, split_train_test=False):
    df = pd.read_csv(file_name)
    data = df.as_matrix()
    Y = data[:, 0]
    X = data[:, 1:]

    X = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)


    if split_train_test:
        X, Y = shuffle(X, Y)
        X_train, Y_train = X[:-1000], Y[:-1000]
        X_test, Y_test = X[-1000:], Y[-1000:]
        return X_train, Y_train, X_test, Y_test

    return X, Y


def error_rate(targets, predictions):
    return np.mean(targets != predictions)


def main():
    get_data()


if __name__ == '__main__':
    main()