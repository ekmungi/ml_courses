import numpy as np 
import pandas as pd 

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


def get_data(file_name='/media/avemuri/DEV/Data/deeplearning/mnist/train.csv', balanced=True):
    #df = pd.read_csv('D:/dev/data/fer2013.csv')
    df = pd.read_csv(file_name)
    data = df.as_matrix()
    Y = data[:, 0]
    X = data[:, 1:]

    X = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)

    return X, Y


def error_rate(targets, predictions):
    return np.mean(targets != predictions)


def main():
    get_data()


if __name__ == '__main__':
    main()