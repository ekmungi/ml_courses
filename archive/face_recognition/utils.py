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


def get_data(file_name, balanced=True):
    #df = pd.read_csv('D:/dev/data/fer2013.csv')
    df = pd.read_csv(file_name)
    Y = df['label'].as_matrix()
    X_str = df['pixels'].as_matrix()
    X = []
    for row in X_str:
        X.append(np.fromstring(row, dtype=int, sep=' '))
        
    X = np.array(X) / 255.0
    X = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)

    usage = df['Usage'].as_matrix()
    X_train, Y_train = X[usage=='Training'], Y[usage=='Training']
    X_test, Y_test = X[(usage=='PrivateTest') | (usage=='PublicTest')], Y[(usage=='PrivateTest') | (usage=='PublicTest')]


    if balanced:
        # balance the 1 class
        X0, Y0 = X_train[Y_train!=1, :], Y_train[Y_train!=1]
        X1 = X_train[Y_train==1, :]
        X1 = np.repeat(X1, 9, axis=0)
        X_train = np.vstack([X0, X1])
        Y_train = np.concatenate((Y0, [1]*len(X1)))



    #print(X.shape, Y.shape)
    return X_train, Y_train, X_test, Y_test


def error_rate(targets, predictions):
    return np.mean(targets != predictions)