import numpy as np
from sklearn.utils import shuffle
import pandas as pd 
import matplotlib.pyplot as plt 


####################################################################


def one_hot_encoder(data):
    # One-hot encoding
    unique_time = np.unique(data)
    #print(unique_time)
    one_hot = np.zeros((data.shape[0], len(unique_time)))
    for t in unique_time:
        one_hot[:,int(t)] = np.where(data==t, 1, 0)
        
    return one_hot


def get_data_ant():
    df = pd.read_csv('ecommerce_data.csv')
    data = df.as_matrix()
    
    X = data[:,:-1]
    Y = data[:,-1].astype(np.int32)
    
    X, Y = shuffle(X, Y, random_state=42)
    
    N, D = X.shape
    
    
    # One-hot encoding
    X2 = np.zeros((N,D+3))
    X2[:,:D-1] = X[:,:D-1]
    X2[:,D-1:D+3] = one_hot_encoder(X[:,D-1])
    X = X2

    X_train = X[:-100,:]
    Y_train = Y[:-100]
    X_test = X[-100:,:]
    Y_test = Y[-100:]
    
    
    # normalize the data
    for i in (1,2):
        X_train[:,i] = (X_train[:,i] - X_train[:,i].mean())/X_train[:,i].std()
        X_test[:,i] = (X_test[:,i] - X_test[:,i].mean())/X_test[:,i].std()
        
    return X_train, Y_train, X_test, Y_test

def softmax_ant(a):
    exp_a = np.exp(a)
    return exp_a/exp_a.sum(axis=1, keepdims=True)

def classification_rate_ant(Y, Y_hat_class):
    return 100*np.mean(Y==Y_hat_class)

def forward_ant(X, W1, b1, W2, b2):
    a1 = X.dot(W1) + b1
    z = sigmoid(a1)
    
    a2 = z.dot(W2) + b2
    Y_hat = softmax_ant(a2)
    
    return Y_hat, z

def cross_entropy_ant(Y, Y_hat):
    return -np.mean(Y*np.log(Y_hat))

def sigmoid(a):
    return 1/(1+np.exp(-a))

def predict_ant(Y_hat):
    return np.argmax(Y_hat, axis=1)


# *************** Derivatives *************** #
def dJ_dw2(T, Y, Z):
    return Z.T.dot(T-Y)

def dJ_dw1(T, Y, Z, X, W2):
    return X.T.dot((T-Y).dot(W2.T) * Z * (1-Z))

def derivative_b2(T, Y):
    return (T - Y).sum(axis=0)

def derivative_b1(T, Y, W2, Z):
    return ((T-Y).dot(W2.T) * Z * (1-Z)).sum(axis=0)
# ******************************************* #

def using_own_code():
    X_train, Y_train, X_test, Y_test = get_data_ant()
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

    D = X_train.shape[1] # number of features
    M = 5                # number of nodes in the hidden layer
    K = len(set(Y_train)) # number of classes

    T_train = one_hot_encoder(Y_train)
    T_test = one_hot_encoder(Y_test)

    W1 = np.random.randn(D, M)
    b1 = np.random.randn(M)

    W2 = np.random.randn(M, K)
    b2 = np.random.randn(K)

    learning_rate = 1e-3
    costs_train = []
    costs_test = []
    for epoch in range(10000):
        Y_hat_train, hidden = forward_ant(X_train, W1, b1, W2, b2)
        Y_hat_test, temp_ignore = forward_ant(X_test, W1, b1, W2, b2)

        W2 += learning_rate * dJ_dw2(T_train, Y_hat_train, hidden)
        b2 += learning_rate * derivative_b2(T_train, Y_hat_train)
        W1 += learning_rate * dJ_dw1(T_train, Y_hat_train, hidden, X_train, W2)
        b1 += learning_rate * derivative_b1(T_train, Y_hat_train, W2, hidden)        
        
        ctrain = cross_entropy_ant(T_train, Y_hat_train)
        ctest = cross_entropy_ant(T_test, Y_hat_test)
        
        costs_train.append(ctrain)
        costs_test.append(ctest)
        
        if epoch % 1000 == 0:
            print(epoch, ctrain, ctest)


    legend1, = plt.plot(costs_train, label='train cost')
    legend2, = plt.plot(costs_test, label='test cost')
    plt.legend([legend1, legend2])
    plt.show()


    print("Final train classification_rate:", classification_rate_ant(Y_train, predict_ant(Y_hat_train)))
    print("Final test classification_rate:", classification_rate_ant(Y_test, predict_ant(Y_hat_test)))

using_own_code()




