import numpy as np
from sklearn.utils import shuffle
import pandas as pd 
from process import get_data

# make predictions
def softmax(a):
    expA = np.exp(a)
    return expA / expA.sum(axis=1, keepdims=True)

def forward(X, W, b):
    return softmax(X.dot(W) + b)

def predict(P_Y_given_X):
    return np.argmax(P_Y_given_X, axis=1)

# calculate the accuracy
def classification_rate(Y, P):
    return np.mean(Y == P)

def cross_entropy(T, pY):
    return -np.mean(T*np.log(pY))

def y2indicator(y, K):
    N = len(y)
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind



def using_course_code():
    Xtrain, Ytrain, Xtest, Ytest = get_data()
    D = Xtrain.shape[1]
    K = len(set(Ytrain) | set(Ytest))

    W = np.random.randn(D, K)
    b = np.zeros(K)

    # convert to indicator
    Ytrain_ind = y2indicator(Ytrain, K)
    Ytest_ind = y2indicator(Ytest, K)

    train_costs = []
    test_costs = []
    learning_rate = 0.001
    for i in range(10000):
        pYtrain = forward(Xtrain, W, b)
        pYtest = forward(Xtest, W, b)

        ctrain = cross_entropy(Ytrain_ind, pYtrain)
        ctest = cross_entropy(Ytest_ind, pYtest)
        train_costs.append(ctrain)
        test_costs.append(ctest)

        # gradient descent
        W -= learning_rate*Xtrain.T.dot(pYtrain - Ytrain_ind)
        b -= learning_rate*(pYtrain - Ytrain_ind).sum(axis=0)
        if i % 1000 == 0:
            print(i, ctrain, ctest)

    print("Final train classification_rate:", classification_rate(Ytrain, predict(pYtrain)))
    print("Final test classification_rate:", classification_rate(Ytest, predict(pYtest)))


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

def forward_ant(X, W, b):
    return softmax(X.dot(W) + b)

def cross_entropy_ant(Y, Y_hat):
    return -np.mean(Y*np.log(Y_hat))

def predict_ant(Y_hat):
    return np.argmax(Y_hat, axis=1)

def dJ_dw(Y, Y_hat, X):
    return X.T.dot(Y_hat-Y)

def derivative_b(Y, Y_hat):
    return (Y_hat-Y).sum(axis=0)

def using_own_code():
    X_train, Y_train, X_test, Y_test = get_data_ant()
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

    D = X_train.shape[1] # number of features
    K = len(set(Y_train)) # number of classes

    T_train = one_hot_encoder(Y_train)
    T_test = one_hot_encoder(Y_test)

    W = np.random.randn(D, K)
    b = np.random.randn(K)

    learning_rate = 1e-3
    costs_train = []
    costs_test = []
    for epoch in range(10000):
        Y_hat_train = forward_ant(X_train, W, b)
        Y_hat_test = forward_ant(X_test, W, b)
        
        ctrain = cross_entropy_ant(T_train, Y_hat_train)
        ctest = cross_entropy_ant(T_test, Y_hat_test)
        
        costs_train.append(ctrain)
        costs_test.append(ctest)
        
        W -= learning_rate * dJ_dw(T_train, Y_hat_train, X_train)
        b -= learning_rate * derivative_b(T_train, Y_hat_train)
        
        if epoch % 1000 == 0:
            print(epoch, ctrain, ctest)


    print("Final train classification_rate:", classification_rate_ant(Y_train, predict_ant(Y_hat_train)))
    print("Final test classification_rate:", classification_rate_ant(Y_test, predict_ant(Y_hat_test)))

using_own_code()

