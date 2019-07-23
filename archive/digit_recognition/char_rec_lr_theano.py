import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from utils import cross_entropy, one_hot_encoder, softmax, get_data, error_rate

import theano.tensor as Th
import theano


def relu(a):
    return a * (a > 0)


def main():
    #file_loc = '/media/avemuri/DEV/Data/deeplearning/mnist/train.csv'
    file_loc = 'D:/dev/data/mnist/train.csv'
    X_train, Y_train, X_test, Y_test = get_data(file_name=file_loc, split_train_test=True)
    
    pca = PCA(n_components=400)
    pca.fit(X_train)
    X = pca.transform(X_train)
    Y = Y_train
    T = one_hot_encoder(Y)

    X_test = pca.transform(X_test)
    T_test = one_hot_encoder(Y_test)
    
    #######################################################

    D = X.shape[1] # number of features
    K = len(set(Y)) # number of classes
    M = 300
    reg = 0.00001
    batch_size = 500
    n_batches = X.shape[0]//batch_size
    learning_rate = 0.0004
    epochs=1000
    print_time = epochs//10

    W_init = np.random.randn(D, K) / np.sqrt(D)
    b_init = np.zeros(K)

    thX = Th.matrix('X')
    thT = Th.matrix('T')
    W = theano.shared(W_init, 'W')
    b = theano.shared(b_init, 'b')

    # Forward model
    thY = Th.nnet.softmax(thX.dot(W) + b)

    # Cost
    cost = -((thT*Th.log(thY)).sum() + reg*((W*W).sum() + (b*b).sum()))
    
    # Predictions
    prediction = Th.argmax(thY, axis=1)

    update_W = W - learning_rate*Th.grad(cost, W)
    update_b = b - learning_rate*Th.grad(cost, b)

    train = theano.function(inputs=[thX, thT], updates=[(W, update_W), 
                                                        (b, update_b)])
    
    get_prediction = theano.function(inputs=[thX, thT], outputs=[cost, prediction])

    costs = []
    for epoch in range(epochs):
        X_shuffled, T_shuffled = shuffle(X, T)
        for batch in range(n_batches):
            # Get the batch
            X_batch = X_shuffled[batch*batch_size:(batch+1)*batch_size,:]
            Y_batch = T_shuffled[batch*batch_size:(batch+1)*batch_size,:]
            
            train(X_batch, Y_batch)

            if batch % print_time == 0:
                test_cost, prediction = get_prediction(X_test, T_test)
                err = error_rate(Y_test, prediction)
                print("epoch [%d], batch [%d] : cost=[%.3f], error=[%.3f]" %(epoch, batch, test_cost, err))
                costs.append(test_cost)

    plt.plot(costs)
    plt.title('Validation cost')
    plt.show()

    #######################################################


if __name__ == '__main__':
    main()
    






