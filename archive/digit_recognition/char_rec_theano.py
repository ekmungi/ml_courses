import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from utils import cross_entropy, one_hot_encoder, softmax, get_data, error_rate, relu

import theano.tensor as Th
import theano


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
    epochs=10

    W1_init = np.random.randn(D, M) / np.sqrt(D)
    b1_init = np.zeros(M)
    W2_init = np.random.randn(M, K) / np.sqrt(M)
    b2_init = np.zeros(K)

    thX = Th.matrix('X')
    thT = Th.matrix('T')
    W1 = theano.shared(W1_init, 'W1')
    b1 = theano.shared(b1_init, 'b1')
    W2 = theano.shared(W2_init, 'W2')
    b2 = theano.shared(b2_init, 'b2')

    # Forward model
    thZ = th.nnet.relu(thX.dot(W1) + b1)
    thY = Th.nnet.softmax(thZ.dot(W2) + b2)

    # Cost
    cost = -((thT*Th.log(thY)).sum() + reg*((W1*W1).sum() + (b1*b1).sum() + (W2*W2).sum() + (b2*b2).sum()))
    
    # Predictions
    prediction = Th.argmax(thY, axis=1)

    update_W1 = W1 - learning_rate*Th.grad(cost, W1)
    update_W2 = W2 - learning_rate*Th.grad(cost, W2)
    update_b1 = b1 - learning_rate*Th.grad(cost, b1)
    update_b2 = b2 - learning_rate*Th.grad(cost, b2)

    train = theano.function(inputs=[thX, thT], updates=[(W1, update_W1), 
                                                        (W2, update_W2), 
                                                        (b1, update_b1), 
                                                        (b2, update_b2)])
    
    get_prediction = theano.function(inputs=[thX, thT], outputs=[cost, prediction])

    costs = []
    for epoch in range(epochs):
        X_shuffled, T_shuffled = shuffle(X, T)
        for batch in range(n_batches):
            # Get the batch
            X_batch = X_shuffled[batch*batch_size:(batch+1)*batch_size,:]
            Y_batch = T_shuffled[batch*batch_size:(batch+1)*batch_size,:]
            
            train(X_batch, Y_batch)

            if batch % 10 == 0:
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
    






