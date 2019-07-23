import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from utils import cross_entropy, one_hot_encoder, softmax, get_data, error_rate, relu

import theano.tensor as th
import theano


# def plot_characters():
#     X, Y, _, _ = get_data()
#     while True:
#         f, axarr = plt.subplots(3, 4)
#         for i in range(10):
#             x_select, y_select = X[Y==i], Y[Y==i]
#             N_select = len(y_select)
#             j = np.random.choice(N_select)
#             axarr[i].imshow(np.reshape(x_select[j], (28,28)), cmap='gray')
#             axarr[i].set_title(label_map[y_select[j]])
#         plt.show()
#         prompt = input('Quit? Enter Y:\n')
#         if (prompt == 'Y') | (prompt == 'y'):
#             break


def main():
    #file_loc = '/media/avemuri/DEV/Data/deeplearning/mnist/train.csv'
    file_loc = 'D:/dev/data/mnist/train.csv'
    X_train, Y_train, X_test, Y_test = get_data(file_name=file_loc, split_train_test=True)
    
    pca = PCA(n_components=400)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    T_train = one_hot_encoder(Y_train)
    T_test = one_hot_encoder(Y_test)

    D = X_train.shape[1] # number of features
    K = len(set(Y_train)) # number of classes
    decay_rate = 0.999
    eps = 1e-10
    epochs = 10
    n_batches = 10
    batch_size = X_train.shape[0]//n_batches
    print_time = n_batches
    M = 300
    reg = 0.00001
    learning_rate = 0.0004
    

    W1_init = np.random.randn(D, M) / np.sqrt(D)
    b1_init = np.zeros(M)
    W2_init = np.random.randn(M, K) / np.sqrt(M)
    b2_init = np.zeros(K)
    

    thX = th.matrix('X')
    thT = th.matrix('Y')
    W1 = theano.shared(W1_init, 'W1')
    b1 = theano.shared(b1_init, 'b1')
    W2 = theano.shared(W2_init, 'W2')
    b2 = theano.shared(b2_init, 'b2')
    cache_W1 = theano.shared(1, 'cache_w1')
    cache_b1 = theano.shared(1, 'cache_b1')
    cache_W2 = theano.shared(1, 'cache_w2')
    cache_b2 = theano.shared(1, 'cache_b2')


    # forward model
    thZ = th.nnet.relu(thX.dot(W1) + b1)
    #thZ[thZ < 0] = 0
    # Z = np.tanh(X.dot(self.W1) + self.b1)
    thY = th.nnet.softmax(thZ.dot(W2) + b2)

    # Cost
    cost = -((thT*th.log(thY)).sum() + reg*((W1*W1).sum() + (b1*b1).sum() + (W2*W2).sum() + (b2*b2).sum()))

    # Prediction
    prediction = th.argmax(thY, axis=1)

    # Updates
    dJ_dW1 = th.grad(cost, W1)
    dJ_db1 = th.grad(cost, b1)
    dJ_dW2 = th.grad(cost, W2)
    dJ_db2 = th.grad(cost, b2)

    cache_W1 = decay_rate*cache_W1 + (1-decay_rate)*dJ_dW1*dJ_dW1
    cache_b1 = decay_rate*cache_b1 + (1-decay_rate)*dJ_db1*dJ_db1
    cache_W2 = decay_rate*cache_W2 + (1-decay_rate)*dJ_dW2*dJ_dW2
    cache_b2 = decay_rate*cache_b2 + (1-decay_rate)*dJ_db2*dJ_db2
    
    update_W1 = W1 - learning_rate*dJ_dW1/(np.sqrt(cache_W1)+eps)
    update_b1 = b1 - learning_rate*dJ_db1/(np.sqrt(cache_b1)+eps)
    update_W2 = W2 - learning_rate*dJ_dW2/(np.sqrt(cache_W2)+eps)
    update_b2 = b2 - learning_rate*dJ_db2/(np.sqrt(cache_b2)+eps)

    train = theano.function(inputs=[thX, thT], updates=[(W1, update_W1), (b1, update_b1), (W2, update_W2), (b2, update_b2)])#

    get_prediction = theano.function(inputs=[thX, thT], outputs=[cost, prediction])
    
    costs = []
    for epoch in range(epochs):
        X_shuffled, T_shuffled = shuffle(X_train, T_train)
        for batch in range(n_batches):
            # Get the batch
            X_batch = X_shuffled[batch*batch_size:(batch+1)*batch_size,:]
            Y_batch = T_shuffled[batch*batch_size:(batch+1)*batch_size,:]

            train(X_batch, Y_batch)
            
            if batch % print_time == 0:
                c, pred = get_prediction(X_test, T_test)
                err = error_rate(Y_test, pred)
                print("epoch [%d], batch [%d] : cost=[%.3f], error=[%.3f]" %(epoch, batch, c, err))
                costs.append(c)

    plt.plot(costs)
    plt.title('Validation cost')
    plt.show()


if __name__ == '__main__':
    main()
    






