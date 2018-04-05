import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from utils import cross_entropy, one_hot_encoder, softmax, get_data, error_rate

import theano.tensor as Th
import theano


def relu(a):
    return a * (a > 0)


class NeuralNet(object):
    def __init__(self, M):
        self.M = M # number of hidden layer units
        pass

    def fit(self, X, Y, learning_rate=1e-8, reg=1e-12, epochs=10000, n_batches=10, show_fig=False):
        
        D = X.shape[1] # number of features
        K = len(set(Y)) # number of classes

        X, Y = shuffle(X, Y)
        X_valid, Y_valid = X[-1000:], Y[-1000:]
        T_valid = one_hot_encoder(Y_valid)
        X, Y = X[:-1000], Y[:-1000]

        batch_size = X.shape[0]//n_batches

        T = one_hot_encoder(Y)

        self.W1 = np.random.randn(D, self.M) / np.sqrt(D)
        self.b1 = np.zeros(self.M)
        self.W2 = np.random.randn(self.M, K) / np.sqrt(self.M)
        self.b2 = np.zeros(K)

        costs = []
        best_validation_error = 1
        for epoch in range(epochs):
            X_shuffled, T_shuffled = shuffle(X, T)
            for ibatch in range(n_batches):
                # Get the batch
                X_batch = X_shuffled[ibatch*batch_size:(ibatch+1)*batch_size,:]
                Y_batch = T_shuffled[ibatch*batch_size:(ibatch+1)*batch_size,:]


                Y_hat, Z = self.forward(X_batch)

                # Weight updates ----------------------
                Y_hat_T = Y_hat-Y_batch
                self.W2 -= learning_rate * (Z.T.dot(Y_hat_T) + reg*self.W2)
                self.b2 -= learning_rate * (Y_hat_T.sum() + reg*self.b2)

                val = Y_hat_T.dot(self.W2.T) * (1-Z*Z) #tanh
                self.W1 -= learning_rate * (X_batch.T.dot(val) + reg*self.W1)
                self.b1 -= learning_rate * (val.sum() + reg*self.b1)
                # -------------------------------------

                Y_hat_valid, _ = self.forward(X_valid)
                c = cross_entropy(T_valid, Y_hat_valid)
                costs.append(c)
                
                if ibatch % (n_batches) == 0:
                    e = error_rate(Y_valid, np.argmax(Y_hat_valid, axis=1))
                    print("epoch:", epoch, " cost:", c, " error:", e)

        if show_fig:
            plt.plot(costs)
            plt.title('Validation cost')
            plt.show()

        print("Final train classification_rate:", self.score(X, Y))

    def predict(self, X):
        Y_hat, _ = self.forward(X)
        return np.argmax(Y_hat, axis=1)

    def score(self, X, Y):
        prediction = self.predict(X)
        return np.round(1 - error_rate(Y, prediction), 4)

    def forward(self, X):
        
        Z = np.tanh(X.dot(self.W1) + self.b1)
        Y_hat = softmax(Z.dot(self.W2) + self.b2)
        return Y_hat, Z        

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
    #nn_classify = NeuralNet(200)
    
    pca = PCA(n_components=400)
    pca.fit(X_train)
    X = pca.transform(X_train)
    Y = Y_train
    T = one_hot_encoder(Y)

    #nn_classify.fit(X_train_compressed, Y_train, epochs=100, learning_rate=0.0004, reg=0.01, n_batches=50, show_fig=True)

    #######################################################

    D = X.shape[1] # number of features
    K = len(set(Y)) # number of classes
    M = 300
    reg = 0.00001
    batch_size = X.shape[0]//n_batches

    W1_init = np.random.randn(D, M) / np.sqrt(D)
    b1_init = np.zeros(M)
    W2_init = np.random.randn(M, K) / np.sqrt(M)
    b2_init = np.zeros(K)

    thX = Th.matrix('X')
    thT = Th.matrix('T')
    W1 = theano.shared(W1_init, 'W1')
    b1 = theano.shared(W1_init, 'b1')
    W2 = theano.shared(W1_init, 'W2')
    b2 = theano.shared(W1_init, 'b2')

    thZ = relu()
    

    costs = []
    for epoch in range(epochs):
        X_shuffled, T_shuffled = shuffle(X, T)
        for batch in range(n_batches):
            # Get the batch
            X_batch = X_shuffled[batch*batch_size:(batch+1)*batch_size,:]
            Y_batch = T_shuffled[batch*batch_size:(batch+1)*batch_size,:]


            Y_hat, Z = forward(X_batch)

            # Weight updates ----------------------
            Y_hat_T = Y_hat-Y_batch
            W2 -= learning_rate * (Z.T.dot(Y_hat_T) + reg*W2)
            b2 -= learning_rate * (Y_hat_T.sum() + reg*b2)

            val = Y_hat_T.dot(W2.T) * (1-Z*Z) #tanh
            W1 -= learning_rate * (X_batch.T.dot(val) + reg*W1)
            b1 -= learning_rate * (val.sum() + reg*b1)
            # -------------------------------------

            Y_hat_valid, _ = self.forward(X_valid)
            c = cross_entropy(T_valid, Y_hat_valid)
            costs.append(c)
            
            if ibatch % (n_batches) == 0:
                e = error_rate(Y_valid, np.argmax(Y_hat_valid, axis=1))
                print("epoch:", epoch, " cost:", c, " error:", e)

    if show_fig:
        plt.plot(costs)
        plt.title('Validation cost')
        plt.show()

    #print("Final train classification_rate:", self.score(X, Y))


    #######################################################
    

    X_test_compressed = pca.transform(X_test)
    print("Test classification_rate:", nn_classify.score(X_test_compressed, Y_test))

if __name__ == '__main__':
    main()
    






