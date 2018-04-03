import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from utils import cross_entropy, one_hot_encoder, softmax, get_data, error_rate


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


        cache_W2 = 1
        cache_b2 = 1
        cache_W1 = 1
        cache_b1 = 1
        decay_rate = 0.999
        eps = 1e-10

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
                dJ_dW2 = Z.T.dot(Y_hat_T) + reg * self.W2
                cache_W2 = decay_rate * cache_W2 + (1-decay_rate) * dJ_dW2 * dJ_dW2
                self.W2 -= learning_rate * dJ_dW2/(np.sqrt(cache_W2) + eps)
                
                dJ_db2 = Y_hat_T.sum() + reg * self.b2
                cache_b2 = learning_rate * cache_b2 + (1-decay_rate) * dJ_db2 * dJ_db2
                self.b2 -= learning_rate * dJ_db2/(np.sqrt(cache_b2) + eps)

                val = (Y_hat - Y_batch).dot(self.W2.T) * (Z > 0) # Relu
                #val = Y_hat_T.dot(self.W2.T) * (1-Z*Z) # tanh
                dJ_dW1 = X_batch.T.dot(val) + reg*self.W1
                cache_W1 = learning_rate * cache_W1 + (1-decay_rate) * dJ_dW1 * dJ_dW1
                self.W1 -= learning_rate * dJ_dW1/(np.sqrt(cache_W1) + eps)

                dJ_db1 = val.sum() + reg*self.b1
                cache_b1 = learning_rate * cache_b1 + (1-decay_rate) * dJ_db1 * dJ_db1
                self.b1 -= learning_rate * dJ_db1/(np.sqrt(cache_b1) + eps)
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
        # Relu
        Z = X.dot(self.W1) + self.b1
        Z[Z < 0] = 0

        # Z = np.tanh(X.dot(self.W1) + self.b1)
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
    nn_classify = NeuralNet(200)
    
    pca = PCA(n_components=400)
    pca.fit(X_train)
    X_train_compressed = pca.transform(X_train)

    nn_classify.fit(X_train_compressed, Y_train, epochs=50, learning_rate=0.001, reg=0.01, n_batches=50, show_fig=True)

    X_test_compressed = pca.transform(X_test)
    print("Test classification_rate:", nn_classify.score(X_test_compressed, Y_test))

if __name__ == '__main__':
    main()
    






