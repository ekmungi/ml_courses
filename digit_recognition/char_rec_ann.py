import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from utils import cross_entropy, one_hot_encoder, softmax, get_data, error_rate, sigmoid


class NeuralNet(object):
    def __init__(self, M):
        self.M = M # number of hidden layer units
        pass

    def fit(self, X, Y, learning_rate=1e-8, reg=1e-12, epochs=10000, show_fig=False):
        
        D = X.shape[1] # number of features
        K = len(set(Y)) # number of classes

        X, Y = shuffle(X, Y)
        X_valid, Y_valid = X[-1000:], Y[-1000:]
        T_valid = one_hot_encoder(Y_valid)
        X, Y = X[:-1000], Y[:-1000]

        T = one_hot_encoder(Y)

        self.W1 = np.random.randn(D, self.M) / np.sqrt(D)
        self.b1 = np.zeros(self.M)
        self.W2 = np.random.randn(self.M, K) / np.sqrt(self.M)
        self.b2 = np.zeros(K)

        costs = []
        best_validation_error = 1
        for epoch in range(epochs):
            Y_hat, Z = self.forward(X)

            # Weight updates ----------------------
            Y_hat_T = Y_hat-T
            self.W2 -= learning_rate * (Z.T.dot(Y_hat_T) + reg*self.W2)
            self.b2 -= learning_rate * (Y_hat_T.sum() + reg*self.b2)

            val = Y_hat_T.dot(self.W2.T) * (1-Z*Z) #tanh
            self.W1 -= learning_rate * (X.T.dot(val) + reg*self.W1)
            self.b1 -= learning_rate * (val.sum() + reg*self.b1)
            # -------------------------------------
            
            if epoch % 10 == 0:
                Y_hat_valid, _ = self.forward(X_valid)
                c = cross_entropy(T_valid, Y_hat_valid)
                costs.append(c)
                e = error_rate(Y_valid, np.argmax(Y_hat_valid, axis=1))
                print("epoch:", epoch, "cost:", c, "error:", e)
                if e < best_validation_error:
                    best_validation_error = e
        print("best_validation_error:", best_validation_error)

        if show_fig:
            plt.plot(costs)
            plt.title('Validation cost')

        print("Final train classification_rate:", self.score(X, Y))

    def predict(self, X):
        Y_hat = self.forward(X)
        return np.argmax(Y_hat, axis=1)

    def score(self, X, Y):
        prediction = self.predict(X)
        return np.round(1 - error_rate(Y, prediction), 4)

    def forward(self, X):
        Z = np.tanh(X.dot(self.W1) + self.b1)
        Y_hat = softmax(Z.dot(self.W2) + self.b2)
        return Y_hat, Z        

def plot_characters():
    X, Y, _, _ = getdata()
    while True:
        f, axarr = plt.subplots(3, 4)
        for i in range(10):
            x_select, y_select = X[Y==i], Y[Y==i]
            N_select = len(y_select)
            j = np.random.choice(N_select)
            axarr[i].imshow(np.reshape(x_select[j], (28,28)), cmap='gray')
            axarr[i].set_title(label_map[y_select[j]])
        plt.show()
        prompt = input('Quit? Enter Y:\n')
        if (prompt == 'Y') | (prompt == 'y'):
            break


def main():
    X_train, Y_train = get_data()
    nn_classify = NeuralNet(400)
    nn_classify.fit(X_train, Y_train, epochs=1000, learning_rate=1e-8, reg=1e-8, show_fig=True)

if __name__ == '__main__':
    main()
    






