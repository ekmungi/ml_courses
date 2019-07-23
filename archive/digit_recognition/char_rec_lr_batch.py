import numpy as np 
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from utils import cross_entropy, one_hot_encoder, softmax, get_data, error_rate

class LogisticRegression(object):
    def __init__(self):
        pass

    def fit(self, X, Y, learning_rate=1e-8, reg=1e-12, epochs=10000, show_fig=False):
        
        D = X.shape[1] # number of features
        K = len(set(Y)) # number of classes

        X, Y = shuffle(X, Y)
        X_valid, Y_valid = X[-1000:], Y[-1000:]
        T_valid = one_hot_encoder(Y_valid)
        X, Y = X[:-1000], Y[:-1000]

        T = one_hot_encoder(Y)

        self.W = np.random.randn(D, K) / np.sqrt(D)
        self.b = np.zeros(K)

        costs = []
        best_validation_error = 1
        for epoch in range(epochs):
            Y_hat = self.forward(X)

            self.W -= learning_rate * (self.dJ_dw(T, Y_hat, X) + reg*self.W)
            self.b -= learning_rate * (self.dJ_db(T, Y_hat) + reg*self.b)
            
            if epoch % 100 == 0:
                Y_hat_valid = self.forward(X_valid)
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
            plt.show()
        print("Final train classification_rate:", self.score(X, Y))

    def predict(self, X):
        Y_hat = self.forward(X)
        return np.argmax(Y_hat, axis=1)

    def score(self, X, Y):
        prediction = self.predict(X)
        return np.round(1 - error_rate(Y, prediction), 4)

    def forward(self, X):
        return softmax(X.dot(self.W) + self.b)

    def dJ_dw(self, Y, Y_hat, X):
        return X.T.dot(Y_hat-Y)

    def dJ_db(self, Y, Y_hat):
        return (Y_hat-Y).sum(axis=0)


def main():
    X, Y = getdata()
    while True:
        f, axarr = plt.subplots(1, 7)
        for i in range(7):
            x_select, y_select = X[Y==i], Y[Y==i]
            N_select = len(y_select)
            j = np.random.choice(N_select)
            axarr[i].imshow(np.reshape(x_select[j], (48,48)), cmap='gray')
            axarr[i].set_title(label_map[y_select[j]])
        plt.show()
        prompt = input('Quit? Enter Y:\n')
        if (prompt == 'Y') | (prompt == 'y'):
            break


if __name__ == '__main__':
    #main()
    X_train, Y_train, X_test, Y_test = get_data(split_train_test=True)
    
    pca = PCA(n_components=100)
    pca.fit(X_train)
    X_train_compressed = pca.transform(X_train)

    lr_classify = LogisticRegression()
    lr_classify.fit(X_train_compressed, Y_train, epochs=1000, learning_rate=0.00004, reg=0.01, show_fig=True)

    X_test_compressed = pca.transform(X_test)
    print("Test classification_rate:", lr_classify.score(X_test_compressed, Y_test))

    






