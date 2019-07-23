import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from face_recognition.utils import cross_entropy, one_hot_encoder, softmax, get_data, error_rate

label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

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
        print("Final train classification_rate:", self.score(Y, self.predict(Y_hat)))

    def predict(self, Y_hat):
        return np.argmax(Y_hat, axis=1)

    def score(self, Y, Y_hat_class):
        return 100*np.mean(Y==Y_hat_class)

    def forward(self, X):
        return softmax(X.dot(self.W) + self.b)

    def dJ_dw(self, Y, Y_hat, X):
        return X.T.dot(Y_hat-Y)

    def dJ_db(self, Y, Y_hat):
        return (Y_hat-Y).sum(axis=0)


def main():
    X, Y, _, _ = getdata()
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
    X_train, Y_train, X_test, Y_test = get_data()
    lr_classify = LogisticRegression()
    lr_classify.fit(X_train, Y_train, epochs=1000, reg=1e-6, show_fig=True)
    






