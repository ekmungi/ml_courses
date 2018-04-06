import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from utils import cross_entropy, one_hot_encoder, softmax, get_data, error_rate

import theano.tensor as th
import theano

def init_weights(M1, M2):
    return np.random.randn(M1, M2) / np.sqrt(M1)


class HiddenLayer(object):
    def __init__(self, M1, M2, activation_fn):
        self.M1 = M1
        self.M2 = M2
        self.activation_fn = activation_fn
        W = init_weights(M1, M2)
        b = np.zeros(M2)

        self.W = theano.shared(W)
        self.b = theano.shared(b)

        self.params = [self.W, self.b]

    def forward(self, X):
        return self.activation_fn(X.dot(self.W) + self.b)

class ANN(object):
    def __init__(self, hidden_layer_sizes):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.layers = []
        self.params = []

    def fit(self, X, Y, activation=th.nnet.relu, learning_rate=1e-8, reg=1e-12, epochs=10000, n_batches=10, decay_rate=0.9, show_fig=False):
        X = X.astype(np.float32)
        Y = Y.astype(np.int32)


        X, Y = shuffle(X, Y)
        X_valid, Y_valid = X[-1000:], Y[-1000:]
        T_valid = one_hot_encoder(Y_valid)
        X, Y = X[:-1000], Y[:-1000]
        T = one_hot_encoder(Y)


        eps = 1e-10
        D = X.shape[1] # number of features
        K = len(set(Y)) # number of classes
        batch_size = X.shape[0]// n_batches
        print_time = n_batches//1

        M1 = D
        for M2 in self.hidden_layer_sizes:
            h = HiddenLayer(M1, M2, activation_fn=activation)
            self.layers.append(h)
            M1 = M2

        # the final layer
        h = HiddenLayer(M1, K, activation_fn=th.nnet.softmax)
        self.layers.append(h)

        for layer in self.layers:
            self.params += layer.params


        dparams = [theano.shared(np.zeros_like(p.get_value())) for p in self.params]
        cache = [theano.shared(np.zeros_like(p.get_value())) for p in self.params]


        thX = th.matrix('X')
        thT = th.matrix('T')
        thY = self.forward(thX)

        # Cost
        regularization_cost = reg * th.mean([(p*p).sum() for p in self.params])
        #cost = -th.mean(th.log(thY[th.arange(thT.shape[0]), thT])) #+ regularization_cost
        cost = -th.mean(thT*th.log(thY)) + regularization_cost

        
        # Predictions
        prediction = th.argmax(thY, axis=1)

        # Gradient
        grads = th.grad(cost, self.params)

        cost_predict_op = theano.function(inputs=[thX, thT], 
                                          outputs=[cost, prediction])

        update_params = [(p, p - learning_rate*(decay_rate*v + (1-decay_rate)*g + reg*p)) for g, v, p in zip(grads, dparams, self.params)]
        update_velocity = [(v, decay_rate*v + (1-decay_rate)*g) for g, v in zip(grads, dparams)]
        # updates = [(p, p - learning_rate*g) for g, p in zip(grads, self.params)]
        updates = update_params + update_velocity
        


        train_op = theano.function(inputs=[thX, thT], updates=updates)

        costs = []
        for epoch in range(epochs):
            X_shuffled, T_shuffled = shuffle(X, T)
            for batch in range(n_batches):
                # Get the batch
                X_batch = X_shuffled[batch*batch_size:(batch+1)*batch_size,:]
                Y_batch = T_shuffled[batch*batch_size:(batch+1)*batch_size,:]
                
                train_op(X_batch, Y_batch)

                if batch % print_time == 0:
                    test_cost, prediction = cost_predict_op(X_valid, T_valid)
                    err = error_rate(Y_valid, prediction)
                    # print(prediction.shape)
                    print("epoch [%d], batch [%d] : cost=[%.3f], error=[%.3f]" %(epoch, batch, test_cost, err))
                    costs.append(test_cost)

        plt.plot(costs)
        plt.title('Validation cost')
        plt.show()


    def forward(self, X):
        Z = X
        for layer in self.layers:
            Z = layer.forward(Z)
        return Z

#### TO TEST
# def main():
#     file_loc = '/media/avemuri/DEV/Data/deeplearning/mnist/train.csv'
#     #file_loc = 'D:/dev/data/mnist/train.csv'
#     X_train, Y_train = get_data(file_name=file_loc, split_train_test=False)
#     pca = PCA(n_components=400)
#     pca.fit(X_train)
#     X_train_compressed = pca.transform(X_train)
    
#     ann_classify = ANN([400])
#     ann_classify.fit(X_train, Y_train, epochs=200, learning_rate=0.07, reg=1e-4)


# if __name__ == '__main__':
#     main()
    






