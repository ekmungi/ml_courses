import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from utils import cross_entropy, one_hot_encoder, softmax, get_data, error_rate
import tensorflow as tf


# class TFNeuralNet(object):
#     def __init__(self, M):
#         self.M = M # number of hidden layer units
#         pass

#     def fit(self, X, Y, learning_rate=1e-8, reg=1e-12, epochs=10000, n_batches=10, show_fig=False):
        
        
        
        

#         # hyperparams
#         beta1 = 0.9
#         beta2 = 0.999
#         eps = 1e-8

#         costs = []
#         t = 1
#         for epoch in range(epochs):
#             X_shuffled, T_shuffled = shuffle(X, T)
#             for ibatch in range(n_batches):
#                 # Get the batch
#                 X_batch = X_shuffled[ibatch*batch_size:(ibatch+1)*batch_size,:]
#                 T_batch = T_shuffled[ibatch*batch_size:(ibatch+1)*batch_size,:]

#                 Y_hat, Z = self.forward(X_batch)

#                 # Weight updates ----------------------
#                 Y_hat_T = Y_hat-T_batch
#                 dJ_dW2 = Z.T.dot(Y_hat_T) + reg * self.W2
#                 dJ_db2 = Y_hat_T.sum() + reg * self.b2

#                 val = (Y_hat - T_batch).dot(self.W2.T) * (Z > 0) # Relu
#                 #val = Y_hat_T.dot(self.W2.T) * (1-Z*Z) # tanh
#                 dJ_dW1 = X_batch.T.dot(val) + reg*self.W1
#                 dJ_db1 = val.sum() + reg*self.b1

#                 # Mean
#                 mW2 = beta1*mW2 + (1-beta1)*dJ_dW2
#                 mb2 = beta1*mb2 + (1-beta1)*dJ_db2
#                 mW1 = beta1*mW1 + (1-beta1)*dJ_dW1
#                 mb1 = beta1*mb1 + (1-beta1)*dJ_db1

#                 # Velocity terms
#                 vW2 = beta2*vW2 + (1-beta2)*dJ_dW2*dJ_dW2
#                 vb2 = beta2*vb2 + (1-beta2)*dJ_db2*dJ_db2
#                 vW1 = beta2*vW1 + (1-beta2)*dJ_dW1*dJ_dW1
#                 vb1 = beta2*vb1 + (1-beta2)*dJ_db1*dJ_db1

#                 correction1 = 1 - beta1**t
#                 hat_mW2 = mW2/correction1
#                 hat_mb2 = mb2/correction1
#                 hat_mW1 = mW1/correction1
#                 hat_mb1 = mb1/correction1

#                 correction2 = 1 - beta2**t
#                 hat_vW2 = vW2/correction2
#                 hat_vb2 = vb2/correction2
#                 hat_vW1 = vW1/correction2
#                 hat_vb1 = vb1/correction2
                
                
#                 self.W2 -= learning_rate * hat_mW2/(np.sqrt(hat_vW2) + eps)
#                 self.b2 -= learning_rate * hat_mb2/(np.sqrt(hat_vb2) + eps)
#                 self.W1 -= learning_rate * hat_mW1/(np.sqrt(hat_vW1) + eps)
#                 self.b1 -= learning_rate * hat_mb1/(np.sqrt(hat_vb1) + eps)
#                 # -------------------------------------

#                 Y_hat_valid, _ = self.forward(X_valid)
#                 c = cross_entropy(T_valid, Y_hat_valid)
#                 costs.append(c)
                
#                 if ibatch % (n_batches) == 0:
#                     e = error_rate(Y_valid, np.argmax(Y_hat_valid, axis=1))
#                     print("epoch:", epoch, " cost:", c, " error:", e)

#                 t += 1

#         if show_fig:
#             plt.plot(costs)
#             plt.title('Validation cost')
#             plt.show()

#         print("Final train classification_rate:", self.score(X, Y))

#     def predict(self, X):
#         Y_hat, _ = self.forward(X)
#         return np.argmax(Y_hat, axis=1)

#     def score(self, X, Y):
#         prediction = self.predict(X)
#         return np.round(1 - error_rate(Y, prediction), 4)

#     def forward(self, X):
#         # Relu
#         Z = X.dot(self.W1) + self.b1
#         Z[Z < 0] = 0

#         # Z = np.tanh(X.dot(self.W1) + self.b1)
#         Y_hat = softmax(Z.dot(self.W2) + self.b2)
#         return Y_hat, Z        

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
    file_loc = '/media/avemuri/DEV/Data/deeplearning/mnist/train.csv'
    #file_loc = 'D:/dev/data/mnist/train.csv'
    X_train, Y_train, X_test, Y_test = get_data(file_name=file_loc, split_train_test=True)
    
    n_batches = 10
    learning_rate = 0.00004
    epochs = 15


    # compress the data
    pca = PCA(n_components=400)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)


    D = X_train.shape[1]    # number of features
    K = len(set(Y_train))   # number of classes
    M1 = 300
    M2 = 400
    

    #X_train, Y_train = shuffle(X_train, Y_train)
    #X_valid, Y_valid = X_train[-1000:], Y_train[-1000:]
    T_test = one_hot_encoder(Y_test)
    T_train = one_hot_encoder(Y_train)

    batch_size = 500#X_train.shape[0]//n_batches
    n_batches = X_train.shape[0]//batch_size

    W1_init = np.random.randn(D, M1) / np.sqrt(D)
    b1_init = np.zeros(M1)
    W2_init = np.random.randn(M1, M2) / np.sqrt(M1)
    b2_init = np.zeros(M2)
    W3_init = np.random.randn(M2, K) / np.sqrt(M2)
    b3_init = np.zeros(K)


    # define tensorflow data and variables
    X = tf.placeholder(tf.float32, shape=(None, D), name='X')
    T = tf.placeholder(tf.float32, shape=(None, K), name='T')
    W1 = tf.Variable(W1_init.astype(np.float32))
    b1 = tf.Variable(b1_init.astype(np.float32))
    W2 = tf.Variable(W2_init.astype(np.float32))
    b2 = tf.Variable(b2_init.astype(np.float32))
    W3 = tf.Variable(W3_init.astype(np.float32))
    b3 = tf.Variable(b3_init.astype(np.float32))

    # Define the computation graph
    # Forward pass
    Z1 = tf.nn.relu(tf.matmul(X, W1) + b1)
    Z2 = tf.nn.relu(tf.matmul(Z1, W2) + b2)
    Y_hat = tf.matmul(Z2, W3) + b3
    # Prediction
    predict_op = tf.argmax(Y_hat, axis=1)

    # Cost function
    cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Y_hat, labels=T))
    #optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.99, momentum=0.9)
    train_op = tf.train.RMSPropOptimizer(learning_rate, decay=0.99, momentum=0.9).minimize(cost)
    

    costs = []
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)

        for epoch in range(epochs):
            X_shuffled, T_shuffled = shuffle(X_train, T_train)
            for batch in range(n_batches):
                X_batch = X_train[batch*batch_size:(batch+1)*batch_size,:]
                T_batch = T_train[batch*batch_size:(batch+1)*batch_size,:]

                session.run(train_op, feed_dict={X:X_batch, T:T_batch})

                if batch % 10 == 0:
                    test_cost = session.run(cost, feed_dict={X:X_test, T:T_test})
                    prediction = session.run(predict_op, feed_dict={X:X_test})
                    err = error_rate(Y_test, prediction)
                    print("Cost / err at epoch=%d, batch=%d: %.3f / %.3f" % (epoch, batch, test_cost, err))
                    costs.append(test_cost)
        

    plt.plot(costs)

    ##############################################################

    file_loc = '/media/avemuri/DEV/Data/deeplearning/mnist/train.csv'
    #file_loc = 'D:/dev/data/mnist/train.csv'
    X_train, Y_train, X_test, Y_test = get_data(file_name=file_loc, split_train_test=True)
    
    n_batches = 10
    learning_rate = 0.00004
    epochs = 15


    # compress the data
    # pca = PCA(n_components=400)
    # pca.fit(X_train)
    # X_train = pca.transform(X_train)
    # X_test = pca.transform(X_test)


    D = X_train.shape[1]    # number of features
    K = len(set(Y_train))   # number of classes
    M1 = 300
    M2 = 400
    

    #X_train, Y_train = shuffle(X_train, Y_train)
    #X_valid, Y_valid = X_train[-1000:], Y_train[-1000:]
    T_test = one_hot_encoder(Y_test)
    T_train = one_hot_encoder(Y_train)

    batch_size = 500#X_train.shape[0]//n_batches
    n_batches = X_train.shape[0]//batch_size

    W1_init = np.random.randn(D, M1) / np.sqrt(D)
    b1_init = np.zeros(M1)
    W2_init = np.random.randn(M1, M2) / np.sqrt(M1)
    b2_init = np.zeros(M2)
    W3_init = np.random.randn(M2, K) / np.sqrt(M2)
    b3_init = np.zeros(K)


    # define tensorflow data and variables
    X = tf.placeholder(tf.float32, shape=(None, D), name='X')
    T = tf.placeholder(tf.float32, shape=(None, K), name='T')
    W1 = tf.Variable(W1_init.astype(np.float32))
    b1 = tf.Variable(b1_init.astype(np.float32))
    W2 = tf.Variable(W2_init.astype(np.float32))
    b2 = tf.Variable(b2_init.astype(np.float32))
    W3 = tf.Variable(W3_init.astype(np.float32))
    b3 = tf.Variable(b3_init.astype(np.float32))

    # Define the computation graph
    # Forward pass
    Z1 = tf.nn.relu(tf.matmul(X, W1) + b1)
    Z2 = tf.nn.relu(tf.matmul(Z1, W2) + b2)
    Y_hat = tf.matmul(Z2, W3) + b3
    # Prediction
    predict_op = tf.argmax(Y_hat, axis=1)

    # Cost function
    cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Y_hat, labels=T))
    #optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.99, momentum=0.9)
    train_op = tf.train.RMSPropOptimizer(learning_rate, decay=0.99, momentum=0.9).minimize(cost)
    

    costs = []
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)

        for epoch in range(epochs):
            X_shuffled, T_shuffled = shuffle(X_train, T_train)
            for batch in range(n_batches):
                X_batch = X_train[batch*batch_size:(batch+1)*batch_size,:]
                T_batch = T_train[batch*batch_size:(batch+1)*batch_size,:]

                session.run(train_op, feed_dict={X:X_batch, T:T_batch})

                if batch % 10 == 0:
                    test_cost = session.run(cost, feed_dict={X:X_test, T:T_test})
                    prediction = session.run(predict_op, feed_dict={X:X_test})
                    err = error_rate(Y_test, prediction)
                    print("Cost / err at epoch=%d, batch=%d: %.3f / %.3f" % (epoch, batch, test_cost, err))
                    costs.append(test_cost)
        

    plt.plot(costs)


    plt.title('Validation cost')
    plt.show()


    #X_test_compressed = pca.transform(X_test)
    #print("Test classification_rate:", nn_classify.score(X_test, Y_test))

if __name__ == '__main__':
    main()
    






