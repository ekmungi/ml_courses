import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from utils import cross_entropy, one_hot_encoder, softmax, get_data, error_rate

import tensorflow as tf


def relu(a):
    return a * (a > 0)


def main():
    #file_loc = '/media/avemuri/DEV/Data/deeplearning/mnist/train.csv'
    file_loc = 'D:/dev/data/mnist/train.csv'
    X_train, Y_train, X_test, Y_test = get_data(file_name=file_loc, split_train_test=True)
    
    pca = PCA(n_components=400)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    #Y = Y_train
    T_train = one_hot_encoder(Y_train)
    X_test = pca.transform(X_test)
    T_test = one_hot_encoder(Y_test)
    
    #######################################################

    D = X_train.shape[1] # number of features
    K = len(set(Y_train)) # number of classes
    M = 300
    reg = 0.00001
    batch_size = 500
    n_batches = X_train.shape[0]//batch_size
    learning_rate = 0.00004
    epochs = 10

    W1_init = np.random.randn(D, M) / np.sqrt(D)
    b1_init = np.zeros(M)
    W2_init = np.random.randn(M, K) / np.sqrt(M)
    b2_init = np.zeros(K)

    # Define all variables
    X = tf.placeholder(tf.float32, shape=(None, D), name='X')
    T = tf.placeholder(tf.float32, shape=(None, K), name='Y')
    W1 = tf.Variable(W1_init.astype(np.float32))
    b1 = tf.Variable(b1_init.astype(np.float32))
    W2 = tf.Variable(W2_init.astype(np.float32))
    b2 = tf.Variable(b2_init.astype(np.float32))

    # Model definition
    Z = tf.nn.relu(tf.matmul(X, W1) + b1)
    Y_hat = tf.matmul(Z, W2) + b2

    # Cost
    cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Y_hat, labels=T))

    # Optimization
    train = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.99, momentum=0.9).minimize(cost)
    
    # Predictions
    predic_op = tf.argmax(Y_hat, axis=1)

    costs = []
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)
        for epoch in range(epochs):
            X_shuffled, T_shuffled = shuffle(X_train, T_train)
            for batch in range(n_batches):
                # Get the batch
                X_batch = X_shuffled[batch*batch_size:(batch+1)*batch_size,:]
                Y_batch = T_shuffled[batch*batch_size:(batch+1)*batch_size,:]
                
                session.run(train, feed_dict={X:X_batch, T:Y_batch})

                if batch % 10 == 0:
                    c = session.run(cost, feed_dict={X:X_test, T:T_test})
                    Y_test_predictions = session.run(predic_op, feed_dict={X:X_test})
                    err = error_rate(Y_test, Y_test_predictions)
                    print("epoch [%d], batch [%d] : cost=[%.3f], error=[%.3f]" %(epoch, batch, c, err))
                    costs.append(c)

    plt.plot(costs)
    plt.title('Validation cost')
    plt.show()

    #######################################################


if __name__ == '__main__':
    main()
    






