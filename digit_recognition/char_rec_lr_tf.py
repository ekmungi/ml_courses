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
    learning_rate = 0.0004
    epochs=100
    print_time = n_batches//2

    W_init = np.random.randn(D, K) / np.sqrt(D)
    b_init = np.zeros(K)

    X = tf.placeholder(tf.float32, shape=(None, D), name='X_train')
    T = tf.placeholder(tf.float32, shape=(None, K), name='Y')
    W = tf.Variable(W_init.astype(np.float32), name='W')
    b = tf.Variable(b_init.astype(np.float32), name='b')

    # Forward model
    Y_hat = tf.matmul(X, W) + b

    # Cost
    cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Y_hat, labels=T))

    # Optimizer
    train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # Predictions
    predict_op = tf.argmax(Y_hat, axis=1)

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
                
                session.run(train_op, feed_dict={X:X_batch, T:Y_batch})

                if batch % print_time == 0:
                    test_cost = session.run(cost, feed_dict={X:X_test, T:T_test})
                    prediction = session.run(predict_op, feed_dict={X:X_test})
                    err = error_rate(Y_test, prediction)
                    print("epoch [%d], batch [%d] : cost=[%.3f], error=[%.3f]" %(epoch, batch, test_cost, err))
                    costs.append(test_cost)

    plt.plot(costs)
    plt.title('Validation cost')
    plt.show()

    #######################################################


if __name__ == '__main__':
    main()
    






