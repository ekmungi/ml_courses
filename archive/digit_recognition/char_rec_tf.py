import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from utils import cross_entropy, one_hot_encoder, softmax, get_data, error_rate
import tensorflow as tf


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

    # file_loc = '/media/avemuri/DEV/Data/deeplearning/mnist/train.csv'
    # #file_loc = 'D:/dev/data/mnist/train.csv'
    # X_train, Y_train, X_test, Y_test = get_data(file_name=file_loc, split_train_test=True)
    
    # n_batches = 10
    # learning_rate = 0.00004
    # epochs = 15


    # # compress the data
    # # pca = PCA(n_components=400)
    # # pca.fit(X_train)
    # # X_train = pca.transform(X_train)
    # # X_test = pca.transform(X_test)


    # D = X_train.shape[1]    # number of features
    # K = len(set(Y_train))   # number of classes
    # M1 = 300
    # M2 = 400
    

    # #X_train, Y_train = shuffle(X_train, Y_train)
    # #X_valid, Y_valid = X_train[-1000:], Y_train[-1000:]
    # T_test = one_hot_encoder(Y_test)
    # T_train = one_hot_encoder(Y_train)

    # batch_size = 500#X_train.shape[0]//n_batches
    # n_batches = X_train.shape[0]//batch_size

    # W1_init = np.random.randn(D, M1) / np.sqrt(D)
    # b1_init = np.zeros(M1)
    # W2_init = np.random.randn(M1, M2) / np.sqrt(M1)
    # b2_init = np.zeros(M2)
    # W3_init = np.random.randn(M2, K) / np.sqrt(M2)
    # b3_init = np.zeros(K)


    # # define tensorflow data and variables
    # X = tf.placeholder(tf.float32, shape=(None, D), name='X')
    # T = tf.placeholder(tf.float32, shape=(None, K), name='T')
    # W1 = tf.Variable(W1_init.astype(np.float32))
    # b1 = tf.Variable(b1_init.astype(np.float32))
    # W2 = tf.Variable(W2_init.astype(np.float32))
    # b2 = tf.Variable(b2_init.astype(np.float32))
    # W3 = tf.Variable(W3_init.astype(np.float32))
    # b3 = tf.Variable(b3_init.astype(np.float32))

    # # Define the computation graph
    # # Forward pass
    # Z1 = tf.nn.relu(tf.matmul(X, W1) + b1)
    # Z2 = tf.nn.relu(tf.matmul(Z1, W2) + b2)
    # Y_hat = tf.matmul(Z2, W3) + b3
    # # Prediction
    # predict_op = tf.argmax(Y_hat, axis=1)

    # # Cost function
    # cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Y_hat, labels=T))
    # #optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.99, momentum=0.9)
    # train_op = tf.train.RMSPropOptimizer(learning_rate, decay=0.99, momentum=0.9).minimize(cost)
    

    # costs = []
    # init = tf.global_variables_initializer()
    # with tf.Session() as session:
    #     session.run(init)

    #     for epoch in range(epochs):
    #         X_shuffled, T_shuffled = shuffle(X_train, T_train)
    #         for batch in range(n_batches):
    #             X_batch = X_train[batch*batch_size:(batch+1)*batch_size,:]
    #             T_batch = T_train[batch*batch_size:(batch+1)*batch_size,:]

    #             session.run(train_op, feed_dict={X:X_batch, T:T_batch})

    #             if batch % 10 == 0:
    #                 test_cost = session.run(cost, feed_dict={X:X_test, T:T_test})
    #                 prediction = session.run(predict_op, feed_dict={X:X_test})
    #                 err = error_rate(Y_test, prediction)
    #                 print("Cost / err at epoch=%d, batch=%d: %.3f / %.3f" % (epoch, batch, test_cost, err))
    #                 costs.append(test_cost)
        

    # plt.plot(costs)


    plt.title('Validation cost')
    plt.show()


    #X_test_compressed = pca.transform(X_test)
    #print("Test classification_rate:", nn_classify.score(X_test, Y_test))

if __name__ == '__main__':
    main()
    






