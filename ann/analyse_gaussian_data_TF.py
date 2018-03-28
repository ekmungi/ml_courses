import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# create random training data again
Nclass = 500
D = 2 # dimensionality of input
M = 3 # hidden layer size
K = 3 # number of classes

X1 = np.random.randn(Nclass, D) + np.array([0, -2])
X2 = np.random.randn(Nclass, D) + np.array([2, 2])
X3 = np.random.randn(Nclass, D) + np.array([-2, 2])
X = np.vstack([X1, X2, X3]).astype(np.float32)

Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)

plt.scatter(X[:,0], X[:,1], c=Y)
plt.show()


def one_hot_encoder(data):
    # One-hot encoding
    unique_time = np.unique(data)
    #print(unique_time)
    one_hot = np.zeros((data.shape[0], len(unique_time)))
    for t in unique_time:
        one_hot[:,int(t)] = np.where(data==t, 1, 0)
        
    return one_hot

Y_ind = one_hot_encoder(Y)
print("Indiacator matrix for the data: ", Y_ind.shape)


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def forward(X, W1, b1, W2, b2):
    Z = tf.nn.sigmoid(tf.matmul(X, W1) + b1)
    return tf.matmul(Z, W2) + b2

tfX = tf.placeholder(tf.float32, [None, D])
tfY = tf.placeholder(tf.float32, [None, K])

W1 = init_weights([D,M])
b1 = init_weights([M])
W2 = init_weights([M,K])
b2 = init_weights([K])


py_x = forward(X, W1, b1, W2, b2)

cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(
        labels=tfY, logits=py_x))

train_op = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(cost)

predict_op = tf.argmax(py_x, 1)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    sess.run(train_op, feed_dict={tfX: X, tfY: Y_ind})
    pred = sess.run(predict_op, feed_dict={tfX: X, tfY: Y_ind})
    if i % 100 == 0:
        print("Accuracy:", np.mean(Y == pred))

