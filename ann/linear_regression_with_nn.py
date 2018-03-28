from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


# Generate data
N = 500
X = np.random.random((N, 2))*4 - 2
Y = X[:,0]*X[:,1]

def sigmoid(a):
    return 1/(1+np.exp(-a))

def forward_ant(X, W1, b1, W2, b2):
    a1 = X.dot(W1) + b1
    z = sigmoid(a1)
    
    Y_hat = z.dot(W2) + b2
        
    return Y_hat, z

def get_cost(Y, Yhat):
    return ((Y - Yhat)**2).mean()

# *************** Derivatives *************** #
def dJ_dw2(T, Y, Z):
    return Z.T.dot(T-Y)

def dJ_dw1(T, Y, Z, X, W2):
    return X.T.dot(np.outer(T-Y, W2) * Z * (1-Z))

def dJ_b2(T, Y):
    return (T - Y).sum(axis=0)

def dJ_b1(T, Y, W2, Z):
    return (np.outer(T-Y, W2) * Z * (1 - Z)).sum(axis=0)
# ******************************************* #


D = X.shape[1]       # number of features
M = 100              # number of nodes in the hidden layer

W1 = np.random.randn(D, M) / np.sqrt(D)
b1 = np.zeros(M)

W2 = np.random.randn(M) / np.sqrt(M)
b2 = 0


learning_rate = 0.00005
costs_train = []
#costs_test = []
for epoch in range(10000):
    Y_hat, hidden = forward_ant(X, W1, b1, W2, b2)
    #Y_hat_test, temp_ignore = forward_ant(X_test, W1, b1, W2, b2)

    W2 += learning_rate * dJ_dw2(Y, Y_hat, hidden)
    b2 += learning_rate * dJ_b2(Y, Y_hat)
    W1 += learning_rate * dJ_dw1(Y, Y_hat, hidden, X, W2)
    b1 += learning_rate * dJ_b1(Y, Y_hat, W2, hidden)
    
    ctrain = get_cost(Y, Y_hat)
    
    costs_train.append(ctrain)
    #costs_test.append(ctest)
    
    if epoch % 1000 == 0:
        print(epoch, ctrain)


legend1, = plt.plot(costs_train, label='train cost')
#legend2, = plt.plot(costs_test, label='test cost')
plt.legend([legend1])
plt.show()




# Visualize data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], Y)
plt.show()



# plot the prediction with the data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], Y)

# surface plot
line = np.linspace(-2, 2, 20)
xx, yy = np.meshgrid(line, line)
Xgrid = np.vstack((xx.flatten(), yy.flatten())).T
Yhat, _ = forward_ant(Xgrid, W1, b1, W2, b2)
print(Xgrid.shape, Yhat.shape)
ax.plot_trisurf(Xgrid[:,0], Xgrid[:,1], Yhat, linewidth=0.2, antialiased=True)
plt.show()




# plot magnitude of residuals
Ygrid = Xgrid[:,0]*Xgrid[:,1]
R = np.abs(Ygrid - Yhat)

plt.scatter(Xgrid[:,0], Xgrid[:,1], c=R)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(Xgrid[:,0], Xgrid[:,1], R, linewidth=0.2, antialiased=True)
plt.show()


