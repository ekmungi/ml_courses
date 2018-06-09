import numpy as np




def forward(x):
    return x*w

def loss(x, y):
    y_hat = forward(x)
    return (y_hat-y)**2

def gradient(x, y):
    return 2*x*(x*w-y)

x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]

w = 1.0
for epoch in range(50):
    for x, y in zip(x_data, y_data):
        grad = gradient(x, y)
        w -= 0.01*grad
        l = loss(x, y)
        print('epoch', epoch, x, y, np.round(l, 2), np.round(grad, 2), np.round(w,2))

    