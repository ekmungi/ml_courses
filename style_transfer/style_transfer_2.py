import numpy as np
import keras.backend as K
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.models import Model

from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave

import matplotlib.pyplot as plt

'''
This code from when I did the Udemy course Advanced computer vision
by Lazy programmer. This is an alternate approach.
https://www.udemy.com/advanced-computer-vision/learn/v4/overview
'''

root_folder = 'C:/dev/data/style_transfer/'

content_image_file = root_folder + 'dog_image.jpg'
style_image_file = root_folder + 'Rembrandt_Dream_of_Joseph.jpg'
generated_image_file = root_folder + 'results/output.jpg'

height = 480
width = 640
size = (height, width)

content_weight = 0.5
style_weight = 10.

### Load content image ###
content_img = image.load_img(content_image_file)
plt.imshow(content_img)
content_arr = np.expand_dims(np.asarray(content_img, dtype='float32'), axis=0)
print('Content image shape: {0}'.format(content_arr.shape))
plt.show()

### Load style image ###
style_img = image.load_img(style_image_file)
plt.imshow(style_img)
style_arr = np.expand_dims(np.asarray(style_img, dtype='float32'), axis=0)
print('Style image shape: {0}'.format(style_arr.shape))
plt.show()

### Mean center image ###
content_arr[:, :, :, 0] -= 103.939
content_arr[:, :, :, 1] -= 116.779
content_arr[:, :, :, 2] -= 123.68
content_arr = content_arr[:, :, :, ::-1]

style_arr[:, :, :, 0] -= 103.939
style_arr[:, :, :, 1] -= 116.779
style_arr[:, :, :, 2] -= 123.68
style_arr = style_arr[:, :, :, ::-1]

### Initialize loss ###
loss = K.variable(0.)

### Setup variables and input tensor ###
content_image = K.variable(content_arr)
style_image = K.variable(style_arr)
generated_image = K.placeholder((1, height, width, 3))
input_tensor = K.concatenate([content_image, style_image, generated_image], axis=0)
print(input_tensor.shape)

### Load VGG16 model ###
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

### Get all layers for later ###
layers = dict([(layer.name, layer.output) for layer in vgg16.layers])
print(layers)

### Content loss ###
def content_loss(C, G):
    return K.sum(K.square(C-G))

layer_features = layers['block2_conv2']

content_features = layer_features[0, :, :, :]
generated_features = layer_features[2, :, :, :]

loss = loss + content_loss(content_features, generated_features)

### Style loss ###
def gram_matrix(a):
    return K.dot(a, K.transpose(a))

def style_loss(S, G):
    n_H, n_W, n_C = G.shape

    S = K.reshape(K.transpose(S), [n_C, n_H*n_W])
    G = K.reshape(K.transpose(G), [n_C, n_H*n_W])

    GS = gram_matrix(S)
    GG = gram_matrix(G)

    return K.sum(K.square(GS-GG)) / (4 * 3*3 * 640**2 * 480**2 )


feature_layers = ['block1_conv2', 'block2_conv2',
                  'block3_conv3', 'block4_conv3',
                  'block5_conv3']

for layer_name in feature_layers:
    layer_features = layers[layer_name]
    style_features = layer_features[1, :, : , :]
    generated_features = layer_features[2, :, : , :]
    loss = loss + (style_weight/len(feature_layers)) * style_loss(style_features, generated_features)


### Total variation loss ###
# IGNORE FOR NOW


grads = K.gradients(loss, generated_image)
outputs = [loss] + grads
f_outputs = K.function([generated_image], outputs)


def compute_loss_grads(x):
    x = x.reshape([1, height, width, 3])
    outs = f_outputs([x])
    loss_value = outs[0]
    grads_value = outs[1].flatten().astype('float64')
    return loss_value, grads_value


class Evaluator(object):
    
    def __init__(self):
        self.loss_value = None
        self.grad_values = None
        
    def loss(self, x):
        assert self.loss_value is None
        self.loss_value, self.grad_values = compute_loss_grads(x)
        return self.loss_value
    
    def grads(self, x):
        assert self.grad_values is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values
    
    
evaluator = Evaluator()

x = np.random.uniform(low=0, high=255, size=(1, height, width, 3))

### Optimize ###
for i in range(10):
    x, fmin_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(), 
                                      fprime=evaluator.grads, maxfun=10)
    print(fmin_val)


x = x.reshape([height, width, 3])
# x = x[:, :, ::-1]
# x[:, :, 0] += 103.939
# x[:, :, 1] += 116.779
# x[:, :, 2] += 123.68
x = np.clip(x, 0, 255).astype('uint8')
plt.imshow(x)




