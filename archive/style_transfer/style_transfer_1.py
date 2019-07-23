import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
import pandas as pd
import urllib.request
import os, tarfile, zipfile
from glob import glob
import distutils
from distutils import dir_util

import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Reshape, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image

from scipy.optimize import fmin_l_bfgs_b


'''
This code from when I did the Udemy course Advanced computer vision
by Lazy programmer.
https://www.udemy.com/advanced-computer-vision/learn/v4/overview
'''



root_folder = 'C:/dev/data/style_transfer/'

content_image = root_folder + 'dog_image.jpg'
style_image = root_folder + 'Rembrandt_Dream_of_Joseph.jpg'
result_image = root_folder + 'results/output.jpg'

target_height = 640
target_width = 480
target_size = (target_height, target_width)

def get_gram_matrix(img):
    X = K.batch_flatten(K.permute_dimensions(img, (2, 0 , 1)))
    G = K.dot(X, K.transpose(X)) / img.get_shape().num_elements()
    return G

def VGG16_AvgPool(shape):
    
    # Base VGG16 model
    vgg16 = VGG16(include_top=False, weights='imagenet', input_shape=shape)

    new_model = Sequential()
    for layer in vgg16.layers:
        if vgg16.__class__ == MaxPooling2D:
            new_model.add(AveragePooling2D())
        else:
            new_model.add(layer)

    return new_model

def VGG16_AvgPool_CutOff(shape, num_convs):

    # Return model after num_convs convolutions
    # VGG16 has 13 convolutions

    if num_convs<1 or num_convs>13:
        print('num_convs should be in the range [1,13]')
        return None

    model = VGG16_AvgPool(shape)
    new_model = Sequential()
    n = 0
    for layer in model.layers:
        if layer.__class__ == Conv2D:
            n += 1
        new_model.add(layer)
        if n >= num_convs:
            break

    return new_model


def unpreprocess(image):

    # Perform the invers of the preprocess function in Keras.

    image[..., 0] += 103.939
    image[..., 1] += 116.779
    image[..., 2] += 126.68
    image = image[..., ::-1]
    return image


def scale_img(image):

    image = image - image.min()
    image = image / image.max()
    return image

def style_loss(y, t):

    return K.mean(K.square(get_gram_matrix(y) - get_gram_matrix(t)))





def minimize(fn, epochs, batch_shape):

    from datetime import datetime
    t0 = datetime.now()
    losses = []
    x = np.random.rand(np.prod(batch_shape))

    for i in range(10):
        x, l, _ = fmin_l_bfgs_b(func=fn, 
                                x0=x, maxfun=20)
        x = np.clip(x, -127, 127)
        print('iter: {}, loss: {}'.format(i, l))
        losses.append(l)
    
    print('Duration: {}'.format(datetime.now() - t0))
    plt.plot(losses)
    plt.show()

    new_image = x.reshape(*batch_shape)
    final_image = unpreprocess(new_image)

    return final_image[0]


############################################
if __name__ == '__main__':


    # Style image
    img = image.load_img(style_image)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    batch_shape = x.shape
    shape = x.shape[1:]

    style_model = VGG16_AvgPool(shape)

    symbolic_conv_outputs = [
        layer.get_output_at(1) for layer in style_model.layers \
        if layer.name.endswith('conv1')
    ]

    multi_output_model = Model(style_model.input, symbolic_conv_outputs)
    style_layers_output = [K.variable(y) for y in multi_output_model.predict(x)]

    loss = 0
    for symbolic, actual in zip(symbolic_conv_outputs, style_layers_output):
        # print('###########################################')
        # print(actual[0].shape, symbolic[0].shape)
        # print('###########################################')
        loss += style_loss(symbolic[0], actual[0])

    grads = K.gradients(loss, multi_output_model.input)
    
    
    # Callable function to get loss and gradients together given an input 'x'
    loss_grads = K.function(inputs=[multi_output_model.input], 
                            outputs=[loss] + grads)


    def get_loss_grads_wrapper(x_vec):

        l, g = loss_grads([x_vec.reshape(*batch_shape)])
        return l.astype(np.float64), g.flatten().astype(np.float64)

    final_image = minimize(get_loss_grads_wrapper, 10, batch_shape)
    plt.imshow(scale_img(final_image))
    plt.show()