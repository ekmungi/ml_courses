import os
from keras import layers
from keras.layers import Dense, Activation, Flatten, BatchNormalization, Conv2D, Input
from keras.layers import MaxPool2D, ZeroPadding2D, AveragePooling2D, GlobalMaxPooling2D
from keras.models import Model
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.engine.topology import get_source_inputs
import keras.backend as K
import numpy as np

WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'


def identity_block(input_tensor, filters, kernel_size):
    filter_1, filter_2, filter_3 = filters

    x = Conv2D(filters=filter_1, kernel_size=(1,1), padding='valid')(input_tensor)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=filter_2, kernel_size=kernel_size, padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=filter_3, kernel_size=(1,1), padding='valid')(x)
    x = BatchNormalization(axis=3)(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)

    return x


def conv_block(input_tensor, filters, kernel_size, strides=(2,2)):
    filter_1, filter_2, filter_3 = filters

    x = Conv2D(filters=filter_1, kernel_size=(1,1), padding='valid', strides=strides)(input_tensor)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=filter_2, kernel_size=(3,3), padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=filter_3, kernel_size=(1,1), padding='valid')(x)
    x = BatchNormalization(axis=3)(x)

    shortcut = Conv2D(filters=filter_3, kernel_size=(1,1), 
                      strides=strides, padding='valid')(input_tensor)
    shortcut = BatchNormalization(axis=3)(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)

    return x



def ResNet50(input_tensor=None, input_shape=None, include_top=False, n_classes=1000):
    
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor


    x = ZeroPadding2D((3,3))(img_input)
    x = Conv2D(filters=64, kernel_size=(7,7), strides=(2,2))(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    x = MaxPool2D((3,3), strides=(2,2))(x)

    x = conv_block(x, [64, 64, 256], strides=(1,1), kernel_size=3)
    x = identity_block(x, [64, 64, 256], kernel_size=3)
    x = identity_block(x, [64, 64, 256], kernel_size=3)

    x = conv_block(x, [128, 128, 512], kernel_size=3)
    x = identity_block(x, [128, 128, 512], kernel_size=3)
    x = identity_block(x, [128, 128, 512], kernel_size=3)
    x = identity_block(x, [128, 128, 512], kernel_size=3)

    x = conv_block(x, [256, 256, 1024], kernel_size=3)
    x = identity_block(x, [256, 256, 1024], kernel_size=3)
    x = identity_block(x, [256, 256, 1024], kernel_size=3)
    x = identity_block(x, [256, 256, 1024], kernel_size=3)
    x = identity_block(x, [256, 256, 1024], kernel_size=3)
    x = identity_block(x, [256, 256, 1024], kernel_size=3)

    x = conv_block(x, [512, 512, 2048], kernel_size=3)
    x = identity_block(x, [512, 512, 2048], kernel_size=3)
    x = identity_block(x, [512, 512, 2048], kernel_size=3)

    x = AveragePooling2D((7,7))(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(n_classes, activation='softmax')(x)
    #else:
    #    x = GlobalMaxPooling2D()(x)
        

    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = Model(inputs=inputs, outputs=x)

    # load weights
    if include_top:
        print('Using weights with the top layers...')
        weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                                WEIGHTS_PATH,
                                cache_subdir='models',
                                md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
    else:
        print('Using weights without the top layers...')
        weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                WEIGHTS_PATH_NO_TOP,
                                cache_subdir='models',
                                md5_hash='a268eb855778b3df3c7506639542a6af')

    model.load_weights(weights_path)

    for layer in model.layers:
        layer.trainable = False

    # Adding additional layers when not using for imagenet.
    # if not include_top:
    #     x = Flatten()(model.output)
    #     predictions = Dense(n_classes, activation='softmax')(x)
    #     resnet = Model(inputs=model.input, outputs=predictions)
    #     return resnet
    # else:
    return model


if __name__ == '__main__':
    
    IMAGE_SIZE = [224,224]
    X = np.random.random((1, 224, 224, 3))

    model = ResNet50(input_shape=IMAGE_SIZE+[3], include_top=False)
    model.summary()

    x = Flatten()(model.output)
    predictions = Dense(units=60, activation='softmax')(x)
    resnet50 = Model(inputs=model.input, outputs=predictions)
    resnet50.summary()