import os
import numpy as np
np.random.seed(1337)
import tensorflow as tf
tf.set_random_seed(1337)

from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPool2D, Conv2DTranspose, AveragePooling2D, Convolution2D, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras import backend as K

img_rows = 400

img_cols = 400

img_depth = 5

batch_size = 6

def my_loss(y_true, y_pred):
    weights = [0.5, 1.5]
    false_positive_factor = 0.5
    logits = np.reshape(y_pred, [-1, y_true.shape[3]])
    labels = np.reshape(y_true, [-1])

    prediction = np.argmax(logits, 1)
    prediction_map = np.equal(prediction, 1)
    label_map = np.equal(labels, 1)
    false_positive_map = np.logical_and(prediction_map, tf.logical_not(label_map))
    label_map = tf.to_float(label_map)
    false_positive_map = tf.to_float(false_positive_map)

    weight_map = label_map * (weights[1]-weights[0]) + weights[0]
    weight_map = tf.add(weight_map, false_positive_map*((false_positive_factor * weights[1])-weights[0]))
    weight_map = tf.stop_gradient(weight_map)

    cross_enetopy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    weighted_cross_entropy = tf.multiply(weight_map, cross_enetopy)
    loss = tf.reduce_mean(weighted_cross_entropy)
    return loss

def conv_block(inputs, filters, kernal_size, padding='same',BatchNormlization=True):
    conv = Conv2D(filters,kernel_size, padding=padding)(inputs)
    if BatchNormalization:
        conv = BatchNormalization(axis=-1)
    conv = Activation('relu')(conv)
    return conv

def get_unet():
    inputs = Input((img_rows, img_cols, depth))
    conv11 = conv_block(inputs, 64, (3, 3), padding='same')
    conv12 = conv_block(conv11, 64, (3,3))
    pool1 = MaxPool2D((2 ,2), strides=(2, 2))(conv12)

    conv21 = conv_block(pool1, 128, (3, 3))
    conv22 = conv_block(conv21, 128, (3, 3))
    conc2 = concatenate([pool1, conv22], axis=-1)
    pool2 = MaxPool2D((2, 2), strides=(2, 2))(conc2)

    conv31 = conv_block(pool2, 256, (3, 3))
    conv32 = conv_block(conv31, 256, (3, 3))
    conv33 = conv_block(conv32, 256, (3, 3))
    conc3 = concatenate([pool2, conv33], axis=-1)
    pool3 = MaxPool2D((2, 2), strides=(2, 2))(conc3)

    conv41 = conv_block(pool3, 512, (3, 3))
    conv42 = conv_block(conv41, 512, (3, 3))
    conv43 = conv_block(conv42, 512, (3, 3))
    conc4 = concatenate([pool3, conv43], axis=-1) # depth = 512
    pool4 = MaxPool2D((2, 2), strides=(2, 2))(conc4) #depth = 512

    conv51 = conv_block(pool4, 512, (3, 3))
    conv52 = conv_block(conv51, 512, (3, 3))
    conv53 = conv_block(conv52, 512, (3, 3))        # depth = 512
    conc5 = concatenate([pool4, conv53], axis=-1)   # depth =  1024
    pool5 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conc5)  #depth = 512

    up6 = concatenate([conc4, pool5], axis=-1) # depth = 512
    conv61 = conv_block(up6, 512, (3, 3))
    conv62 = conv_block(conv61, 512, (3, 3))
    conv63 = conv_block(conv62, 512, (3, 3))
    conc6 = concatenate([up6, conv63], axis=-1)
    pool6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conc6)

    up7 = concatenate([conc3, pool6], axis=-1)
    conv71 = conv_block(up7, 256, (3, 3))
    conv72 = conv_block(conv71, 256, (3, 3))
    conv73 = conv_block(conv72, 256, (3, 3))
    conc7 = concatenate([up7, conv73], axis=-1)
    pool7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conc7)

    up8 = concatenate([conc2, pool7],axis=-1)
    conv81 = conv_block(up8, 128, [3, 3])
    conv82 = conv_block(conv81, 128, (3, 3))
    conc8 = concatenate([up8, conv82], axis=-1)
    pool8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conc8)

    up9 = concatenate([conv21, pool8], axis=-1)
    conv91 = conv_block(up9, 64, (3, 3))
    conv92 = conv_block(conv91, 64, (3, 3))
    conc9 = concatenate([up9, conv92], axis = -1)

    out = Conv2D(1, (3, 3), activation='sigmoid')(conc9)

    model = Model(inputs=[inputs], outputs=[out])
    model.summary()
    model.compile(optimizer=(1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000000199), loss = my_loss, metrics=['accuracy'])

    return model



def train():
    print('-'*30)
    print("Loading and preprocessing train data...")
    print('-'*30)
    dir = ''
    train_generator = data_generator(batch_size)
    imgs = imgs.astype('float32')

    model = get_unet()
    model.fit_generator(train_generator, steps_per_epoch=50, epochs=100, verbose=1)









