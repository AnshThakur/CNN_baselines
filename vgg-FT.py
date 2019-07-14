# -*- coding: utf-8 -*-
"""
Summary:  VGG-FT with audioset weights
Author:   Anshul Thakur
Created:  15/08/2018
"""




import keras
from keras import backend as K
import keras.layers
from keras.models import Sequential, Model, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Permute, Lambda, RepeatVector
from keras.layers.convolutional import Conv2D, ZeroPadding2D, AveragePooling2D, MaxPooling2D, Conv1D, MaxPooling1D
from keras.layers.pooling import GlobalMaxPooling2D
from keras.layers import merge, Input, GRU, TimeDistributed, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras.layers.merge import Multiply
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.layers import Activation
initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=2)
import h5py
import scipy.io as sio
import matplotlib.pyplot as plt
############

import numpy as np

###########

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# Conv block with Gated linear unit 
def conv_block(input):
    size=input.shape
    cnn = TimeDistributed(Conv2D(64, (3, 3), strides=(1, 1), padding="same", activation="relu", kernel_initializer=initializer),size)(input)
    cnn = BatchNormalization(axis=-1)(cnn)

#    gates = Lambda(slice1, output_shape=slice_output_shape)(cnn)
#    sig = Lambda(slice2, output_shape=slice_output_shape)(cnn)
#
#    gates = Activation('linear')(gates)
#    sig = Activation('sigmoid')(sig)
#
#    out = Multiply()([gates, sig])
    return cnn




def slice1(x):
    return x[:, :, :, 0:16]

def slice2(x):
    return x[:, :, :, 16:]



def slice_output_shape(input_shape):
    return tuple([input_shape[0],input_shape[1],input_shape[2],16])

def outfunc(vects):
    cla, att = vects    # (N, n_time, n_out), (N, n_time, n_out)
    att = K.clip(att, 1e-7, 1.)
    out = K.sum(cla * att, axis=1) / K.sum(att, axis=1)     # (N, n_out)
    return out

def train_model_1():
    inputs=Input(shape=(40,200,1), name='in_layer')
    x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv1')(inputs)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool1')(x)

    # Block 2
    x = Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool2')(x)

    # Block 3
    x = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv3/conv3_1')(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv3/conv3_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool3')(x)

    # Block 4
    x = Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv4/conv4_1')(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv4/conv4_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool4')(x)
   
    model=Model(inputs,x)
    model.summary()
    return model


model=train_model_1()
from keras.models import load_model
model.load_weights('vggish_audioset_weights_without_fc2.h5')
x = Flatten()(model.output)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = Dense(71, activation='softmax')(x)
model = Model(input = model.input, output =x)




classes = 71
train = np.load('Birdcalls71_train.npy')
train_labels = np.load('Birdcalls71_train_labels.npy')


test = np.load('Birdcalls71_val.npy')
test_labels = np.load('Birdcalls71_val_labels.npy')



label_train = to_categorical(train_labels, classes)
label_test = to_categorical(test_labels, classes)

#x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.35, shuffle=True)
model=train_model_1()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[f1])
filepath="Birdcalls_71_VGG-FT.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_f1', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
history = model.fit(train, label_train,callbacks=callbacks_list,validation_data=(test,label_test), epochs=100, batch_size=32,verbose=2)
#model.save('attn_glu_seg_no_GRU_3_3.h5')

print(history.history.keys())
#print(history.history['categorical_accuracy'])
# summarize history for accuracy
'''
plt.plot(history.history['f1'])
plt.plot(history.history['val_f1'])
plt.title('model f1')
plt.ylabel('f1')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')

plt.savefig('vgg_f1')
# summarize history for loss

plt.clf()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.savefig('vgg_loss')






