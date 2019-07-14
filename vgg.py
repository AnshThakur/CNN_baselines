# -*- coding: utf-8 -*-
"""
Summary:  VGG_bioacoustics
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


def train_model_1():

    inputs=Input(shape=(40,150,1), name='in_layer')

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
    x=Flatten()(x)
    x=Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x=Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x) 
    x = Dense(42, activation='softmax')(x)


    model = Model(inputs,x)
    # model=Model(cnn)
    model.summary()
    return model


'''
def train_model():
    
    a1=Input(shape=(40,150,1), name='in_layer')
    cnn=Conv2D(64, (5, 5), strides=(1, 1), padding="same", activation="relu")(a1)
    cnn=Conv2D(64, (2, 1), strides=(2, 1), padding="same", activation='relu')(cnn)
    cnn=Conv2D(64, (5, 5), strides=(1, 1), padding="same", activation="relu")(cnn)
    cnn=Conv2D(64, (2, 1), strides=(2, 1), padding="same", activation='relu')(cnn)
    cnn=Conv2D(64, (5, 5), strides=(1, 1), padding="same", activation="relu")(cnn)
    cnn=Conv2D(64, (2, 1), strides=(2, 1), padding="same", activation='relu')(cnn)
    cnn=Conv2D(64, (5, 5), strides=(1, 1), padding="same", activation="relu")(cnn)
    cnn=Conv2D(64, (5, 1), strides=(5, 1), padding="same", activation='relu')(cnn)
    cnn=Reshape((1,150,64))(cnn)
    cnn=GlobalAveragePooling2D()(cnn)
    cnn=Dense(512,activation='relu')(cnn)
    cnn=Dense(256,activation='relu')(cnn)
    cnn=Dense(64,activation='relu')(cnn)
    cnn=Dense(26,activation='softmax')(cnn)
   ## fourth block

    model = Model(a1,cnn)
   # model=Model(cnn)
    model.summary()
    return model

'''






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
filepath="Birdcalls_71_VGG.h5"
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




