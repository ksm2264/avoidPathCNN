#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 13:34:28 2021

@author: karl
"""

import tensorflow as tf
from tensorflow.keras import regularizers

#my_reg = regularizers.l2(1e-2)
my_reg = None
filtSize = 10

def getCNN():
    
    model = tf.keras.models.Sequential([  
        tf.keras.layers.Conv2D(4, filtSize,padding='same', activation='relu', input_shape=(50,300,1),kernel_initializer='random_uniform',activity_regularizer=my_reg),
#        tf.keras.layers.Dropout(0.25),
#        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(8, filtSize,padding='same', activation='relu',kernel_initializer='random_uniform',activity_regularizer=my_reg),
#        tf.keras.layers.Dropout(0.25),
#        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(16, filtSize,padding='same',activation='relu',kernel_initializer='random_uniform',activity_regularizer=my_reg),   
#        tf.keras.layers.Dropout(0.25),
#        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(20, activation='relu'),
#        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(1, activation='sigmoid')])

    
    return model

def getCNN_saliency():
    
    model = tf.keras.models.Sequential([  
        tf.keras.layers.Conv2D(4, filtSize,padding='same', activation='relu', input_shape=(101,101,1),kernel_initializer='glorot_uniform',activity_regularizer=my_reg),
#        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(8, filtSize,padding='same', activation='relu',kernel_initializer='glorot_uniform',activity_regularizer=my_reg),
#        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(16, filtSize,padding='same',activation='relu',kernel_initializer='glorot_uniform',activity_regularizer=my_reg),   
#        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.BatchNormalization(),      
        tf.keras.layers.Conv2DTranspose(16, filtSize,padding='same',activation='relu',kernel_initializer='glorot_uniform',activity_regularizer=my_reg),
#        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.UpSampling2D(2),
        tf.keras.layers.Conv2DTranspose(8, filtSize,padding='same',activation='relu',kernel_initializer='glorot_uniform',activity_regularizer=my_reg),
#        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.BatchNormalization(),
         tf.keras.layers.UpSampling2D(2),
        tf.keras.layers.Conv2DTranspose(4, filtSize,padding='same',activation='relu',kernel_initializer='glorot_uniform',activity_regularizer=my_reg),
#        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Conv2DTranspose(1, filtSize,padding='same',activation='relu',kernel_initializer='glorot_uniform',activity_regularizer=my_reg),
#        tf.keras.layers.Dropout(0.1),
#        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Softmax(),
        tf.keras.layers.Reshape((100,100,1)),
        tf.keras.layers.ZeroPadding2D(((1,0),(1,0)))])
    
    return model