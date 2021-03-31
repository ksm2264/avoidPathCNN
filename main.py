#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 13:34:05 2021

@author: karl
"""
from keras.utils import Sequence
import numpy as np   
import scipy.io as sio
import glob
import cv2
from PIL import Image

#def getEvens(matList):

def getOdds(fileList):

    oddsList = []
    for filename in fileList:
        
        if int(filename.split('/')[-1].split('_')[1])%2==1:
            oddsList.append(filename)
            
    
    return oddsList


def getEvens(fileList):

    evensList = []
    for filename in fileList:
        
        if int(filename.split('/')[-1].split('_')[1])%2==0:
            evensList.append(filename)
            
    
    return evensList
        
trainListCurved = glob.glob('../turns_cnn/maps/curved/J*.mat')
trainListStraight=glob.glob('../turns_cnn/maps/straight/J*.mat')

trainListCurved = getOdds(trainListCurved)
trainListStraight = getOdds(trainListStraight)

trainList = trainListCurved+trainListStraight
trainClasses =np.concatenate((np.ones(len(trainListCurved)),np.zeros(len(trainListStraight))),axis=0)

#trainList = 4*trainList
#trainClasses = np.array(4*list(trainClasses))

shuffIdx = np.arange(0,len(trainList))
np.random.shuffle(shuffIdx)

trainList = [trainList[idx] for idx in list(shuffIdx)]
trainClasses = [trainClasses[idx] for idx in list(shuffIdx)]

testListCurved = glob.glob('../turns_cnn/maps/curved/J*.mat')
testListStraight=glob.glob('../turns_cnn/maps/straight/J*.mat')

testListCurved = getEvens(testListCurved)
testListStraight = getEvens(testListStraight)

testList = testListCurved+testListStraight
testClasses =np.concatenate((np.ones(len(testListCurved)),np.zeros(len(testListStraight))),axis=0)

def my_readfunction(filename):
    
    mat = sio.loadmat(filename,squeeze_me=True,struct_as_record=False)['depth_map']
    mat = cv2.resize(mat,(int(mat.shape[1]/2),int(mat.shape[0]/2)))
    mat = Image.fromarray(mat)
#    mat=mat.rotate(np.random.rand()*360)
    mat = np.array(mat)
#    mat = np.abs(mat)
    mat = mat[:,:,None]
    
    return mat

class Mygenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        # read your data here using the batch lists, batch_x and batch_y
        x = [my_readfunction(filename) for filename in batch_x] 
        y = batch_y
        return np.array(x), np.array(y)

def getInputStack(fileList):
    
    x = np.stack([my_readfunction(filename) for filename in fileList])
    
    return x

train_gen = Mygenerator(trainList,trainClasses,20)
#%%
from nn_utils import getCNN
import tensorflow as tf
from sklearn.utils import class_weight

model = getCNN()

#my_opt = tf.keras.optimizers.Nadam(learning_rate=1e-3)
opt= tf.keras.optimizers.Adadelta(lr=0.1, rho=0.95, epsilon=1e-08, decay=0.0)

model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])

reduceLR_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=5,min_lr = 1e-16)
ES_cb = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', 
                                      min_delta=0, patience=20, verbose=0, mode='auto', baseline=None, restore_best_weights=True)

weights = class_weight.compute_class_weight('balanced',np.unique(trainClasses),trainClasses)

weights = {i : weights[i] for i in range(2)}
#
x_in = getInputStack(trainList)
y_in = np.array(trainClasses)

batch_size = 30
s_p_e = len(x_in)/batch_size

history = model.fit(x_in,y_in,validation_split=0.1,batch_size=batch_size,validation_steps=1,steps_per_epoch=s_p_e,epochs=100,
                    verbose=1,callbacks=[reduceLR_cb, ES_cb],class_weight=weights)

test_gen = Mygenerator(testList,testClasses,1)

x_test = getInputStack(testList)
y_test = np.array(testClasses)

results=model.evaluate(x_test,y_test)

print(results)