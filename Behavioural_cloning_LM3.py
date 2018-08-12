# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 10:28:44 2018

@author: menglin
"""
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np
import cv2
import csv
import os 
from itertools import islice


#%%  read images and measurements
csvFile= open(r'D:\Jupyter\Udacity\CarND-Behavioral-Cloning-P3\data\data\driving_log.csv')
reader=csv.reader(csvFile)
samples=[]
for line in islice(reader,1,None):
    samples.append(line)
#%% use generator
import sklearn  

def generator(samples, batch_size=64):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples=sklearn.utils.shuffle(samples)
        correction=0.2
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            for batch_sample in batch_samples:
                steering_center=float(batch_sample[3])
                steering_left=steering_center+correction
                steering_right=steering_center-correction
                image_center=cv2.imread("data/data/IMG/"+batch_sample[0].split('/')[-1])
                images.append(image_center)
                measurements.append(steering_center)
                image_left=cv2.imread('data/data/IMG/'+batch_sample[1].split('/')[-1])
                images.append(image_left)
                measurements.append(steering_left)
                image_right=cv2.imread('data/data/IMG/'+batch_sample[2].split('/')[-1])
                images.append(image_right)
                measurements.append(steering_right)
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)
#%%
#plt.imshow(image)
    
#%% buid the moldel
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda,Conv2D,Cropping2D,MaxPooling2D
# from keras.layers.pooling import MaxPooling2D

'''possible change in keras 
   keras.layers.convolutional.Cropping2D
   keras.layers.core.Lambda
'''
def NvidiaModel():
    model=Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    model.add(Conv2D(24, (5, 5), activation="relu", strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), activation="relu", strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation="relu", strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    model.summary()
    return model
    
#%%
from sklearn.model_selection import train_test_split
model = NvidiaModel()
model.compile(loss='mse',optimizer='adam')
train_samples, validation_samples =train_test_split(samples, test_size=0.2)
train_generator = generator(train_samples, batch_size=64)
validation_generator = generator(validation_samples, batch_size=64)
history_object = model.fit_generator(train_generator, steps_per_epoch=3*len(train_samples)/64, validation_data=validation_generator,
                                     nb_val_samples=3*len(validation_samples)/64, nb_epoch=3, verbose=1)
model.save('models/nVidea_5data.h5')
print("model saved at:models/nVidea_data.h5")

with open('log_hist.txt','w') as f:
    f.write(str(history_object.history))
#%%

from keras.utils import plot_model
plot_model(model, to_file='model.png')  


#%% Outputting Training and Validation Loss Metrics

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

    
    



#%% 