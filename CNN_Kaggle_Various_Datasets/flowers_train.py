import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import math
from sklearn.metrics import classification_report, confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
import matplotlib.pyplot as plt
#%matplotlib inline
from keras.models import Model
#from keras.engine.input_layer import Input
from keras.applications.vgg16 import VGG16
import itertools  
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import sys
train11=sys.argv[1]

target_names = ['daisy','dandelion','rose','sunflower','tulips']
image_height = 150
image_width  = 150
batch_size = 32
train_datagen = ImageDataGenerator(
        rescale=1./255.,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split = 0.2)

train_generator = train_datagen.flow_from_directory(
        train11,
        target_size=(image_height, image_width),
        color_mode="rgb",
        batch_size=batch_size,
        class_mode='categorical',
        subset = 'training',
        shuffle=False)
valid_generator = train_datagen.flow_from_directory(
        train11,
        target_size=(image_height, image_width),
        color_mode="rgb",
        batch_size=batch_size,
        subset = 'validation',
        class_mode='categorical',
        shuffle=False)
def get_labels(gen):
    labels = []
    sample_no = len(gen.filenames)
    call_no = int(math.ceil(sample_no / batch_size))
    for i in range(call_no):
        labels.extend(np.array(gen[i][1]))
    
    return np.array(labels)

#input_tensor = Input(shape=(150,150,3))
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150,150,3))
train_data = np.array(base_model.predict_generator(train_generator))
train_labels = get_labels(train_generator)
valid_data = np.array(base_model.predict_generator(valid_generator))
valid_labels = get_labels(valid_generator)
model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_data, train_labels,
          epochs=25,
          batch_size=32,
          validation_data=(valid_data, valid_labels),
          verbose = 0)
mode=sys.argv[2]
model.save(mode)
