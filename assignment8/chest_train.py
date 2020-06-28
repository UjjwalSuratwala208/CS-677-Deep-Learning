import os
import os.path
from PIL import Image
from PIL import ImageFilter
import numpy as np
import keras
import pandas as pd
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
import sys
from keras.layers import Input
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras import applications
from keras.models import Sequential,Model,load_model
train11 = sys.argv[1]
#val=sys.argv[3]
mode=sys.argv[2]
train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
training_set = train_datagen.flow_from_directory(train11,target_size=(224,224),batch_size=32,class_mode='categorical')
#val_datagen = ImageDataGenerator(rescale=1./255)
#val_set = val_datagen.flow_from_directory(val,target_size=(224,224),batch_size=32,class_mode='categorical')
t_model = applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape= (224,224,3))

for layer in t_model.layers:
    layer.trainable=False

x = t_model.output
x = Flatten()(x)

c = Dense(2, activation='softmax')(x)
model = Model(inputs = t_model.input, outputs = c)
opt = keras.optimizers.Adam(lr=0.001, decay=1e-7)
#opt = keras.optimizers.Adagrad(lr=0.01, decay=1e-7
model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])

def train():
        model.fit_generator(training_set,steps_per_epoch=4000,epochs=3)

train()

model.save(mode)
