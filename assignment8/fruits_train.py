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
mode = (sys.argv[2])

train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
training_set = train_datagen.flow_from_directory(train11,target_size=(100,100),batch_size=32,class_mode='categorical')

t_model = applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape= (100,100,3))

for layer in t_model.layers:
    layer.trainable=False

x = t_model.output
x = Flatten()(x)
x = Dense(output_dim = 256, activation ='relu')(x)
x = Dropout(0.3)(x)
c = Dense(101, activation='softmax')(x)

model = Model(inputs = t_model.input, outputs = c)

opt = keras.optimizers.Adagrad(lr=0.01,decay=1e-7)
model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])

def train():
        model.fit_generator(training_set,steps_per_epoch=7000,epochs=3,validation_steps=700)

train()
filepath = (sys.argv[2])
model.save(filepath)
