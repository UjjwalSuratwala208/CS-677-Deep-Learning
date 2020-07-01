import os
import os.path
from PIL import Image
from PIL import ImageFilter
import numpy as np
import keras
import pandas as pd
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.models import Sequential
from keras.layers import Input
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing.image import ImageDataGenerator
from keras import applications
from keras.models import Sequential,Model,load_model
from keras.applications.vgg16 import VGG16
import math
import sys

train11 = sys.argv[1]
mode = sys.argv[2]

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
# Function to get labels from generators to separate them
def get_labels(gen):
    labels = []
    sample_no = len(gen.filenames)
    call_no = int(math.ceil(sample_no / batch_size))
    for i in range(call_no):
        labels.extend(np.array(gen[i][1]))
    
    return np.array(labels)

base_model = VGG16(weights='imagenet', include_top=False,input_shape = (150,150,3))

#train_data = np.array(base_model.predict_generator(train_generator))
#train_labels = get_labels(train_generator)
valid_data = np.array(base_model.predict_generator(valid_generator))
valid_labels = get_labels(valid_generator)

model = load_model(mode)


def test():
	mm = model.evaluate(valid_data,valid_labels)
	print('Accuracy:', round(mm[1],2))
	print('Test Error:', round(1-mm[1],2))

test()
