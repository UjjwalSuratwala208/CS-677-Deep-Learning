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
val = sys.argv[1]
mode=sys.argv[2]
val_datagen = ImageDataGenerator(rescale=1./255)
val_set = val_datagen.flow_from_directory(val,target_size=(100,100),batch_size=32,class_mode='categorical')
#model.load(mode)
model = load_model(mode)
def test():
    scores = model.evaluate_generator(val_set)
    print("Accuracy = ", scores[1])
    print("test error",1-scores[1])

test()

