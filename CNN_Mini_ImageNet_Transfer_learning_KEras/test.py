from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential,Model,load_model
from keras.layers import Dropout, Flatten, Dense,Input
from keras.utils import np_utils, generic_utils, to_categorical
from keras.models import load_model
from keras.utils import np_utils
from keras.utils import Sequence
import os
from PIL import Image
from PIL import ImageFilter
import numpy as np
import pandas as pd
import os
from keras.layers import Input
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
K.set_image_dim_ordering('th')
import sys
import keras
val =sys.argv[1]
mode=sys.argv[2]
mg_width, img_height = 229,229
input_tensor = Input(shape=(229,229,3))
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 2
batch_size = 16
inout_shape=[229,229,3]

test_para = ImageDataGenerator(rescale=1./255)

test1 = test_para.flow_from_directory(val,
        target_size=(229,229),
        batch_size=32,
        class_mode='categorical')

model = load_model(mode)

def test():
    scores = model.evaluate_generator(test1)
    print("Accuracy = ", scores[1])
    print("test error",1-scores[1])

test()

