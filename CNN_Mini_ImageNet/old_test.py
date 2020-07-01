from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.utils import np_utils, generic_utils, to_categorical
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
import sys
import numpy as np
import keras

batch_size = 64
nb_classes = 10
nb_epoch = 100

img_channels = 3
img_rows = 112
img_cols = 112

#X_train = np.load('train_image.npy')
#X_test = np.load('test_image.npy')
#Y_train = np.load('train_label.npy')
#Y_test = np.load('test_label.npy')
train=sys.argv[1]
test=sys.argv[2]
mode=sys.argv[3]

X_train = np.load(train)
Y_train = np.load(test)
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
Y_train = to_categorical(Y_train, nb_classes)
model = load_model(mode)

def test():
    accuracy  = model.evaluate(X_train,Y_train)
    print(model.metrics_names[1], accuracy[1]*100)

test() 

'''
def train():
    model.fit(X_train,Y_train,
              batch_size=batch_size,
              epochs=nb_epoch,
              shuffle=True)

train()
'''



