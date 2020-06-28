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

'''
X_train = np.load('Imagenet/x_train.npy')
X_test = np.load('Imagenet/x_test.npy')
Y_train = np.load('Imagenet/y_train.npy')
Y_test = np.load('Imagenet/y_test.npy')
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
#print(X_test)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

print('Y_train shape:', Y_train.shape)
print('Y_test shape:', Y_test.shape)

#Y_train = keras.utils.to_categorical(Y_train, nb_classes)
#Y_test = keras.utils.to_categorical(Y_test, nb_classes)#convert label into one-hot vector
Y_train = to_categorical(Y_train, nb_classes)
Y_test = to_categorical(Y_test, nb_classes)#convert label into one-hot vector

print('Y_train shape:', Y_train.shape)
print('Y_test shape:', Y_test.shape)

#exit()
'''
train=sys.argv[1]
test=sys.argv[2]
mode=sys.argv[3]

X_train = np.load(train)
Y_train = np.load(test)
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
Y_train = to_categorical(Y_train, nb_classes)

model = Sequential()

#Layer 1

#model.add(Conv2D(filters=32, kernel_size=(3,3), padding='valid', input_shape=[3,112,112]))#Convo
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='valid', input_shape=(img_rows,img_cols,img_channels)))
model.add(BatchNormalization())
model.add(Activation('relu'))#Activation function
model.add(MaxPooling2D(pool_size=(2, 2)))
#14x14 output
'''
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='valid'))#Convo$
model.add(Activation('relu'))#Activation function
model.add(AveragePooling2D(pool_size=(3, 3)))
'''

#Layer 2
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='valid'))#Convo$
model.add(BatchNormalization())
model.add(Activation('relu'))#Activation function
model.add(MaxPooling2D(pool_size=(2, 2)))
#6x6
#keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
'''
#Layer 3
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='valid'))#Convo$
model.add(BatchNormalization())
model.add(Activation('relu'))#Activation function
model.add(MaxPooling2D(pool_size=(2, 2)))
'''
#2x2
#Layer 3
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='valid'))#Convo$
model.add(BatchNormalization())
model.add(Activation('relu'))#Activation function
model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(keras.layers.Dropout(0.5))
#Dense layer
model.add(Flatten())# shape equals to [batch_size, 32] 32 is the number of filters
model.add(Dropout(0.5))
model.add(Dense(10))#Fully connected layer
model.add(BatchNormalization())
model.add(Activation('softmax'))
#model.add(keras.layers.Dropout(0.5))

#opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)
#opt = keras.optimizers.SGD(lr=0.01, decay=1e-6)
#opt = keras.optimizers.Adam(lr=0.01, decay=1e-6)
opt = keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=1e-6)
#opt = keras.optimizers.Adamax(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
#model.save('my_model.h5')

def train():
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=nb_epoch,
              shuffle=True)

train()
model.save(mode)

