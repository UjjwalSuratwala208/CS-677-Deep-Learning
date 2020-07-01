from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential,Model,load_model
from keras.layers import Dropout, Flatten, Dense,Input
from keras.utils import np_utils, generic_utils, to_categorical
from keras.models import load_model
import os
from keras import backend as K
K.set_image_dim_ordering('th')
import sys
import keras
train =sys.argv[1]
mm=sys.argv[2]
img_width, img_height = 229,229
input_tensor = Input(shape=(229,229,3))
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 2
batch_size = 16
inout_shape=[229,229,3]

train_para = ImageDataGenerator(rescale=1./255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True)

train1 = train_para.flow_from_directory(train,
        target_size=(229,229),
        batch_size=32,
        class_mode='categorical')

add_model = applications.inception_resnet_v2.InceptionResNetV2(weights='imagenet',include_top=False, input_shape= (3,229,229))

for layer in add_model.layers:
    layer.trainable=False

mode = add_model.output
mode = Dropout(0.3)(mode)
mode = Flatten()(mode)
mode = Dense(output_dim = 256, activation ='relu')(mode)
mode = Dropout(0.3)(mode)
mode = Dense(output_dim = 128, activation ='relu')(mode)
mode = Dropout(0.3)(mode)
mode = Dense(output_dim = 64, activation ='relu')(mode)
mode = Dropout(0.3)(mode)
c = Dense(10, activation='softmax')(mode)

model = Model(inputs = add_model.input, outputs = c)
#opt = keras.optimizers.Adagrad(lr=0.1,decay=0)
opt=keras.optimizers.Adagrad(lr=0.01,decay=1e-6)
model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])

def train():
        model.fit_generator(train1,steps_per_epoch=2000,epochs=1)
train()
model.save(mm)
