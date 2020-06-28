from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras.backend.tensorflow_backend as tfback
from keras.models import load_model
import sys
from keras.preprocessing.image import ImageDataGenerator
model_file=sys.argv[1]
output_file=sys.argv[2]
generator=load_model(model_file)

r, c = 5, 5
latent_dim=100
noise = np.random.normal(0, 1, (r * c, latent_dim))
gen_imgs = generator.predict(noise)
gen_imgs = 0.5 * gen_imgs + 0.5
fig, axs = plt.subplots(r, c)
cnt = 0
for i in range(r):
	for j in range(c):
		axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
		axs[i, j].axis('off')
		cnt += 1
		#print('ch2')

fig.savefig(output_file)
plt.close()