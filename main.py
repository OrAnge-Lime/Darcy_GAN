from keras import Sequential
from keras.initializers import TruncatedNormal
from keras.layers import Activation, BatchNormalization, Conv2D, Conv2DTranspose, Dense, Flatten, LeakyReLU, Reshape
from keras.optimizers import Adam
import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image


WIDH, HEIGHT, CHANELS = 64, 64, 3
BATCH_SIZE = 128
NOISE_DIM = 100
EPOCHS = 10000
DATASET_DIR = "/content/ds"
OUTPUT = "/content/"

import os
print(len(os.listdir("/content/ds")))

class DcGan:
    def __init__(self):
        self.build_dc_gan()

    def build_generator_model(self):
        self.generator_model = Sequential()

        self.generator_model.add(Dense(8 * 8 * 512, input_dim=100,
                                       kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.02)))
        self.generator_model.add(BatchNormalization(momentum=0.5))
        self.generator_model.add(LeakyReLU(alpha=0.2))
        self.generator_model.add(Reshape((8, 8, 512)))

        self.generator_model.add(Conv2DTranspose(256, 3, strides=2, padding='same',
                                                 kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.02)))
        self.generator_model.add(BatchNormalization(momentum=0.5))
        self.generator_model.add(LeakyReLU(alpha=0.2))

        self.generator_model.add(Conv2DTranspose(128, 3, strides=2, padding='same',
                                                 kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.02)))
        self.generator_model.add(BatchNormalization(momentum=0.5))
        self.generator_model.add(LeakyReLU(alpha=0.2))

        self.generator_model.add(Conv2DTranspose(64, 3, strides=2, padding='same',
                                                 kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.02)))
        self.generator_model.add(BatchNormalization(momentum=0.5))
        self.generator_model.add(LeakyReLU(alpha=0.2))

        self.generator_model.add(Conv2D(3, 3, padding='same',
                                        kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.02)))
        self.generator_model.add(Activation('tanh'))

        self.generator_model.summary()
        tf.keras.utils.plot_model(self.generator_model, to_file = "generator.png")

        return self.generator_model

    def build_discriminator_model(self):
        self.discriminator_model = Sequential()

        self.discriminator_model.add(Conv2D(128, 3, strides=2, input_shape=(WIDH, HEIGHT, CHANELS), padding='same',
                                            kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.02)))
        self.discriminator_model.add(LeakyReLU(alpha=0.2))

        self.discriminator_model.add(Conv2D(256, 3, strides=2, padding='same',
                                            kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.02)))
        self.generator_model.add(BatchNormalization(momentum=0.5))
        self.discriminator_model.add(LeakyReLU(alpha=0.2))

        self.discriminator_model.add(Conv2D(512, 3, strides=2, padding='same',
                                            kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.02)))
        self.generator_model.add(BatchNormalization(momentum=0.5))
        self.discriminator_model.add(LeakyReLU(alpha=0.2))

        self.discriminator_model.add(Conv2D(1024, 3, strides=2, padding='same',
                                            kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.02)))
        self.generator_model.add(BatchNormalization(momentum=0.5))
        self.discriminator_model.add(LeakyReLU(alpha=0.2))

        self.discriminator_model.add(Flatten())
        self.discriminator_model.add(Dense(1, kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.02)))
        self.generator_model.add(BatchNormalization(momentum=0.5))
        self.discriminator_model.add(Activation('sigmoid'))

        self.discriminator_model.summary()
        tf.keras.utils.plot_model(self.discriminator_model, to_file = "discriminator.png")

        return self.discriminator_model

    def build_concatenated_model(self):
        self.concatenated_model = Sequential()
        self.concatenated_model.add(self.generator_model)
        self.concatenated_model.add(self.discriminator_model)

        self.concatenated_model.summary()

        return self.concatenated_model

    def build_dc_gan(self):
        self.build_generator_model()
        self.build_discriminator_model()
        self.build_concatenated_model()

        self.discriminator_model.trainable = True
        optimizer = Adam(learning_rate=0.0001, beta_1=0.5)
        self.discriminator_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self.discriminator_model.trainable = False 
        
        optimizer = Adam(learning_rate=0.0001, beta_1=0.5)
        self.concatenated_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self.discriminator_model.trainable = True
    
    def generate_images(self, output_folder_path, iteration_no):
        filepath = os.path.join(output_folder_path, f'Image{iteration_no}.png')
        noise = np.random.uniform(-1, 1, size=[16, 100])

        images = (self.generator_model.predict(noise) + 1) / 2

        plt.figure(figsize=(10, 10))
        for i in range(16):
            plt.subplot(4, 4, i + 1)
            image = images[i, :, :, :]
            image = np.reshape(image, [WIDH, HEIGHT, CHANELS])
            plt.imshow(image)
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(filepath)
        plt.close('all')
    
    def save_models(self, output_folder_path, iteration_no):
        self.generator_model.save(
            os.path.join(output_folder_path, 'generator_{0}.h5'.format(iteration_no)))
        self.discriminator_model.save(
            os.path.join(output_folder_path, 'discriminator_{0}.h5'.format(iteration_no)))
        self.concatenated_model.save(
            os.path.join(output_folder_path, 'concatenated_{0}.h5'.format(iteration_no)))

    # TRAINING
    def train(self, x_train, epochs=EPOCHS, batch_size=BATCH_SIZE, output = "/content", save_interval=100):
              
        for i in range(epochs):
            # Get real (Database) Images
            images_real = x_train[np.random.randint(0, x_train.shape[0], size=batch_size), :, :, :]

            # Generate Fake Images
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            images_fake = self.generator_model.predict(noise)

            # Train discriminator 
            x = np.concatenate((images_real, images_fake), axis=0)
            y = np.ones([2 * batch_size, 1])
            y[batch_size:, :] = 0
            d_loss = self.discriminator_model.train_on_batch(x, y)

            # Train concatenated model
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            y = np.ones([batch_size, 1])
            g_loss = self.concatenated_model.train_on_batch(noise, y)

            # Save Models, generate sample images
            if (i + 1) % save_interval == 0:
                self.save_models(output, i + 1)
                self.generate_images(output, i + 1)

# # DATA PREPROCESSING
img_list = os.listdir(DATASET_DIR)
x_train = []
for img, index in list(zip(img_list, range(len(img_list)))):
    img = DATASET_DIR + "/" + img
    img = np.array(Image.open(img).resize((WIDH, HEIGHT)))
    x_train.append(img)
        
# Standartization
x_train = np.array(x_train)
x_train = (x_train.astype('float32') * 2 / 255) - 1
print(x_train.shape)
np.random.shuffle(x_train)

model = DcGan()
model.train(x_train)