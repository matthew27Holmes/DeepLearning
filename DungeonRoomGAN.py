import matplotlib.pyplot as plt
import os, time
import numpy as np

import tensorflow as tf
from tensorflow import keras

from keras.backend.tensorflow_backend import set_session
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

from keras.models import load_model

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose,UpSampling1D
from keras.models import Sequential, Model

from keras.optimizers import Adam


class GAN():
    def __init__(self):
        #shape
        self.Height = 64
        self.Width = 64
        self.Channel = 3
        self.img_shape = (self.Width, self.Height, self.Channel)
        self.latent_dim = 100

        #set up images 
        self.imagesDir = "OGZelda/Rooms(496)"
        self.n_sample = 4

        self.image_gen = ImageDataGenerator(horizontal_flip=True,
                               vertical_flip=True,
                               fill_mode='wrap')

        # optimizer
        optimizer = Adam(0.00009, 0.6)

        # Build and compile the discriminator
        self.discriminator = self.Discriminator()# load_model('model/newDungeon/Discriminator.h5') #
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.Generator()#load_model('model/newDungeon/Generator.h5')#


        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)#load_model('model/newDungeon/TestModels/Save_Combined_Test_Combined.h5')#
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
        
    def Generator(self):
        # latent variable as input
        model = Sequential()
        #model.add(Dense(1024, activation="relu", input_dim= self.latent_dim))
        model.add(Dense(1024, activation="relu", input_dim= self.latent_dim))
        model.add(Dense(128 * 8 * 8, activation="relu"))
        model.add(Reshape((8, 8, 128)))
        model.add(UpSampling2D())

        model.add(Conv2DTranspose(128, kernel_size=5, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())

        model.add(Conv2D(64, kernel_size=5, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())


        model.add(Conv2DTranspose(64, kernel_size=5, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(Conv2D(32, kernel_size=5, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))


        model.add(Conv2DTranspose(32, kernel_size=5, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(Conv2D(28, kernel_size=5, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(Conv2D(self.Channel, kernel_size=5, padding="same"))
        model.add(Activation("tanh"))
        
        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def Discriminator(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))

        model.add(Conv2D(64, kernel_size=5, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))

        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))
        # x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))

        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))
        #x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        '''model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.2))'''

        #x = layers.MaxPooling2D((2, 2), strides=(1, 1), name='block4_pool')(x)

        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        OutModel = Model(img, validity)

        return OutModel


    def train(self, epochs, batch_size=128,save_interval=50):

        # get save Directory 
        dir_result = "./DungeonRoomSavedImages/"
        try:
            os.mkdir(dir_result)
        except:
            pass

        # Load the dataset
        X_train = self.get_image_data()

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        #epoch = 0
        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]


            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
            

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.generator.save('model/newDungeon/Test_Generator.h5')
                self.discriminator.save('model/newDungeon/Test_Discriminator.h5')

                self.plot_generated(path_save=dir_result + "/image_{:05.0f}.png".format(epoch),
                    title_add="Epoch %d [D loss: %f, acc.: %.2f%%] [G loss: %f]" 
                    % (epoch, d_loss[0], 100*d_loss[1], g_loss))
        
    
    def get_image_data(self):
        x_train = []
        trainNumber = 200000
        #testNumber = 100

        images = np.sort(os.listdir(self.imagesDir))
        
        trainingImages = images[:trainNumber]
        #testingImages = self.images[trainNumber:trainNumber + testNumber]

        for i, myid in enumerate(trainingImages):
            image = load_img(self.imagesDir + "/" + myid,
                             target_size=self.img_shape[:2])
            
            image = img_to_array(image)/255.0
            
            aug_interator = self.image_gen.flow(np.expand_dims(image,0)) #,batch_size=5,shuffle=True,save_to_dir='./OGZelda/Rooms(496)', save_prefix='aug_image', save_format='jpg' 
            aug_images = [next(aug_interator)[0].astype(np.uint8) for i in range(3)]  #

            #for aug_image in aug_images: 
            #   x_train.append(aug_image)
            x_train.append(image)
                       
        x_train = np.array(x_train)
        

        np.random.shuffle(x_train)
        return x_train



    def plot_generated(self,path_save=None, title_add=""):

        noise = np.random.normal(0, 1, (self.n_sample, self.latent_dim))
        imgs = self.generator.predict(noise)
        fig = plt.figure(figsize=(40, 10))
        for i, img in enumerate(imgs):
            ax = fig.add_subplot(1, self.n_sample, i + 1)
            ax.imshow(img)
        fig.suptitle("Generated images " + title_add, fontsize=30)

        if path_save is not None:
            plt.savefig(path_save,
                        bbox_inches='tight',
                        pad_inches=0)
            plt.close()
        else:
            plt.show()


if __name__ == '__main__':

    gan = GAN()

    start_time = time.time()
    gan.train(epochs=100000, batch_size=32, save_interval=50)
    end_time = time.time()

    print("-" * 10)
    print("Time took: {:4.2f} min".format((end_time - start_time) / 60))


