import tensorflow as tf
import time
import numpy as np
from numpy.random import randn
from numpy.random import randint
from matplotlib import pyplot
from keras.datasets import mnist
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, LeakyReLU, ZeroPadding2D, BatchNormalization,\
    Reshape, Activation, UpSampling2D

tf.config.experimental.set_memory_growth = True						# This prevents a memory access error if the GPU is
																	# being used by other processes
tf.compat.v1.GPUOptions.per_process_gpu_memory_fraction = 0.9

# Loading the data from the MNIST dataset
def load_MNIST_data():
    (X_train, _), (_, _) = mnist.load_data()
    X = X_train.astype('float32')                           # Converting from int to float
    X = (X - 127.5) / 127.5                                 # Scaling from [0,255] to [-1,1]
    X = np.expand_dims(X, axis=3)
    return X

# Choose n real samples from the data
def choose_real_samples(data, n):
    indices = randint(0, data.shape[0], n)                  # Choose n random indices
    X = data[indices]                                       # Select the corresponding samples
    Y = np.ones((n, 1))                                     # Generate "real" class labels
    return X, Y

# Generate random vectors to be fed into the generator
# n - number of samples, dim - dimension of the vector space
def generate_random_vectors(dim, n):
    input = randn(dim * n)
    input = input.reshape(n, dim)
    return input

class DCGAN:
    def __init__(self):
        self.data = load_MNIST_data()
        self.input_shape = (28, 28, 1)                      # Size of images
        self.latent_dim = 50                                # Dimension of random vectors for generator
        self.batch_size = 128                               # Training batch size
        self.epochs = 300                                   # Number of epochs for training the GAN
        self.grid_n = 5                                     # Size of grid for display of generated results
        self.gen_optimizer = Adam(lr=0.0001, beta_1=0.5)    # Generator optimizer
        self.disc_optimizer = Adam(lr=0.0004, beta_1=0.5)   # Discriminator optimizer
        self.generator = self.create_generator()
        self.discriminator = self.create_discriminator()
        self.GAN = self.create_GAN(self.generator, self.discriminator)

    # Use the generator to create n samples from random vectors
    def create_fake_samples(self, n):
        input = generate_random_vectors(self.latent_dim, n)  # Generate the random input to be fed into the generator
        X = self.generator.predict(input)                    # Use the generator
        Y = np.zeros((n, 1))                                 # Generate "fake" class labels
        return X, Y

    # The discriminator receives an image of dimensions (28, 28, 1) and classifies it as real (1) or fake (0)
    def create_discriminator(self):
        model = Sequential()
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same", input_shape=self.input_shape))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=self.disc_optimizer, metrics=['accuracy'])
        return model

    def create_generator(self):
        model = Sequential()
        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((7, 7, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(1, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))
        return model

    # Combine the generator and discriminator into a single model (intended for generator update)
    def create_GAN(self, generator, discriminator):
        discriminator.trainable = False                      # Setting the discriminator to not update weights
        model = Sequential()
        model.add(generator)
        model.add(discriminator)
        model.compile(loss='binary_crossentropy', optimizer=self.gen_optimizer)
        return model

    def evaluate_GAN(self, epoch, samples):
        X_real, Y_real = choose_real_samples(self.data, samples)
        X_fake, Y_fake = self.create_fake_samples(samples)
        _, accuracy_real = self.discriminator.evaluate(X_real, Y_real, verbose=0)
        _, accuracy_fake = self.discriminator.evaluate(X_fake, Y_fake, verbose=0)

        # Create an n*n grid of generated images and saving it
        n = self.grid_n

        for i in range(n ** 2):
            pyplot.subplot(n, n, i + 1)
            pyplot.axis('off')
            pyplot.imshow(X_fake[i])
        filename = 'generated_images_epoch_%d.png' % (epoch + 1)
        pyplot.savefig(filename)

        # Save the current version of the generator
        filename = 'generator_model_%03d.h5' % (epoch + 1)
        self.generator.save(filename)

    # Train the generator and discriminator
    def train(self, verbose=True):
        batches_per_epoch = int(self.data.shape[0] / self.batch_size)
        half_batch = int(self.batch_size / 2)

        for i in range(self.epochs):
            for j in range(batches_per_epoch):

                X_real, Y_real = choose_real_samples(self.data, half_batch)
                discriminator_loss_real, _ = self.discriminator.train_on_batch(X_real, Y_real)
                X_fake, Y_fake = self.create_fake_samples(half_batch)
                discriminator_loss_fake, _ = self.discriminator.train_on_batch(X_fake, Y_fake)

                X_GAN = generate_random_vectors(self.latent_dim, self.batch_size)
                Y_GAN = 0.9*np.ones((self.batch_size, 1))
                generator_loss = self.GAN.train_on_batch(X_GAN, Y_GAN)

                if verbose:
                    print('>%d, %d/%d, disc_loss_real=%.3f, disc_loss_fake=%.3f gen_loss=%.3f' %
                          (i + 1, j + 1, batches_per_epoch, discriminator_loss_real, discriminator_loss_fake,
                           generator_loss))

            if (i + 1) % 20 == 0:
                self.evaluate_GAN(i, samples=100)

if __name__ == '__main__':
    gan = DCGAN()
    time0 = time.time()
    gan.train(verbose=True)
    time1 = time.time()
    print('The training took %.3f minutes') % ((time1 - time0)/60)
