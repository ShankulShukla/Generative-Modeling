
# Generating images of handwritten digits using a Deep Convolutional Generative Adversarial Network

import numpy as np
import tensorflow as tf
from tensorflow.layers import batch_normalization
from tensorflow.keras.layers import UpSampling2D

import matplotlib.pyplot as plt


class DCGAN:
    def __init__(self, z_shape=100, img_shape=(28, 28), channels=1, learning_rate=0.0001):

        # input characteristics
        self.channels = channels
        self.z_shape = z_shape
        self.img_rows, self.img_cols = img_shape

        # defining Initializing discriminator weights and network
        with tf.variable_scope('d'):
            self.disc_W1 = tf.Variable(tf.random_normal(shape=[5, 5, channels, 64]) * 2 / np.sqrt(64))
            self.disc_b1 = tf.Variable(tf.zeros([64]))
            self.disc_W2 = tf.Variable(tf.random_normal(shape=[3, 3, 64, 64]) * 2 / np.sqrt(64))
            self.disc_b2 = tf.Variable(tf.zeros([64]))
            self.disc_W3 = tf.Variable(tf.random_normal(shape=[3, 3, 64, 128]) * 2 / np.sqrt(128))
            self.disc_b3 = tf.Variable(tf.zeros([128]))
            self.disc_W4 = tf.Variable(tf.random_normal(shape=[2, 2, 128, 256]) * 2 / np.sqrt(256))
            self.disc_b4 = tf.Variable(tf.zeros([256]))
            self.disc_W5 = tf.Variable(tf.random_normal(shape=[7 * 7 * 256, 1]) * 2 / np.sqrt(1))
            self.disc_b5 = tf.Variable(tf.zeros([1]))

        # defining Initializing generator weights and network
        with tf.variable_scope('g'):
            self.gen_W1 = tf.Variable(tf.random_normal(shape=[100, 7 * 7 * 512]) * 2 / np.sqrt(7 * 7 * 512))
            self.gen_W2 = tf.Variable(tf.random_normal(shape=[3, 3, 512, 256]) * 2 / np.sqrt(256))
            self.gen_W3 = tf.Variable(tf.random_normal(shape=[3, 3, 256, 128]) * 2 / np.sqrt(128))
            self.gen_W4 = tf.Variable(tf.random_normal(shape=[3, 3, 128, 1]) * 2 / np.sqrt(1))

        # placeholder for inputs
        self.X = tf.placeholder(tf.float32, [None, self.img_rows, self.img_cols])
        self.Z = tf.placeholder(tf.float32, [None, self.z_shape])

        # generated output
        self.output_gen = self.gen_forward(self.Z)

        disc_logits_fake = self.disc_forward(self.output_gen)
        disc_logits_real = self.disc_forward(self.X)

        # defining gan costs
        disc_fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(disc_logits_fake), logits=disc_logits_fake))
        disc_real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(disc_logits_real), logits=disc_logits_real))

        self.disc_loss = tf.add(disc_fake_loss, disc_real_loss)
        self.gen_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(disc_logits_fake), logits=disc_logits_fake))

        # learned parameters
        train_vars = tf.trainable_variables()
        disc_vars = [var for var in train_vars if 'd' in var.name]
        gen_vars = [var for var in train_vars if 'g' in var.name]

        # optimizing network parameters
        self.disc_opt = tf.train.AdamOptimizer(learning_rate).minimize(self.disc_loss, var_list=disc_vars)
        self.gen_opt = tf.train.AdamOptimizer(learning_rate).minimize(self.gen_loss, var_list=gen_vars)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    # Discriminator feed forward
    def disc_forward(self, X):
        X = tf.reshape(X, [-1, self.img_rows, self.img_cols, self.channels])
        # layer 1
        z = tf.nn.conv2d(X, self.disc_W1, [1, 2, 2, 1], padding="SAME")
        z = tf.nn.bias_add(z, self.disc_b1)
        z = tf.nn.leaky_relu(z)
        # layer 2
        z = tf.nn.conv2d(z, self.disc_W2, [1, 1, 1, 1], padding="SAME")
        z = tf.nn.bias_add(z, self.disc_b2)
        z = batch_normalization(z)
        z = tf.nn.leaky_relu(z)
        # layer 3
        z = tf.nn.conv2d(z, self.disc_W3, [1, 2, 2, 1], padding="SAME")
        z = tf.nn.bias_add(z, self.disc_b3)
        z = batch_normalization(z)
        z = tf.nn.leaky_relu(z)
        # layer 4
        z = tf.nn.conv2d(z, self.disc_W4, [1, 1, 1, 1], padding="SAME")
        z = tf.nn.bias_add(z, self.disc_b4)
        z = batch_normalization(z)
        z = tf.nn.leaky_relu(z)
        # layer 5
        z = tf.reshape(z, [-1, 7 * 7 * 256])
        logits = tf.matmul(z, self.disc_W5)
        logits = tf.nn.bias_add(logits, self.disc_b5)
        return logits

    # Generator feed forward
    def gen_forward(self, X):
        # layer 1
        z = tf.matmul(X, self.gen_W1)
        z = tf.nn.relu(z)
        z = tf.reshape(z, [-1, 7, 7, 512])
        # layer 2
        z = UpSampling2D()(z)
        z = tf.nn.conv2d(z, self.gen_W2, [1, 1, 1, 1], padding="SAME")
        z = batch_normalization(z)
        z = tf.nn.leaky_relu(z)
        # layer 3
        z = UpSampling2D()(z)
        z = tf.nn.conv2d(z, self.gen_W3, [1, 1, 1, 1], padding="SAME")
        z = batch_normalization(z)
        z = tf.nn.leaky_relu(z)

        z = tf.nn.conv2d(z, self.gen_W4, [1, 1, 1, 1], padding="SAME")

        return tf.nn.tanh(z)

    # generate sample from generator
    def generate_sample(self, epoch, batch_size):
        z = np.random.uniform(-1, 1, (batch_size, self.z_shape))
        imgs = self.sess.run(self.output_gen, feed_dict={self.Z: z})
        imgs = imgs * 0.5 + 0.5
        fig, axs = plt.subplots(5, 5)
        cnt = 0
        for i in range(5):
            for j in range(5):
                axs[i, j].imshow(imgs[cnt, :, :, 0], cmap="gray")
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("samples/%d.png" % epoch)
        plt.close()

    def train(self, X_train, batch_size=128, epoch=15):
        n_batches = len(X_train) // batch_size
        for e in range(epoch):
            for i in range(n_batches):
                x_batch = X_train[i * batch_size:(i + 1) * batch_size]

                Z = np.random.uniform(-1, 1, (batch_size, self.z_shape))
                _, d_loss = self.sess.run([self.disc_opt, self.disc_loss], feed_dict={self.X: x_batch, self.Z: Z})

                Z = np.random.uniform(-1, 1, (batch_size, self.z_shape))
                _, g_loss = self.sess.run([self.gen_opt, self.gen_loss], feed_dict={self.Z: Z})
                if i % 20 == 0:
                    self.generate_sample(i, batch_size)
                    print(f"Epoch: {i}. Discriminator loss: {d_loss}. Generator loss: {g_loss}")


import pandas as pd

# processing the dataset
df1 = pd.read_csv('./dataset/mnist_train.csv')

df2 = pd.read_csv('./dataset/mnist_test.csv')

X1 = df1.iloc[:, 1:].values
X2 = df2.iloc[:, 1:].values

X = np.concatenate([X1, X2])
X = X.reshape(-1, 28, 28)

# normalize between -1 and 1
X = X / 127.5 - 1

# creating and training the GAN
gan = DCGAN()

gan.train(X)

