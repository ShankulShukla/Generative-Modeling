
# Conditional-GAN

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# training parameters
lr = 0.0001
batch_size = 128
epochs = 100

# hyperparameters of C-GAN
inp_dim = 784
y_dimension = 10  # for each label
gen_hidden_dim = 256
disc_hidden_dim = 128
z_dimension = 100  # image noise datapoint

# discriminator network
# additional information of label using y_dim
disc_hidden_W = tf.Variable(
    tf.random_normal(shape=(inp_dim + y_dimension, disc_hidden_dim)) * 2 / np.sqrt(disc_hidden_dim))
disc_hidden_b = tf.Variable(np.zeros(disc_hidden_dim).astype(np.float32))

disc_output_W = tf.Variable(tf.random_normal(shape=(disc_hidden_dim, 1)) * 2 / np.sqrt(1))
disc_output_b = tf.Variable(np.zeros(1).astype(np.float32))

# generator network

gen_hidden_W = tf.Variable(
    tf.random_normal(shape=(z_dimension + y_dimension, gen_hidden_dim)) * 2 / np.sqrt(gen_hidden_dim))
gen_hidden_b = tf.Variable(np.zeros(gen_hidden_dim).astype(np.float32))

gen_output_W = tf.Variable(tf.random_normal(shape=(gen_hidden_dim, inp_dim)) * 2 / np.sqrt(inp_dim))
gen_output_b = tf.Variable(np.zeros(inp_dim).astype(np.float32))


# leaky_relu
def lrelu(X, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * X + f2 * tf.abs(X)


# creating computatioanl graph

# placeholder for input
Z_input = tf.placeholder(tf.float32, shape=[None, z_dimension])
Y_input = tf.placeholder(tf.float32, shape=[None, y_dimension])
X_input = tf.placeholder(tf.float32, shape=[None, inp_dim])


def Discriminator(x, y):
    inputs = tf.concat(axis=1, values=[x, y])
    hidden_layer = lrelu(tf.matmul(inputs, disc_hidden_W) + disc_hidden_b)
    final_layer = tf.matmul(hidden_layer, disc_output_W) + disc_output_b
    disc_output = tf.nn.sigmoid(final_layer)
    return final_layer, disc_output


def Generator(x, y):
    inputs = tf.concat(axis=1, values=[x, y])
    hidden_layer = tf.nn.relu(tf.matmul(inputs, gen_hidden_W) + gen_hidden_b)
    final_layer = tf.matmul(hidden_layer, gen_output_W) + gen_output_b
    gen_output = tf.nn.tanh(final_layer)
    return gen_output


# building the generator network
output_gen = Generator(Z_input, Y_input)

# building the discriminator network
# D(x)
real_logit_disc, real_output_disc = Discriminator(X_input, Y_input)
# D(G(x))
fake_logit_disc, fake_output_disc = Discriminator(output_gen, Y_input)

# C-GAN losses
# discriminator loss
Disc_real_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logit_disc, labels=tf.ones_like(real_logit_disc)))
Disc_fake_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logit_disc, labels=tf.zeros_like(fake_logit_disc)))
Discriminator_loss = Disc_real_loss + Disc_fake_loss

# generator loss
Generator_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logit_disc, labels=tf.ones_like(fake_logit_disc)))

# define train variables

Generator_var = [gen_hidden_W, gen_output_W, gen_hidden_b, gen_output_b]
Discriminator_var = [disc_hidden_W, disc_output_W, disc_hidden_b, disc_output_b]

# defining optimizer
disc_opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5).minimize(Discriminator_loss, var_list=Discriminator_var)
gen_opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5).minimize(Generator_loss, var_list=Generator_var)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

import pandas as pd

df = pd.read_csv('./dataset/mnist_train.csv')

X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

y_onehot = pd.get_dummies(y)

X_train = X.astype('float32') / 255.

n_batches = len(X_train) // batch_size

# training the C-GAN
for e in range(epochs):
    for i in range(n_batches):
        x_batch = X_train[i * batch_size:(i + 1) * batch_size]
        y_batch = y_onehot[i * batch_size:(i + 1) * batch_size]

        # generate sample for discriminator
        Z_sample = np.random.uniform(0., 1., size=[batch_size, z_dimension])
        _, disc_loss_batch = sess.run([disc_opt, Discriminator_loss],
                                      feed_dict={X_input: x_batch, Y_input: y_batch, Z_input: Z_sample})
        _, gen_loss_batch = sess.run([gen_opt, Generator_loss], feed_dict={Y_input: y_batch, Z_input: Z_sample})

        if i % 30 == 0:
            Z_sample = np.random.uniform(0., 1., size=[1, z_dimension])
            y_label = np.zeros(shape=[1, 10])
            from random import randrange

            y_label[:, randrange(10)] = 1
            generated_sample = sess.run(output_gen, feed_dict={Z_input: Z_sample, Y_input: y_label})
            plt.imshow(generated_sample.reshape(28, 28), cmap="gray")
            plt.show()
            print("Epoch-{} Step-{} Generator loss-{} Discriminator loss-{}".format(e, i, gen_loss_batch,
                                                                                    disc_loss_batch))


# testing the model by plotting digit construction
Z_sample = np.random.uniform(0., 1., size=[1, z_dimension])
y_label = np.zeros(shape=[1, 10])
y_label[:, 2] = 1
generated_sample = sess.run(output_gen, feed_dict={Z_input: Z_sample, Y_input: y_label})

plt.imshow(generated_sample.reshape(28, 28), cmap="gray")
plt.imshow(generated_sample.reshape(28, 28), cmap="gray")



