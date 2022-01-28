
# Variational Autoencoders (variational inference + autoencoders)

import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# class defining variational autoencoder
class VariationalAutoencoder:
    def __init__(self, inp_feature, hidden_feature, output_feature, encoding_size):

        # represents a batch of training data
        self.X = tf.placeholder(tf.float32, shape=(None, inp_feature))

        # defining encoder layers
        # encoder layer # 1
        self.encoder_W1 = tf.Variable(
            tf.random_normal(shape=(inp_feature, hidden_feature)) * 2 / np.sqrt(hidden_feature))
        self.encoder_b1 = tf.Variable(np.zeros(hidden_feature).astype(np.float32))

        # encoder layer # 2
        self.encoder_W2 = tf.Variable(
            tf.random_normal(shape=(hidden_feature, output_feature)) * 2 / np.sqrt(output_feature))
        self.encoder_b2 = tf.Variable(np.zeros(output_feature).astype(np.float32))

        # we need 2 times as many units for means and variances so 2*encoding_size
        self.encoded_space_w = tf.Variable(
            tf.random_normal(shape=(output_feature, 2 * encoding_size)) * 2 / np.sqrt(2 * encoding_size))
        self.encoded_space_b = tf.Variable(np.zeros(2 * encoding_size).astype(np.float32))

        # defining the encoder forward propagation
        Z1 = tf.nn.relu(tf.matmul(self.X, self.encoder_W1) + self.encoder_b1)
        Z2 = tf.nn.relu(tf.matmul(Z1, self.encoder_W2) + self.encoder_b2)

        # the encoder's final layer output is unbounded so there is no activation function
        Z3 = tf.matmul(Z2, self.encoded_space_w) + self.encoded_space_b

        # getting the mean and variance / std dev of Z (latent space).
        # standard dev must be > 0, we can get this by passing Z(std dev) through the softplus function and adding a small amount for smoothing.
        self.means = Z3[:, :encoding_size]
        self.stddev = tf.nn.softplus(Z3[:, encoding_size:]) + 1e-6

        # get a sample of Z
        # using parameterization trick in order for the errors to be backpropagated past this point
        standard_normal = tf.contrib.distributions.Normal(
            loc=np.zeros(encoding_size, dtype=np.float32),
            scale=np.ones(encoding_size, dtype=np.float32)
        )
        e = standard_normal.sample(tf.shape(self.means)[0])
        self.Z = e * self.stddev + self.means

        # defining decoder layers
        # decoder layer # 1
        self.decoder_W1 = tf.Variable(
            tf.random_normal(shape=(encoding_size, output_feature)) * 2 / np.sqrt(output_feature))
        self.decoder_b1 = tf.Variable(np.zeros(output_feature).astype(np.float32))

        # decoder layer # 2
        self.decoder_W2 = tf.Variable(
            tf.random_normal(shape=(output_feature, hidden_feature)) * 2 / np.sqrt(hidden_feature))
        self.decoder_b2 = tf.Variable(np.zeros(hidden_feature).astype(np.float32))

        # the decoder's final layer should technically go through a sigmoid
        # so that the final output is a binary probability (e.g. Bernoulli)
        # but Bernoulli accepts logits (pre-sigmoid) so we will take those
        # so no activation function is needed at the final layer
        self.decoded_space_w = tf.Variable(
            tf.random_normal(shape=(hidden_feature, inp_feature)) * 2 / np.sqrt(inp_feature))
        self.decoded_space_b = tf.Variable(np.zeros(inp_feature).astype(np.float32))

        # defining the decoder forward propagation
        Z4 = tf.nn.relu(tf.matmul(self.Z, self.decoder_W1) + self.decoder_b1)
        Z5 = tf.nn.relu(tf.matmul(Z4, self.decoder_W2) + self.decoder_b2)

        # no activation function
        logits = tf.matmul(Z5, self.decoded_space_w) + self.decoded_space_b

        self.X_hat = tf.contrib.distributions.Bernoulli(logits=logits)

        # take samples from X_hat
        # the posterior predictive sample
        self.posterior_predictive = self.X_hat.sample()
        self.posterior_predictive_probs = tf.nn.sigmoid(logits)

        # take sample from a Z ~ N(0, 1)
        # and put it through the decoder
        # we will call this the prior predictive sample
        standard_normal = tf.contrib.distributions.Normal(
            loc=np.zeros(encoding_size, dtype=np.float32),
            scale=np.ones(encoding_size, dtype=np.float32)
        )

        Z_std = standard_normal.sample(1)

        # defining the decoder forward propagation
        Z6 = tf.nn.relu(tf.matmul(Z_std, self.decoder_W1) + self.decoder_b1)
        Z7 = tf.nn.relu(tf.matmul(Z6, self.decoder_W2) + self.decoder_b2)

        # no activation function
        logits = tf.matmul(Z7, self.decoded_space_w) + self.decoded_space_b

        prior_predictive_dist = tf.contrib.distributions.Bernoulli(logits=logits)
        self.prior_predictive = prior_predictive_dist.sample()
        self.prior_predictive_probs = tf.nn.sigmoid(logits)

        # prior predictive from input
        # only used for generating visualization
        self.Z_input = tf.placeholder(tf.float32, shape=(None, encoding_size))
        Z8 = tf.nn.relu(tf.matmul(self.Z_input, self.decoder_W1) + self.decoder_b1)
        Z9 = tf.nn.relu(tf.matmul(Z8, self.decoder_W2) + self.decoder_b2)

        # no activation function
        logits = tf.matmul(Z9, self.decoded_space_w) + self.decoded_space_b
        self.prior_predictive_from_input_probs = tf.nn.sigmoid(logits)

        # now build the cost
        kl = -tf.log(self.stddev) + 0.5 * (self.stddev ** 2 + self.means ** 2) - 0.5
        kl = tf.reduce_sum(kl, axis=1)

        expected_log_likelihood = tf.reduce_sum(self.X_hat.log_prob(self.X), 1)

        # ELBO cost function
        self.elbo = tf.reduce_sum(expected_log_likelihood - kl)
        self.train_op = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(-self.elbo)

        # set up session and variables for later
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    # training the model with dataset
    def fit(self, X, epochs=50, batch_sz=64):
        costs = []
        n_batches = len(X) // batch_sz
        for i in range(epochs):
            np.random.shuffle(X)
            for j in range(n_batches):
                batch = X[j * batch_sz:(j + 1) * batch_sz]
                _, c, = self.sess.run((self.train_op, self.elbo), feed_dict={self.X: batch})
                c /= batch_sz
                costs.append(c)
                if j % 100 == 0:
                    print("epoch:%d iter: %d, cost: %.3f" % (i, j, c))

        # visualizing the training history
        plt.plot(costs)
        plt.show()

    def transform(self, X):
        return self.sess.run(
            self.means,
            feed_dict={self.X: X}
        )

    def prior_predictive_with_input(self, Z):
        return self.sess.run(
            self.prior_predictive_from_input_probs,
            feed_dict={self.Z_input: Z}
        )

    def posterior_predictive_sample(self, X):
        # returns a sample from p(x_new | X)
        return self.sess.run(self.posterior_predictive, feed_dict={self.X: X})

    def prior_predictive_sample_with_probs(self):
        # returns a sample from p(x_new | z), z ~ N(0, 1)
        return self.sess.run((self.prior_predictive, self.prior_predictive_probs))


# prepare input data
df = pd.read_csv('./dataset/mnist_train.csv')

# prepare test data
df_test = pd.read_csv('./dataset/mnist_test.csv')

# converting to numpy array
X = df.iloc[:, 1:].values
x_test = df_test.iloc[:, 1:].values

# normalizing
X_train = X.astype('float32') / 255.
X_test = x_test.astype('float32') / 255.

# creating VAE model
vae = VariationalAutoencoder(784, 264, 128, 64)

# training the VAE model
vae.fit(X_train)

# plotting reconstructions of randomly selected datapoints
# original and reconstructed along side for comparison
for i in range(10):
    ax = plt.subplot(2, 10, i + 1)
    plt.imshow(X[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, 10, i + 1 + 10)
    im = vae.posterior_predictive_sample([X[i]]).reshape(28, 28)
    plt.imshow(im)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

# plot output from random samples in latent space
# plotting prior predictive samples (samples from a standard normal)
# sample and mean sample
for i in range(10):
    im, probs = vae.prior_predictive_sample_with_probs()
    im = im.reshape(28, 28)
    probs = probs.reshape(28, 28)
    ax = plt.subplot(2, 10, i + 1)
    plt.imshow(im, cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, 10, i + 1 + 10)
    plt.imshow(probs, cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

