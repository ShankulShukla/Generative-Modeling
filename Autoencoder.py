
# Autoencoder to learn distribution of first 10 digits

# importing dependencies
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

# class to generate the model of the Autoencoder
class AutoEncoder:
    def __init__(self, inp_feature, hidden_feature, output_feature):

        # training placeholder
        self.X = tf.placeholder(tf.float32, shape=(None, inp_feature))

        # encoding layer # 1
        self.encoder_W1 = tf.Variable(
            tf.random_normal(shape=(inp_feature, hidden_feature)) * 2 / np.sqrt(hidden_feature))
        self.encoder_b1 = tf.Variable(np.zeros(hidden_feature).astype(np.float32))

        # encoding layer # 2
        self.encoder_W2 = tf.Variable(
            tf.random_normal(shape=(hidden_feature, output_feature)) * 2 / np.sqrt(output_feature))
        self.encoder_b2 = tf.Variable(np.zeros(output_feature).astype(np.float32))

        # decoding layer # 1
        self.decoder_W1 = tf.Variable(
            tf.random_normal(shape=(output_feature, hidden_feature)) * 2 / np.sqrt(hidden_feature))
        self.decoder_b1 = tf.Variable(np.zeros(hidden_feature).astype(np.float32))

        # decoding layer # 2
        self.decoder_W2 = tf.Variable(tf.random_normal(shape=(hidden_feature, inp_feature)) * 2 / np.sqrt(inp_feature))
        self.decoder_b2 = tf.Variable(np.zeros(inp_feature).astype(np.float32))

        # encoding layers operations
        self.Z1 = tf.nn.sigmoid(tf.matmul(self.X, self.encoder_W1) + self.encoder_b1)
        self.Z2 = tf.nn.sigmoid(tf.matmul(self.Z1, self.encoder_W2) + self.encoder_b2)

        # decoding layers operations
        self.Z3 = tf.nn.sigmoid(tf.matmul(self.Z2, self.decoder_W1) + self.decoder_b1)
        self.X_hat = tf.nn.sigmoid(tf.matmul(self.Z3, self.decoder_W2) + self.decoder_b2)

        # loss function (squared error)
        self.loss = tf.reduce_mean(tf.pow(self.X - self.X_hat, 2))

        # RMSprop optimiser
        self.opt = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(self.loss)

        # setting tensorflow session
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    # training the autoencoder
    def fit(self, X, epochs=30, batch_size=128):
        loss = []
        num_batch = len(X) // batch_size
        for ep in range(epochs):
            for i in range(num_batch):
                x_batch = X[i * batch_size:(i + 1) * batch_size]
                _, l = self.sess.run((self.opt, self.loss), feed_dict={self.X: x_batch})
                loss.append(l / batch_size)
                if i % 100:
                    print("epoch:%d, iter: %d, loss:%.3f" % (ep, i, l))

        # visualizing training history
        plt.plot(loss)
        plt.show()

    # generating samples from autoencoder
    def predict(self, x):
        return self.sess.run(self.X_hat, feed_dict={self.X: x})


# prepare input data
df = pd.read_csv('./dataset/mnist_train.csv')

# plotting random digit
plt.imshow(df.iloc[100, 1:].values.reshape(28, 28))

# creating the model object of the autoencoder with providing the layer sizes
model = AutoEncoder(784, 256, 128)

# normalize all values between 0 and 1
X = df.iloc[:, 1:].values / 255.

# Train autoencoder on MNIST dataset
model.fit(X)

# now visualizing the reconstructed digit representations
im = model.predict([X[7001]]).reshape(28, 28)

plt.imshow(im, cmap='gray')

plt.imshow(df.iloc[7001, 1:].values.reshape(28, 28), cmap='gray')

im = model.predict([X[701]]).reshape(28, 28)

plt.imshow(im, cmap='gray')

plt.imshow(df.iloc[701, 1:].values.reshape(28, 28), cmap='gray')

