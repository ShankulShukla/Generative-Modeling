
# Removing noise from MNIST digit dataset using autoencoder

# using tensorflow's keras interface
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

# Defining CNN autoencoder architecture from the paper
def CNNAutoEncoder(input):

    # Encoder of the Auto-encoder (3 layer of convolution)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    encoded_layer = MaxPooling2D((2, 2), padding='same')(x)

    # Decoder of the Auto-encoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded_layer)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded_layer = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    return decoded_layer


# Adds random noise to each image in the supplied array.
def noise(image):
    noise_factor = 0.4
    noisy_image = image + noise_factor * np.random.normal(
        loc=0.0, scale=1.0, size=image.shape
    )

    return np.clip(noisy_image, 0.0, 1.0)


# Building the network with input size
input = Input(shape=(28, 28, 1))

# regenerated de-noised result
decoded_layer = CNNAutoEncoder(input)

# Defining the parameters of the Auto-encoder for training
autoencoder = Model(input, decoded_layer)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

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

# reshaping the dataset
X_train = np.reshape(X_train, (len(X_train), 28, 28, 1))
X_test = np.reshape(X_test, (len(X_test), 28, 28, 1))

# adding defined noise to the training and test dataset
noisy_train_data = noise(X_train)
noisy_test_data = noise(X_test)

# training the CNN based autoencoder model with defined hyper-parameters
autoencoder.fit(x=noisy_train_data,
                y=X_train,
                epochs=30,
                batch_size=128,
                shuffle=True,
                validation_data=(noisy_test_data, X_test))

# reconstructed test digits
pred = autoencoder.predict(noisy_test_data)

# creating the plot comparing the result  after training for each digit
for i in range(10):
    ax = plt.subplot(2, 10, i + 1)
    plt.imshow(noisy_test_data[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, 10, i + 1 + 10)
    plt.imshow(pred[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

