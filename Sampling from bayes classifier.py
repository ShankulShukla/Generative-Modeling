
# sampling MINIST digits using bayes classifier

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
import pandas as pd

# Gaussian bayes classifier
class BayesClassifier:
    def fit(self, X, Y):
        # classes
        self.K = set(Y)

        # store the mean and covariance
        self.gaussian = {}
        # for each class calculate the gaussian
        for k in self.K:
            Xk = X[Y == k]
            mean = Xk.mean(axis=0)
            cov = np.cov(Xk.T)
            g = {'m': mean, 'c': cov}
            self.gaussian[k] = g

    # Draw random samples from a multivariate normal distribution
    def sampleGenerate(self, y):
        g = self.gaussian[y]
        return mvn.rvs(mean=g['m'], cov=g['c'])


# prepare input data
df = pd.read_csv('./dataset/mnist_train.csv')

X, Y = df.iloc[:, 1:], df['label']

clf = BayesClassifier()

clf.fit(X.values, Y.values)

# Creating and plotting mean sample and approximated sample from the each digit distribution
for k in set(Y):
    gen_sam = clf.sampleGenerate(k)
    gen_mean = clf.gaussian[k]['m']
    plt.subplot(1, 2, 1)
    plt.imshow(gen_sam.reshape(28, 28), cmap="gray")
    plt.title("sample generated for {}".format(k))
    plt.subplot(1, 2, 2)
    plt.imshow(gen_mean.reshape(28, 28), cmap="gray")
    plt.title("mean sample generated for {}".format(k))
    plt.show()

