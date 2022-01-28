
# mixture of gaussin fit so that may sample better digits

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.mixture import BayesianGaussianMixture

# gaussian mixture bayes classifier
class BGM:
    def fit(self, X, Y):
        # classes
        self.K = set(Y)

        # store gmm model for each class
        self.gaussian = {}
        # for each class calculate the gaussian
        for k in self.K:
            Xk = X[Y == k]
            model = BayesianGaussianMixture(5)
            model.fit(Xk)
            self.gaussian[k] = model

    def sampleGenerate(self, y):
        gmm = self.gaussian[y]
        # sampling from gaussian mixture
        sample = gmm.sample()

        return sample[0].reshape(28, 28)


# prepare input data
df = pd.read_csv('./dataset/mnist_train.csv')

X, Y = df.iloc[:, 1:], df['label']

clf = BGM()

clf.fit(X.values, Y.values)

# plotting the generated sample for each image
for k in set(Y):
    gen_sam = clf.sampleGenerate(k)
    plt.subplot(1, 2, 1)
    plt.imshow(gen_sam.reshape(28, 28), cmap="gray")
    plt.title("sample generated for {}".format(k))
    plt.show()

