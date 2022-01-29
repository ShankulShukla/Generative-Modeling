# Generative-Modeling

<p align="center"><i>"What I cannot create, I do not understand"</i> — Richard Feynman </p>

Modeling the environment is the most crucial and challenging aspect of a problem. For example, understanding how an unknown probabilistic distribution explains why some data points are likely to be found in specific circumstances and others are not. Our job is to build a model that mimics this distribution as closely as possible and then sample from it to generate new, distinct observations that look as if they are from the original data set. By doing this, we can create more realistic and thought-through solutions. 

As most data present in the real world is unsupervised, this modeling task becomes super essential.

## Conditional Generative Adversarial Network
<p align="center"><img src="/images/MNIST_cGAN_generation_animation.gif" height="400px" width="400px"></p>

Reference - https://arxiv.org/abs/1411.1784

## Deep Convolutional Generative Adversarial Network
<p align="center"><img src="/images/MNIST_DCGAN_15.png" height="400px" width="400px"></p>

Reference - https://arxiv.org/abs/1511.06434

## Autoencoder
<p align="center"><img src="/images/autoencoderimg.png" height="400px" width="400px"></p>

### Denoising Autoencoder
<p align="center"><img src="/images/denoising auto.png" height="220px" width="220px"></p>
<p align="center">Denoising the MNIST digits</p>

Reference - https://www.jeremyjordan.me/autoencoders/

## Variational Autoencoder
<p align="center"><img src="/images/vae-priors.png" height="220px" width="220px">   <img src="/images/vae-reconstructions.png" height="220px" width="220px"></p>

Reference -  https://arxiv.org/abs/1606.05908, https://www.jeremyjordan.me/variational-autoencoders/
## Gaussian Mixture Model
<p align="center"><img src="/images/gmm syn.png"></p>

### With MNIST
<p align="center"><img src="/images/gmm mnist1.png"><img src="/images/gmm mnist2.png"><img src="/images/gmm mnist3.png"></p>

Reference - http://www.cse.psu.edu/~rtc12/CSE586Spring2010/papers/prmlMixturesEM.pdf

## Sampling from Bayes classifier
<p align="center"><img src="/images/bayes1.png"></p>
<p align="center"><img src="/images/bayes2.png"></p>

Reference - https://ermongroup.github.io/cs228-notes/inference/sampling/
