# Generative-Modeling

<p align="center"><i>"What I cannot create, I do not understand"</i> — Richard Feynman </p>

Modeling the environment is the most crucial and challenging aspect of a problem. For example, understanding how an unknown probabilistic distribution explains why some data points are likely to be found in specific circumstances and others are not. Our job is to build a model that mimics this distribution as closely as possible and then sample from it to generate new, distinct observations that look as if they are from the original data set. By doing this, we can create more realistic and thought-through solutions. 

As most data present in the real world is unsupervised, this modeling task becomes super essential.

In this repository, I implemented various generative models to learn the distribution of the MNIST digit dataset and reconstruct the digit images. I have added the result of the models in this readme.

## Conditional Generative Adversarial Network
In GANs we do not deal with explicit distributions, the goal is to reach nash equilibrium of a game. 

Game - generator job is to fool discriminator .. discrimator is to classify correctly.

In GAN, we donot have any control on type of image generated, control on output disadvantage of GANs. CGAN tackles this by inserting label information. So, in cost fucntion we insert condition of label, i.e., we use conditional probability in cost, conditioned on the label y.

<p align="center"><img src="/images/MNIST_cGAN_generation_animation.gif" height="400px" width="400px"></p>

Reference - https://arxiv.org/abs/1411.1784

## Deep Convolutional Generative Adversarial Network

Features in DCGAN include batch normalization (adaptive normalisation of data at each layer), conv-> conv-> conv (it is a all convolutional network), adam optimizer, leaky relu for discriminator, for upsampling we use fractionally strided convolution: conv2d transpose does this.

<p align="center"><img src="/images/MNIST_DCGAN_15.png" height="400px" width="400px"></p>

Reference - https://arxiv.org/abs/1511.06434

## Autoencoder
In this unsupervised learning technique, we impose a bottleneck in the neural network architecture which forces a compressed knowledge representation of the original input. Then we try to optimise this representation, by reconstructing the original image, minimizing the reconstruction error between original input image and its the consequent reconstruction. 

> Result

<p align="center"><img src="/images/autoencoderimg.png" height="400px" width="400px"></p>

### Denoising Autoencoder
Autoencoders had generally performed weakly in generalizing the encoding and decoding. So, to obtain a generalizable model, we slightly corrupt the input data but still maintain the uncorrupted data as our target output. This way, our model is not just memorizing the data but learning the latent representation. A denoising autoencoder processes a noisy image, generating a clean image on the output side. I added random noise to training set images and let the autoencoder reconstruct the original clean image from it.

In the paper, [Extracting and Composing Robust Features with Denoising Autoencoders](https://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf), the authors reported that "unsupervised initialization of layers with an explicit denoising criterion helps to capture interesting structure in the input distribution." Furthermore, they were able to improve the robustness of their internal network layers (i.e., latent-space representation) by deliberately introducing noise to their signal.

> Result
<p align="center"><img src="/images/denoising auto.png" height="220px" width="220px"></p>
<p align="center">Denoising the MNIST digits</p>

Reference - https://www.jeremyjordan.me/autoencoders/

## Variational Autoencoder
<p align="center"><i>Variational Autoencoder = variational inference + autoencoders </i></p>

Variational Autoencoder (VAE) gives precise control over your latent representations and what we would like them to represent, but vanilla autoencoders does not.

In VAE, we are learn the distribution, in the bottleneck, then we sample from this distribution as a input to decoder. So, encoder output is not a value but a distribution.

In VAE, our objective function is elbo - evidence lower bound, we want to maximize the ELBO, therefore our "cost" (we want to minimize) is (-ELBO).

ELBO = expected log likelihood + KL divergence

> Result 

- Prior predictive samples (compared sample and the mean from which sample generated)
<p align="center"><img src="/images/vae-priors.png" height="300px" width="350px"></p>

- Compared original and reconstruction 
<p align="center"><img src="/images/vae-reconstructions.png" height="300px" width="350px"></p>
Reference -  https://arxiv.org/abs/1606.05908, https://www.jeremyjordan.me/variational-autoencoders/

## Gaussian Mixture Model
<p align="center"><img src="/images/gmm syn.png"></p>

### With MNIST

We have used a mixture of gaussians to fit, so that may sample better digits (less blurry).
<p align="center"><img src="/images/gmm mnist1.png"><img src="/images/gmm mnist2.png"><img src="/images/gmm mnist3.png"></p>

Reference - http://www.cse.psu.edu/~rtc12/CSE586Spring2010/papers/prmlMixturesEM.pdf

## Sampling from Bayes classifier
We have fitted a single gaussian in multi model distribution that is why the generated images are somewhat blurry.
<p align="center"><img src="/images/bayes1.png"></p>
<p align="center"><img src="/images/bayes2.png"></p>

Reference - https://ermongroup.github.io/cs228-notes/inference/sampling/
