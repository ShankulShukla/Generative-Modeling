# Generative-Modeling

<p align="center"><i>"What I cannot create, I do not understand"</i> — Richard Feynman </p>

Modeling the environment is the most crucial and challenging aspect of a problem. For example, understanding how an unknown probabilistic distribution explains why some data points are likely to be found in specific circumstances and others are not. Our job is to build a model that mimics this distribution as closely as possible and then sample from it to generate new, distinct observations that look as if they are from the original data set. By doing this, we can create more realistic and thought-through solutions. 

As most data present in the real world is unsupervised, this modeling task becomes super essential.

In this repository, I implemented various generative models to learn the distribution of the MNIST digit dataset and reconstruct the digit images. I have added the result of the models in this readme.

## Conditional Generative Adversarial Network
The Generative Adversarial Network (GAN) is one of the most influential ideas of computer science, in which we not only understand the distribution of data using neural networks but we try to improve upon the inference by using two networks: generator and discriminator.

In GANs, we do not deal with explicit distributions; the goal is to reach a nash equilibrium of a game.

Game - generator job is to fool discriminator .. discrimator is to classify correctly.

In GAN, we donot have any control on type of image generated, control on output disadvantage of GANs. Conditional Generative Adversarial Network (CGAN) tackles this by inserting label information. So, in cost fucntion we insert condition of label, i.e., we use conditional probability in cost, conditioned on the label y.

I used CGAN to achieve targeted image generation.

> Result 

<p align="center"><img src="/images/MNIST_cGAN_generation_animation.gif" height="400px" width="400px"></p>

Reference - https://arxiv.org/abs/1411.1784

## Deep Convolutional Generative Adversarial Network
The Deep Convolutional Generative Adversarial Network (DCGAN) brings best of both the worlds i.e., the power of image extraction by convolution and the pragmatic inference of the data by GAN.

The DCGAN is important because it suggested the constraints on the model required to effectively develop high-quality generator models in practice. This architecture, in turn, provided the basis for the rapid development of a large number of GAN extensions and applications.

In DCGAN, the discriminator consists of strided convolution layers, batch normalization layers(adaptive normalisation of data at each layer), and LeakyRelu as activation function. The generator consists of convolutional-transpose layers, batch normalization layers, and ReLU activations.

> Result 

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
Gaussian Mixture Models (GMMs) assume that there are a certain number of Gaussian distributions from which all the data points are generated.

It is basically a latent variable model, we try to model z, "z" latent variable represents basically which cluster data belong to.

GMM are trained using expectation-maximization(EM), we use EM for latent variable. The goal is to improve likelihood at each step.

In the below example, I randomly generated three data clusters and let GMM iteratively model the distribution.
> Result 

<p align="center"><img src="/images/gmm syn.png"></p>

### With MNIST

I have used a mixture of gaussians to fit, so that I sampled better digits (less blurry than bayes).

> Result 

<p align="center"><img src="/images/gmm mnist1.png"><img src="/images/gmm mnist2.png"><img src="/images/gmm mnist3.png"></p>

Reference - http://www.cse.psu.edu/~rtc12/CSE586Spring2010/papers/prmlMixturesEM.pdf

## Sampling from Bayes classifier
Using the Bayesian theorem as the basis, for each class y (digit class), we model p(x|y) rather than directly modeling p(y|x). In this, we found mean and covariance corresponding to each class in the dataset, then for sampling, for each category, we pick the class, and we have defined in such a way that p(x|y) is a gaussian, so we sample from this particular class gaussian.

> Result 

<p align="center"><img src="/images/bayes1.png"></p>
<p align="center"><img src="/images/bayes2.png"></p>

Reference - https://ermongroup.github.io/cs228-notes/inference/sampling/
