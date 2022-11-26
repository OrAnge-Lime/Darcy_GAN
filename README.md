# Realization of Deep Convolutional Generative Adversarial Network

DCGAN model is a direct extension of the GAN, except that it explicitly uses convolutional and transpose convolutional layers in the discriminator and generator, respectively

![image](https://user-images.githubusercontent.com/71509624/204069188-0ea420cd-fd58-41e0-96bb-1ddc595f3177.png)

This repo contains the dataset of ragdoll cats and a code of DCGAN model, that generates them.
Before training, created dataset was augmented using python albumentations library with the `preprocessing.py` module code, which increases the size of dataset several times

Architecture of generator model contains the several transpose convolutional layers with also Leaky ReLU activation function (except for the last layer where it is tanh). Also, all of the layers separetad with Batch Normalization layers, to avoid owerfiting.

Architecture of discriminator model contains the several convolutional layers with Leaky ReLU activation function, as shown on the image below.

![image](https://user-images.githubusercontent.com/71509624/204064196-02e4faea-3d30-4d1b-9290-1b6702e8f653.png)

The result of the training prosess are provided below. It is still remains blurry, but in some cases it is possibly to catch a glimps of a cat outline.

![movie](https://user-images.githubusercontent.com/71509624/204069021-76901389-9805-4dc5-b336-83785aba432e.gif)





