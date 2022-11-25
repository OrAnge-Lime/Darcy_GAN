# Realization of Deep Convolutional Generative Adversarial Network

DCGAN model is a direct extension of the GAN, except that it explicitly uses convolutional and transpose convolutional layers in the discriminator and generator, respectively
This repo contains the dataset of ragdoll cats and a code of DCGAN model, that generates them.
Before training, created dataset was augmented using python albumentations library with the `preprocessing.py` module code, which increases the size of dataset several times

Architecture of discriminator model contains the several convolutional layers with Leaky ReLU activation function, as shown on the image below.

![discriminator](https://user-images.githubusercontent.com/71509624/204063354-d5918a99-dcde-4d3c-8f32-a8093a534bfa.png)

Architecture of generator model contains the several transpose convolutional layers with also Leaky ReLU activation function (except for the last layer where it is tanh).
Also, all of the layers separetad with Batch Normalization layers, to avoid owerfiting.

![generator](https://user-images.githubusercontent.com/71509624/204063370-347690f4-b822-4176-a9c0-4678aae4cfd2.png)





