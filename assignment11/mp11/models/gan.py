"""Generative adversarial network."""

import numpy as np
import tensorflow as tf

from tensorflow import contrib
from tensorflow.contrib import layers


class Gan(object):
    """Adversary based generator network.
    """
    def __init__(self, ndims=784, nlatent=2):
        """Initializes a GAN

        Args:
            ndims(int): Number of dimensions in the feature.
            nlatent(int): Number of dimensions in the latent space.
        """

        self._ndims = ndims
        self._nlatent = nlatent

        # Input images
        self.x_placeholder = tf.placeholder(tf.float32, [None, ndims])
        # Input noise
        self.z_placeholder = tf.placeholder(tf.float32, [None, nlatent])
        # Add learning rate
        self.lr_placeholder = tf.placeholder(tf.float32, [])

        # Build graph.
        self.x_hat = self._generator(self.z_placeholder)                    # gen_fake
        y_hat      = self._discriminator(self.x_hat)                        # disc_fake
        y          = self._discriminator(self.x_placeholder, reuse=True)    # disc_real 
        #reuse=True since reuse the same cell twice

        # Discriminator loss
        self.d_loss = self._discriminator_loss(y, y_hat)
        # Generator loss
        self.g_loss = self._generator_loss(y_hat)

        # Add optimizers for appropriate variables
        params = tf.trainable_variables()
        G_params = [i for i in params if 'generator' in i.name]
        D_params = [i for i in params if 'discriminator' in i.name]
        self.optimizer_disc = tf.train.AdamOptimizer(
            learning_rate=self.lr_placeholder).minimize(self.d_loss, var_list=D_params)
        self.optimizer_gen = tf.train.AdamOptimizer(
            learning_rate=self.lr_placeholder).minimize(self.g_loss, var_list=G_params)

        # Create session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())


    def _discriminator(self, x, reuse=False):
        """Discriminator block of the network.

        Args:
            x (tf.Tensor): The input tensor of dimension (None, 784).
            reuse (Boolean): re use variables with same name in scope instead of creating
              new ones, check Tensorflow documentation
        Returns:
            y (tf.Tensor): Scalar output prediction D(x) for true vs fake image(None, 1). 
              DO NOT USE AN ACTIVATION FUNCTION AT THE OUTPUT LAYER HERE.

        """
        
        with tf.variable_scope("discriminator", reuse=reuse) as scope:
            h1 = layers.fully_connected(inputs=x, num_outputs=256, activation_fn=tf.nn.relu)
            #h2 = layers.fully_connected(inputs=h1, num_outputs=128, activation_fn=tf.nn.relu)
            #y = layers.fully_connected(inputs=h1, num_outputs=1, activation_fn=tf.sigmoid)
            y = layers.fully_connected(inputs=h1, num_outputs=1, activation_fn=None)



            return y


    def _discriminator_loss(self, y, y_hat):
        """Loss for the discriminator.

        Args:
            y (tf.Tensor): The output tensor of the discriminator for true images of dimension (None, 1).
            y_hat (tf.Tensor): The output tensor of the discriminator for fake images of dimension (None, 1).
        Returns:
            l (tf.Scalar): average batch loss for the discriminator.

        """
        D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=tf.ones_like(y)))
        D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat, labels=tf.zeros_like(y_hat)))
        l = D_loss_real + D_loss_fake
        #l = -tf.reduce_mean(tf.log(y) + tf.log(1. - y_hat))
        return l


    def _generator(self, z, reuse=False):
        """From a sampled z, generate an image.

        Args:
            z(tf.Tensor): z from _sample_z of dimension (None, 2).
            reuse (Boolean): re use variables with same name in scope instead of creating
              new ones, check Tensorflow documentation 
        Returns:
            x_hat(tf.Tensor): Fake image G(z) (None, 784).
        """
        with tf.variable_scope("generator", reuse=reuse) as scope:
            h1 = layers.fully_connected(inputs=z, num_outputs=256, activation_fn=tf.nn.relu)
            #h2 = layers.fully_connected(inputs=h1, num_outputs=256, activation_fn=tf.nn.relu)
            x_hat = layers.fully_connected(inputs=h1, num_outputs=784, activation_fn=tf.nn.sigmoid)

            return x_hat


    def _generator_loss(self, y_hat):
        """Loss for the discriminator.

        Args:
            y_hat (tf.Tensor): The output tensor of the discriminator for fake images of dimension (None, 1).
        Returns:
            l (tf.Scalar): average batch loss for the discriminator.

        """
        l = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat, labels=tf.ones_like(y_hat)))
        #l = - tf.reduce_mean(tf.log(y_hat))
        return l
