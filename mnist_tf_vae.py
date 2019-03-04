"""
VAE on MNIST data.
Adapted from https://jmetzen.github.io/2015-11-27/vae.html.
"""


import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

random.seed(0)
np.random.seed(0)
tf.set_random_seed(0)


BATCH_SIZE = 64
LEARNING_RATE = 0.01
NUM_EPOCHS = 10
LATENT_DIM = 2
TENSORBOARD_TRAIN_DIR = '/Users/olex/tb/train'
TENSORBOARD_TEST_DIR = '/Users/olex/tb/test'


def variable_summaries(var):
    """
    Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    Taken from https://www.tensorflow.org/guide/summaries_and_tensorboard.
    """

    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


class VAE(object):
    """
    Dense VAE adapted from https://jmetzen.github.io/2015-11-27/vae.html.
    """

    def __init__(self, network_architecture, transfer_fct=tf.nn.softplus):
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct  # Activation.

        # tf Graph input - these will be flattened MNIST images.
        self.x = tf.placeholder(tf.float32, [None, network_architecture["n_input"]])

        # Create autoencoder network.
        self._create_network()

        # ELBO and corresponding optimizer.
        self._create_loss_optimizer()

        # Launch the session.
        self.sess = tf.InteractiveSession()

        # Setup Tensorboard monitoring.
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(TENSORBOARD_TRAIN_DIR, self.sess.graph)
        self.test_writer = tf.summary.FileWriter(TENSORBOARD_TEST_DIR)

        # Initializing the tensor flow variables.
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def _create_network(self):
        # Initialize autoencoder network weights and biases.
        network_weights = self._initialize_weights(**self.network_architecture)

        # Use recognition network to determine mean and
        # (log) variance of Gaussian distribution in latent
        # space.
        self.z_mean, self.z_log_sigma_sq = \
            self._recognition_network(network_weights["weights_recog"],
                                      network_weights["biases_recog"])

        # Draw one sample z from Gaussian distribution.
        n_z = self.network_architecture["n_z"]
        eps = tf.random_normal((BATCH_SIZE, n_z), 0, 1, dtype=tf.float32)
        self.z = tf.add(self.z_mean,
                        tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

        # Use generator to determine mean of
        # Bernoulli distribution of reconstructed input.
        # Bernoulli is used because the MNIST input is binarized!
        # TODO When switching to Gaussian (grey-scale MNIST), should output Sigmas as well
        self.x_reconstr_mean = \
            self._generator_network(network_weights["weights_gener"],
                                    network_weights["biases_gener"])

    def _initialize_weights(self, n_hidden_recog_1, n_hidden_recog_2,
                            n_hidden_gener_1, n_hidden_gener_2,
                            n_input, n_z):
        xavier_init = xavier_initializer()

        all_weights = dict()
        all_weights['weights_recog'] = {
            'h1': tf.Variable(xavier_init(shape=(n_input, n_hidden_recog_1))),
            'h2': tf.Variable(xavier_init(shape=(n_hidden_recog_1, n_hidden_recog_2))),
            'out_mean': tf.Variable(xavier_init(shape=(n_hidden_recog_2, n_z))),
            'out_log_sigma': tf.Variable(xavier_init(shape=(n_hidden_recog_2, n_z)))
        }
        all_weights['biases_recog'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32))
        }
        all_weights['weights_gener'] = {
            'h1': tf.Variable(xavier_init(shape=(n_z, n_hidden_gener_1))),
            'h2': tf.Variable(xavier_init(shape=(n_hidden_gener_1, n_hidden_gener_2))),
            'out_mean': tf.Variable(xavier_init(shape=(n_hidden_gener_2, n_input))),
            # As we are doing Bernoulli in the output, log sigma is not needed for now.
            # 'out_log_sigma': tf.Variable(xavier_init(shape=(n_hidden_gener_2, n_input)))
        }
        all_weights['biases_gener'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_gener_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_gener_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_input], dtype=tf.float32)),
            # As we are doing Bernoulli in the output, log sigma is not needed for now.
            # 'out_log_sigma': tf.Variable(tf.zeros([n_input], dtype=tf.float32))
        }

        # Add Tensorboard summaries to all weights and biases.  # TODO Normal scope names for layers!
        # Encoder.
        variable_summaries(all_weights['weights_recog']['h1'])
        variable_summaries(all_weights['weights_recog']['h2'])
        variable_summaries(all_weights['weights_recog']['out_mean'])
        variable_summaries(all_weights['weights_recog']['out_log_sigma'])
        variable_summaries(all_weights['biases_recog']['b1'])
        variable_summaries(all_weights['biases_recog']['b2'])
        variable_summaries(all_weights['biases_recog']['out_mean'])
        variable_summaries(all_weights['biases_recog']['out_log_sigma'])
        # Decoder.
        variable_summaries(all_weights['weights_gener']['h1'])
        variable_summaries(all_weights['weights_gener']['h2'])
        variable_summaries(all_weights['weights_gener']['out_mean'])
        variable_summaries(all_weights['biases_gener']['b1'])
        variable_summaries(all_weights['biases_gener']['b2'])
        variable_summaries(all_weights['biases_gener']['out_mean'])

        return all_weights

    def _recognition_network(self, weights, biases):
        # Generate probabilistic encoder (recognition network), which
        # maps inputs onto a normal distribution in latent space.
        # The transformation is parametrized and can be learned.
        # TODO TB summaries for activations and preactivations
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.x, weights['h1']),
                                           biases['b1']))
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']),
                                           biases['b2']))
        z_mean = tf.add(tf.matmul(layer_2, weights['out_mean']),
                        biases['out_mean'])
        z_log_sigma_sq = tf.add(tf.matmul(layer_2, weights['out_log_sigma']),
                                biases['out_log_sigma'])
        return z_mean, z_log_sigma_sq

    def _generator_network(self, weights, biases):
        # Generate probabilistic decoder (decoder network), which
        # maps points in latent space onto a Bernoulli distribution in data space.
        # The transformation is parametrized and can be learned.
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.z, weights['h1']),
                                           biases['b1']))
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']),
                                           biases['b2']))
        x_reconstr_mean = \
            tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['out_mean']),
                                 biases['out_mean']))
        return x_reconstr_mean

    def _create_loss_optimizer(self):
        # The loss is composed of two terms:
        # 1.) The reconstruction loss (the negative log probability
        #     of the input under the reconstructed Bernoulli distribution
        #     induced by the decoder in the data space).
        #     This can be interpreted as the number of "nats" required
        #     for reconstructing the input when the activation in latent
        #     is given.
        # Adding 1e-10 to avoid evaluation of log(0.0).
        reconstr_loss = \
            -tf.reduce_sum(self.x * tf.log(1e-10 + self.x_reconstr_mean)
                           + (1 - self.x) * tf.log(1e-10 + 1 - self.x_reconstr_mean),
                           1)
        # 2.) The latent loss, which is defined as the Kullback-Leibler divergence
        #     between the distribution in latent space induced by the encoder on
        #     the data and some prior. This acts as a kind of regularizer.
        #     This can be interpreted as the number of "nats" required
        #     for transmitting the the latent space distribution given
        #     the prior.
        latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq
                                           - tf.square(self.z_mean)
                                           - tf.exp(self.z_log_sigma_sq), 1)

        self.cost = tf.reduce_mean(reconstr_loss + latent_loss)  # Average over batch.

        # Use ADAM optimizer  # TODO Use SGD
        self.optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(self.cost)

    def partial_fit(self, x):
        """
        Train model based on mini-batch of input data.

        Return cost of mini-batch.
        """

        summary, opt, cost = self.sess.run((self.merged, self.optimizer, self.cost),
                                           feed_dict={self.x: x})
        return summary, opt, cost

    def encode(self, x):
        """Transform data by mapping it into the latent space."""

        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution.
        return self.sess.run(self.z_mean, feed_dict={self.x: x})

    def decode(self, z_mu=None):
        """
        Generate data by sampling from latent space.

        If z_mu is not None, data for this point in latent space is
        generated. Otherwise, z_mu is drawn from prior in latent
        space.
        """
        if z_mu is None:
            z_mu = np.random.normal(size=self.network_architecture["n_z"])
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.x_reconstr_mean,
                             feed_dict={self.z: z_mu})

    def reconstruct(self, x):
        """Use VAE to reconstruct given data."""

        return self.sess.run(self.x_reconstr_mean,
                             feed_dict={self.x: x})


def train(network_architecture, mnist):
    vae = VAE(network_architecture)
    n_samples = mnist.train.num_examples
    n_batches = int(n_samples / BATCH_SIZE)  # Number of batches in 1 epoch.

    # Training cycle.
    for epoch in range(NUM_EPOCHS):
        avg_cost = 0.  # Avg cost per all batches processed so far.

        # Loop over all batches.
        for i in range(n_batches):
            # TODO Need to generate batches by myself as we want to work with mmaps
            batch_xs, _ = mnist.train.next_batch(BATCH_SIZE)  # Labels are dropped.

            # Fit training using batch data.
            summary, opt, cost = vae.partial_fit(batch_xs)
            vae.train_writer.add_summary(summary, epoch * n_batches + i)

            # Compute average loss.
            avg_cost += cost / n_samples * BATCH_SIZE

        print("Epoch:", '%04d' % (epoch + 1),
              "avg seen batch cost=", "{:.9f}".format(avg_cost))
    return vae


def plot_example_reconstr(vae, mnist):
    x_sample, _ = mnist.test.next_batch(BATCH_SIZE)  # Labels are dropped.
    x_reconstruct = vae.reconstruct(x_sample)

    plt.figure(figsize=(8, 12))

    for i in range(5):
        plt.subplot(5, 2, 2 * i + 1)
        plt.imshow(x_sample[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
        plt.title("Test input")
        plt.colorbar()
        plt.subplot(5, 2, 2 * i + 2)
        plt.imshow(x_reconstruct[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
        plt.title("Reconstruction")
        plt.colorbar()

    plt.tight_layout()
    plt.savefig('tf_vae_example_reconstr.png', dpi=150)
    plt.gcf().clear()


def main():
    network_architecture = \
        dict(n_hidden_recog_1=256,  # 1st layer encoder neurons
             n_hidden_recog_2=128,  # 2nd layer encoder neurons
             n_hidden_gener_1=128,  # 1st layer decoder neurons
             n_hidden_gener_2=256,  # 2nd layer decoder neurons
             n_input=784,  # MNIST data input (img shape: 28*28)
             n_z=LATENT_DIM)  # dimensionality of latent space

    # Load MNIST data in a format suited for tensorflow.
    # The script input_data is available under this URL:
    # https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/tutorials/mnist/input_data.py
    import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    vae = train(network_architecture, mnist)

    plot_example_reconstr(vae, mnist)


if __name__ == '__main__':
    main()
