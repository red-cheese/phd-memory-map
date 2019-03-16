"""
VAE on MNIST data.
Adapted from https://jmetzen.github.io/2015-11-27/vae.html.
"""


import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
from tensorflow.contrib.layers import optimize_loss, xavier_initializer
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import init_ops

random.seed(0)
np.random.seed(0)
tf.set_random_seed(0)

import mnist_tf_data


BATCH_SIZE = 64
LEARNING_RATE = 0.01
NUM_EPOCHS = 3
LATENT_DIM = 2
TENSORBOARD_TRAIN_DIR = '/Users/olex/tb/train'
TENSORBOARD_TEST_DIR = '/Users/olex/tb/test'
COLOURS = ['blue', 'red', 'green', 'cyan', 'magenta', 'yellow', 'black']


# Needs to be defined for tf.contrib.layers.optimize_loss().
GLOBAL_STEP = variable_scope.get_variable(
      "global_step", [],
      trainable=False,
      dtype=dtypes.int64,
      initializer=init_ops.constant_initializer(0, dtype=dtypes.int64))


def variable_summaries(layer_name, var):
    """
    Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    Taken from https://www.tensorflow.org/guide/summaries_and_tensorboard.
    """

    # Cut off ':0' from the variable name.
    scope_name = '{}/{}'.format(layer_name, var.name[:-2])
    with tf.name_scope(scope_name):
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
            'w1': tf.Variable(name='enc_w1', initial_value=xavier_init(shape=(n_input, n_hidden_recog_1))),
            'w2': tf.Variable(name='enc_w2', initial_value=xavier_init(shape=(n_hidden_recog_1, n_hidden_recog_2))),
            'w_z_mean': tf.Variable(name='enc_z_mean_w', initial_value=xavier_init(shape=(n_hidden_recog_2, n_z))),
            'w_z_log_sigma': tf.Variable(name='enc_z_log_sigma_w', initial_value=xavier_init(shape=(n_hidden_recog_2, n_z)))
        }
        all_weights['biases_recog'] = {
            'b1': tf.Variable(name='enc_b1', initial_value=tf.zeros([n_hidden_recog_1], dtype=tf.float32)),
            'b2': tf.Variable(name='enc_b2', initial_value=tf.zeros([n_hidden_recog_2], dtype=tf.float32)),
            'b_z_mean': tf.Variable(name='enc_z_mean_b', initial_value=tf.zeros([n_z], dtype=tf.float32)),
            'b_z_log_sigma': tf.Variable(name='enc_z_log_sigma_b', initial_value=tf.zeros([n_z], dtype=tf.float32))
        }
        all_weights['weights_gener'] = {
            'w1': tf.Variable(name='dec_w1', initial_value=xavier_init(shape=(n_z, n_hidden_gener_1))),
            'w2': tf.Variable(name='dec_w2', initial_value=xavier_init(shape=(n_hidden_gener_1, n_hidden_gener_2))),
            'w_x_mean': tf.Variable(name='dec_x_mean_w', initial_value=xavier_init(shape=(n_hidden_gener_2, n_input))),
            # As we are doing Bernoulli in the output, log sigma is not needed for now.
            # 'w_x_log_sigma': tf.Variable(xavier_init(shape=(n_hidden_gener_2, n_input)))
        }
        all_weights['biases_gener'] = {
            'b1': tf.Variable(name='dec_b1', initial_value=tf.zeros([n_hidden_gener_1], dtype=tf.float32)),
            'b2': tf.Variable(name='dec_b2', initial_value=tf.zeros([n_hidden_gener_2], dtype=tf.float32)),
            'b_x_mean': tf.Variable(name='dec_x_mean_b', initial_value=tf.zeros([n_input], dtype=tf.float32)),
            # As we are doing Bernoulli in the output, log sigma is not needed for now.
            # 'b_x_log_sigma': tf.Variable(tf.zeros([n_input], dtype=tf.float32))
        }

        # Add Tensorboard summaries to all weights and biases.
        # Encoder.
        variable_summaries('encoder_h1', all_weights['weights_recog']['w1'])
        variable_summaries('encoder_h2', all_weights['weights_recog']['w2'])
        variable_summaries('encoder_z', all_weights['weights_recog']['w_z_mean'])
        variable_summaries('encoder_z', all_weights['weights_recog']['w_z_log_sigma'])
        variable_summaries('encoder_h1', all_weights['biases_recog']['b1'])
        variable_summaries('encoder_h2', all_weights['biases_recog']['b2'])
        variable_summaries('encoder_z', all_weights['biases_recog']['b_z_mean'])
        variable_summaries('encoder_z', all_weights['biases_recog']['b_z_log_sigma'])
        # Decoder.
        variable_summaries('decoder_h1', all_weights['weights_gener']['w1'])
        variable_summaries('decoder_h2', all_weights['weights_gener']['w2'])
        variable_summaries('decoder_x', all_weights['weights_gener']['w_x_mean'])
        variable_summaries('decoder_h1', all_weights['biases_gener']['b1'])
        variable_summaries('decoder_h2', all_weights['biases_gener']['b2'])
        variable_summaries('decoder_x', all_weights['biases_gener']['b_x_mean'])

        return all_weights

    def _recognition_network(self, weights, biases):
        # Generate probabilistic encoder (recognition network), which
        # maps inputs onto a normal distribution in latent space.
        # The transformation is parametrized and can be learned.
        preactiv_1 = tf.add(tf.matmul(self.x, weights['w1']),
                            biases['b1'], name='w1x_plus_b1')
        layer_1 = self.transfer_fct(preactiv_1, name='a1')
        variable_summaries('encoder_h1', preactiv_1)
        variable_summaries('encoder_h1', layer_1)

        preactiv_2 = tf.add(tf.matmul(layer_1, weights['w2']),
                            biases['b2'], name='w2a1_plus_b2')
        layer_2 = self.transfer_fct(preactiv_2, name='a2')
        variable_summaries('encoder_h2', preactiv_2)
        variable_summaries('encoder_h2', preactiv_2)

        z_mean = tf.add(tf.matmul(layer_2, weights['w_z_mean']),
                        biases['b_z_mean'],
                        name='z_mean')
        z_log_sigma_sq = tf.add(tf.matmul(layer_2, weights['w_z_log_sigma']),
                                biases['b_z_log_sigma'],
                                name='z_log_sigma_sq')
        # I'm not sure what it will show but adding it anyway.
        variable_summaries('encoder_z', z_mean)
        variable_summaries('encoder_z', z_log_sigma_sq)

        return z_mean, z_log_sigma_sq

    def _generator_network(self, weights, biases):
        # Generate probabilistic decoder (decoder network), which
        # maps points in latent space onto a Bernoulli distribution in data space.
        # The transformation is parametrized and can be learned.
        preactiv_1 = tf.add(tf.matmul(self.z, weights['w1']),
                            biases['b1'],
                            name='w1z_plus_b1')
        layer_1 = self.transfer_fct(preactiv_1, name='a1')
        variable_summaries('decoder_h1', preactiv_1)
        variable_summaries('decoder_h1', layer_1)

        preactiv_2 = tf.add(tf.matmul(layer_1, weights['w2']),
                            biases['b2'],
                            name='w2a1_plus_b2')
        layer_2 = self.transfer_fct(preactiv_2, name='a2')
        variable_summaries('decoder_h2', preactiv_2)
        variable_summaries('decoder_h2', layer_2)

        x_reconstr_mean = \
            tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['w_x_mean']),
                                 biases['b_x_mean']),
                          name='x_reconstr_mean')
        variable_summaries('decoder_x', x_reconstr_mean)

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
        # Actually 1e-10 didn't work as great, so changed it to 1e-7...
        reconstr_loss = \
            -tf.reduce_sum(self.x * tf.log(1e-7 + self.x_reconstr_mean)
                           + (1 - self.x) * tf.log(1e-7 + 1 - self.x_reconstr_mean),
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

        self.optimizer = optimize_loss(
            self.cost, GLOBAL_STEP, learning_rate=LEARNING_RATE, optimizer='SGD',
            summaries=["gradients"])

        with tf.name_scope('metrics'):
            tf.summary.scalar('loss', self.cost)
            tf.summary.scalar('reconstr_loss', tf.reduce_mean(reconstr_loss))
            tf.summary.scalar('kl_loss', tf.reduce_mean(latent_loss))

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


def update_plot(i, epochs, batches, data, labels, scat, ax):
    ax.set_xlabel('Epoch: {}, batch: {}'.format(epochs[i] + 1, batches[i] + 1))
    scat.set_offsets(data[i])
    scat.set_color(labels[i])
    return scat,


def train(network_architecture, train_data, train_labels, plot_train=True,
          control_data=None, control_labels=None, plot_control=False):
    vae = VAE(network_architecture)
    n_samples = train_data.shape[0]
    n_batches = int(n_samples / BATCH_SIZE)  # Number of batches in 1 epoch.
    print('Training batches:', n_batches)

    if plot_train:
        plot_data, plot_labels, plot_epochs, plot_batches = [], [], [], []
        fig, ax = plt.subplots()
        ax.set_xlim((-5, 5))
        ax.set_ylim((-5, 5))
        scat = ax.scatter([], [], s=1)
    else:
        plot_data, plot_labels, plot_epochs, plot_batches = None, None, None, None
        fig, ax = None, None
        scat = None

    # Training cycle.
    for epoch in range(NUM_EPOCHS):
        avg_cost = 0.  # Avg cost per all batches processed so far.

        # Loop over all batches.
        for i in range(n_batches):
            batch_xs = train_data[(i * BATCH_SIZE):(i * BATCH_SIZE + BATCH_SIZE), :]

            # Fit training using batch data.
            summary, opt, cost = vae.partial_fit(batch_xs)
            print("Epoch {} / batch {} / cost {}".format(epoch + 1, i, cost))

            vae.train_writer.add_summary(summary, epoch * n_batches + i)

            # Compute average loss.
            avg_cost += cost / n_samples * BATCH_SIZE

            # Plot latent space on all data seen so far.
            if plot_train:
                z_mu = vae.encode(train_data[:(i * BATCH_SIZE + BATCH_SIZE), :])
                plot_data.append(z_mu)
                colours = [COLOURS[l] for l in train_labels[:(i * BATCH_SIZE + BATCH_SIZE)]]
                plot_labels.append(colours)
                # It is a bit ugly to accumulate epoch/batch idxs like this, but...
                plot_epochs.append(epoch)
                plot_batches.append(i)

        print("Epoch:", '%04d' % (epoch + 1),
              "avg seen batch cost=", "{:.9f}".format(avg_cost))

    if plot_train:
        ani = animation.FuncAnimation(fig, update_plot,
                                      frames=np.arange(len(plot_data)),
                                      fargs=(plot_epochs, plot_batches, plot_data, plot_labels, scat, ax),
                                      interval=300)
        ani.save('latent.gif', dpi=80, writer='imagemagick')
        plt.gcf().clear()

    return vae


def plot_example_reconstr(vae, x_sample):
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

    class_0_train = mnist_tf_data.get_class_data(1, train=True)
    class_1_train = mnist_tf_data.get_class_data(0, train=True)
    # N batches of 0, then N batches of 1, then 0.
    num_entries = BATCH_SIZE * 20
    train_data = np.vstack((class_0_train[:num_entries, :],
                            class_1_train[:num_entries, :],
                            class_0_train[num_entries:(2 * num_entries), :]))
    train_data = train_data.reshape((train_data.shape[0], -1))
    train_data = train_data / 255  # An attempt at normalization.
    train_labels = np.hstack((np.ones(shape=(num_entries,), dtype=np.int32),
                              np.zeros(shape=(num_entries,), dtype=np.int32),
                              np.ones(shape=(num_entries,), dtype=np.int32)))
    train_labels = train_labels[:train_data.shape[0]]

    # Build the control and the test data sets.
    class_0_test = mnist_tf_data.get_class_data(0, train=False)
    class_1_test = mnist_tf_data.get_class_data(1, train=False)
    test_data = np.vstack((class_0_test[BATCH_SIZE:(2 * BATCH_SIZE), :],
                           class_1_test[BATCH_SIZE:(2 * BATCH_SIZE), :]))
    test_data = test_data.reshape((test_data.shape[0], -1))
    test_data = test_data / 255
    np.random.shuffle(test_data)

    # Shuffle to make sure it works when shuffled.
    # idx = np.arange(train_data.shape[0])
    # np.random.shuffle(idx)
    # train_data = train_data[idx, :]
    # train_labels = train_labels[idx]

    vae = train(network_architecture, train_data, train_labels)
    plot_example_reconstr(vae, test_data[:BATCH_SIZE, :])


if __name__ == '__main__':
    main()
