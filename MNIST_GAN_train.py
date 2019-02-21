#Import the libraries we will need.
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
import os
import scipy.misc
import scipy
import datetime
# %matplotlib inline

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")


def discriminator(images, reuse=False): # 32x32
    if (reuse):
        tf.get_variable_scope().reuse_variables()

    # First convolutional and pool layers
    # This finds 32 different 5 x 5 pixel features
    d_w1 = tf.get_variable('d_w1', [5, 5, 1, 32], initializer=tf.truncated_normal_initializer(stddev=0.02))
    d_b1 = tf.get_variable('d_b1', [32], initializer=tf.constant_initializer(0))
    d1 = tf.nn.conv2d(input=images, filter=d_w1, strides=[1, 1, 1, 1], padding='SAME')
    d1 = d1 + d_b1
    d1 = tf.nn.relu(d1)
    d1 = tf.nn.avg_pool(d1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') # 16x16

    # Second convolutional and pool layers
    # This finds 64 different 5 x 5 pixel features
    d_w2 = tf.get_variable('d_w2', [5, 5, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.02))
    d_b2 = tf.get_variable('d_b2', [64], initializer=tf.constant_initializer(0))
    d2 = tf.nn.conv2d(input=d1, filter=d_w2, strides=[1, 1, 1, 1], padding='SAME')
    d2 = d2 + d_b2
    d2 = tf.nn.relu(d2)
    d2 = tf.nn.avg_pool(d2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') # 8x8

    # First fully connected layer
    d_w3 = tf.get_variable('d_w3', [8 * 8 * 64, 1024], initializer=tf.truncated_normal_initializer(stddev=0.02))
    d_b3 = tf.get_variable('d_b3', [1024], initializer=tf.constant_initializer(0))
    d3 = tf.reshape(d2, [-1, 8 * 8 * 64])
    d3 = tf.matmul(d3, d_w3)
    d3 = d3 + d_b3
    d3 = tf.nn.relu(d3)

    # Second fully connected layer
    d_w4 = tf.get_variable('d_w4', [1024, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
    d_b4 = tf.get_variable('d_b4', [1], initializer=tf.constant_initializer(0))
    d4 = tf.matmul(d3, d_w4) + d_b4

    # d4 contains unscaled values [batch_size, 1]
    return d4

# def generator(z, batch_size, z_dim, reuse=False):
#     g_w1 = tf.get_variable('g_w1', [z_dim, 3136], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
#     g_b1 = tf.get_variable('g_b1', [3136], initializer=tf.truncated_normal_initializer(stddev=0.02))
#     g1 = tf.matmul(z, g_w1) + g_b1
#     g1 = tf.reshape(g1, [-1, 56, 56, 1])
#     g1 = tf.contrib.layers.batch_norm(g1, epsilon=1e-5, scope='bn1')
#     g1 = tf.nn.relu(g1)
#
#     # Generate 50 features
#     g_w2 = tf.get_variable('g_w2', [3, 3, 1, z_dim/2], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
#     g_b2 = tf.get_variable('g_b2', [z_dim/2], initializer=tf.truncated_normal_initializer(stddev=0.02))
#     g2 = tf.nn.conv2d(g1, g_w2, strides=[1, 2, 2, 1], padding='SAME')
#     g2 = g2 + g_b2
#     g2 = tf.contrib.layers.batch_norm(g2, epsilon=1e-5, scope='bn2')
#     g2 = tf.nn.relu(g2)
#     g2 = tf.image.resize_images(g2, [56, 56])
#
#     # Generate 25 features
#     g_w3 = tf.get_variable('g_w3', [3, 3, z_dim/2, z_dim/4], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
#     g_b3 = tf.get_variable('g_b3', [z_dim/4], initializer=tf.truncated_normal_initializer(stddev=0.02))
#     g3 = tf.nn.conv2d(g2, g_w3, strides=[1, 2, 2, 1], padding='SAME')
#     g3 = g3 + g_b3
#     g3 = tf.contrib.layers.batch_norm(g3, epsilon=1e-5, scope='bn3')
#     g3 = tf.nn.relu(g3)
#     g3 = tf.image.resize_images(g3, [56, 56])
#
#     # Final convolution with one output channel
#     g_w4 = tf.get_variable('g_w4', [1, 1, z_dim/4, 1], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
#     g_b4 = tf.get_variable('g_b4', [1], initializer=tf.truncated_normal_initializer(stddev=0.02))
#     g4 = tf.nn.conv2d(g3, g_w4, strides=[1, 2, 2, 1], padding='SAME')
#     g4 = g4 + g_b4
#     g4 = tf.sigmoid(g4)
#
#     # Dimensions of g4: batch_size x 28 x 28 x 1
#     return g4

#This initializaer is used to initialize all the weights of the network.
initializer = tf.truncated_normal_initializer(stddev=0.02)
def generator(z, reuse=False):
    with tf.variable_scope('generator') as scope:
        if (reuse):
            tf.get_variable_scope().reuse_variables()

        zP = slim.fully_connected(z, 4 * 4 * 256, normalizer_fn=slim.batch_norm,
                                  activation_fn=tf.nn.relu, scope='g_project', weights_initializer=initializer)
        zCon = tf.reshape(zP, [-1, 4, 4, 256])

        gen1 = slim.convolution2d_transpose(
            zCon, num_outputs=64, kernel_size=[5, 5], stride=[2, 2],
            padding="SAME", normalizer_fn=slim.batch_norm,
            activation_fn=tf.nn.relu, scope='g_conv1', weights_initializer=initializer) #8x8

        gen2 = slim.convolution2d_transpose(
            gen1, num_outputs=32, kernel_size=[5, 5], stride=[2, 2],
            padding="SAME", normalizer_fn=slim.batch_norm,
            activation_fn=tf.nn.relu, scope='g_conv2', weights_initializer=initializer) # 16x16

        gen3 = slim.convolution2d_transpose(
            gen2, num_outputs=16, kernel_size=[5, 5], stride=[2, 2],
            padding="SAME", normalizer_fn=slim.batch_norm,
            activation_fn=tf.nn.relu, scope='g_conv3', weights_initializer=initializer) # 32x32

        g_out = slim.convolution2d_transpose(
            gen3, num_outputs=1, kernel_size=[32, 32], padding="SAME",
            biases_initializer=None, activation_fn=tf.nn.tanh,
            scope='g_out', weights_initializer=initializer)

    return g_out

z_dimensions = 100
# z_placeholder = tf.placeholder(tf.float32, [None, z_dimensions])

# generated_image_output = generator(z_placeholder, 1, z_dimensions)
# z_batch = np.random.normal(0, 1, [1, z_dimensions])
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     generated_image = sess.run(generated_image_output,
#                                 feed_dict={z_placeholder: z_batch})
#     generated_image = generated_image.reshape([28, 28])
#     plt.imshow(generated_image, cmap='Greys')
#
tf.reset_default_graph()
sess = tf.Session()
batch_size = 8

z_placeholder = tf.placeholder(tf.float32, [None, z_dimensions], name='z_placeholder')
# z_placeholder is for feeding input noise to the generator

x_placeholder = tf.placeholder(tf.float32, shape = [None,32,32,1], name='x_placeholder')
# x_placeholder is for feeding input images to the discriminator

Gz = generator(z_placeholder)
# Gz = generator(z_placeholder, batch_size, z_dimensions)
# Gz holds the generated images

Dx = discriminator(x_placeholder)
# Dx will hold discriminator prediction probabilities
# for the real MNIST images

Dg = discriminator(Gz, reuse=True)
# Dg will hold discriminator prediction probabilities for generated images

d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dx, labels=tf.ones_like(Dx)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.zeros_like(Dg)))

g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.ones_like(Dg)))

tvars = tf.trainable_variables()

d_vars = [var for var in tvars if 'd_' in var.name]
g_vars = [var for var in tvars if 'g_' in var.name]

# Train the discriminator
with tf.variable_scope('fake_real', reuse=tf.AUTO_REUSE):
    d_trainer_fake = tf.train.AdamOptimizer(0.0003).minimize(d_loss_fake, var_list=d_vars)
    d_trainer_real = tf.train.AdamOptimizer(0.0003).minimize(d_loss_real, var_list=d_vars)

# Train the generator
with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
    g_trainer = tf.train.AdamOptimizer(0.0001).minimize(g_loss, var_list=g_vars)

tf.summary.scalar('Generator_loss', g_loss)
tf.summary.scalar('Discriminator_loss_real', d_loss_real)
tf.summary.scalar('Discriminator_loss_fake', d_loss_fake)

# images_for_tensorboard = generator(z_placeholder, batch_size, z_dimensions)
# tf.summary.image('Generated_images', images_for_tensorboard, 5)
merged = tf.summary.merge_all()
tensorboard_logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(tensorboard_logdir, sess.graph)

sess.run(tf.global_variables_initializer())

# Add ops to save and restore all the variables.
saver = tf.train.Saver(max_to_keep=10)
checkpoint_logdir = "./checkpoint/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/model.ckpt"
os.makedirs(checkpoint_logdir)

# Pre-train discriminator
for i in range(1000):
    z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
    # real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
    real_image_batch = mnist.train.next_batch(batch_size)[0]
    real_image_batch = (np.reshape(real_image_batch,[batch_size,28,28,1]) - 0.5) * 2.0 #Transform it to be between -1 and 1
    real_image_batch = np.lib.pad(real_image_batch, ((0,0),(2,2),(2,2),(0,0)),'constant', constant_values=(-1, -1)) #Pad the images so the are 32x32
    _, __, dLossReal, dLossFake = sess.run([d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake],
                                           {x_placeholder: real_image_batch, z_placeholder: z_batch})

    # if(i % 100 == 0):
    #     print("dLossReal:", dLossReal, "dLossFake:", dLossFake)

# Train generator and discriminator together
for i in range(1000000):
    # real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
    real_image_batch = mnist.train.next_batch(batch_size)[0]
    real_image_batch = (np.reshape(real_image_batch,[batch_size,28,28,1]) - 0.5) * 2.0 #Transform it to be between -1 and 1
    real_image_batch = np.lib.pad(real_image_batch, ((0,0),(2,2),(2,2),(0,0)),'constant', constant_values=(-1, -1)) #Pad the images so the are 32x32
    z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])

    # Train discriminator on both real and fake images
    _, __, dLossReal, dLossFake = sess.run([d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake],
                                           {x_placeholder: real_image_batch, z_placeholder: z_batch})

    # Train generator twice
    z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
    _ = sess.run(g_trainer, feed_dict={z_placeholder: z_batch})

    z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
    _ = sess.run(g_trainer, feed_dict={z_placeholder: z_batch})

    if i % 100 == 0:
        # Update TensorBoard with summary statistics
        z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
        summary = sess.run(merged, {z_placeholder: z_batch, x_placeholder: real_image_batch})
        writer.add_summary(summary, i)

    if i % 1000 == 0:
        # Every 1000 iterations, show a generated image
        print("Iteration:", i, "at", datetime.datetime.now())
        z_batch = np.random.normal(0, 1, size=[1, z_dimensions])
        generated_images = generator(z_placeholder, reuse=True)
        images = sess.run(generated_images, {z_placeholder: z_batch})
        plt.imshow(images[0].reshape([32, 32]), cmap='Greys')
        plt.show()

    if i % 200000 == 0:
        # Save the variables to disk.
        save_path = saver.save(sess, checkpoint_logdir, global_step=i)
        print("Model saved in path: %s" % save_path)