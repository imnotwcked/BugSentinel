import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import pickle

# Load the MNIST data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Parameters
learning_rate = 0.001
training_epochs = 10
batch_size = 100

# Network Parameters
n_classes = 10  # total classes (0-9 digits)
dropout = 0.5  # Dropout, probability to keep units

# TensorFlow Graph Input
X = tf.placeholder("float", [None, 28, 28, 1])  # MNIST data input (28x28 pixels)
Y = tf.placeholder("float", [None, n_classes])
keep_prob = tf.placeholder(tf.float32)  # Dropout (keep probability)

# Create the model
def conv_net(x):
    # Convolution Layer 1
    conv1 = tf.layers.conv2d(x, filters=32, kernel_size=[5, 5], padding='same', activation=tf.nn.relu, name='conv1')
    conv1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=2)

    # Convolution Layer 2
    conv2 = tf.layers.conv2d(conv1, filters=32, kernel_size=[5, 5], padding='same', activation=tf.nn.relu, name='conv2')
    conv2 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=2)

    # Flatten the layer
    fc1 = tf.reshape(conv2, [-1, 7 * 7 * 32])

    # Fully Connected Layer 1
    fc1 = tf.layers.dense(fc1, 128, name='fc1')
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, keep_prob)

    # Fully Connected Layer 2 (Output Layer)
    out_layer = tf.layers.dense(fc1, n_classes, name='out')
    return out_layer

# Construct the model
logits = conv_net(X)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables
init = tf.global_variables_initializer()

# Training the model
with tf.Session() as sess:
    sess.run(init)

    # Reshape MNIST data for training and testing
    train_images = mnist.train.images.reshape(-1, 28, 28, 1)
    test_images = mnist.test.images.reshape(-1, 28, 28, 1)

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Reshape batch_x to match input format
            batch_x = batch_x.reshape(-1, 28, 28, 1)
            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y, keep_prob: dropout})
            avg_cost += c / total_batch

        print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(avg_cost))

    print("Optimization Finished!")

    # Test model
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: test_images, Y: mnist.test.labels, keep_prob: 1.0}))

    # Save model weights and biases
    # Get weights and biases for conv and fc layers
    conv1_weights = tf.get_default_graph().get_tensor_by_name('conv1/kernel:0')
    conv2_weights = tf.get_default_graph().get_tensor_by_name('conv2/kernel:0')
    fc1_weights = tf.get_default_graph().get_tensor_by_name('fc1/kernel:0')
    out_weights = tf.get_default_graph().get_tensor_by_name('out/kernel:0')
    file_path = "./params/"
    # Save weights
    with open(file_path+"conv1_weights.param", "wb") as f:
        pickle.dump(sess.run(conv1_weights), f)
    with open(file_path+"conv2_weights.param", "wb") as f:
        pickle.dump(sess.run(conv2_weights), f)
    with open(file_path+"fc1_weights.param", "wb") as f:
        pickle.dump(sess.run(fc1_weights), f)
    with open(file_path+"out_weights.param", "wb") as f:
        pickle.dump(sess.run(out_weights), f)

    # Save biases (if needed)
    conv1_biases = tf.get_default_graph().get_tensor_by_name('conv1/bias:0')
    conv2_biases = tf.get_default_graph().get_tensor_by_name('conv2/bias:0')
    fc1_biases = tf.get_default_graph().get_tensor_by_name('fc1/bias:0')
    out_biases = tf.get_default_graph().get_tensor_by_name('out/bias:0')

    with open(file_path+"conv1_biases.param", "wb") as f:
        pickle.dump(sess.run(conv1_biases), f)
    with open(file_path+"conv2_biases.param", "wb") as f:
        pickle.dump(sess.run(conv2_biases), f)
    with open(file_path+"fc1_biases.param", "wb") as f:
        pickle.dump(sess.run(fc1_biases), f)
    with open(file_path+"out_biases.param", "wb") as f:
        pickle.dump(sess.run(out_biases), f)