#coding: utf-8
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets

# Functions in model.py has referenced the implementation of https://github.com/hzy46/fast-neural-style-tensorflow.

def conv2d(x, filter_num, kernel_size, strides, name):
    return slim.conv2d(x, filter_num, [kernel_size, kernel_size], stride=strides, weights_regularizer=slim.l2_regularizer(1e-6), biases_regularizer=slim.l2_regularizer(1e-6), padding='SAME', activation_fn=None, scope=name)

def resize_conv2d(x, filters_num, kernel_size, strides, training, name):
    with tf.variable_scope(name):
        height = x.get_shape()[1].value if training else tf.shape(x)[1]
        width = x.get_shape()[2].value if training else tf.shape(x)[2]

        new_height = height * strides * 2
        new_width = width * strides * 2

        x_resized = tf.image.resize_images(x, [new_height, new_width], tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return conv2d(x_resized, filters_num, kernel_size, strides, 'conv1')

def instance_norm(x):
    epsilon = 1e-9
    mean, var = tf.nn.moments(x, [1,2], keep_dims=True)
    return tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(var, epsilon)))

def residual(x, filter_num, kernel_size, strides):
    conv1 = conv2d(x, filter_num, kernel_size, strides, 'conv1')
    conv2 = conv2d(tf.nn.relu(conv1), filter_num, kernel_size, strides, 'conv2')
    residual = x + conv2
    return residual

# Network to be trained
# Using slim-vgg to extract content & style
def net_gen(image, training):
    # Padding for reducing border effects
    image = tf.pad(image, [[0, 0], [10, 10], [10, 10], [0, 0]], mode='REFLECT')

    with tf.variable_scope('net_gen'):
        with tf.variable_scope('conv1'):
            conv1 = tf.nn.relu(instance_norm(conv2d(image, 32, 9, 1, 'conv1')))
        with tf.variable_scope('conv2'):
            conv2 = tf.nn.relu(instance_norm(conv2d(conv1, 64, 3, 2, 'conv2')))
        with tf.variable_scope('conv3'):
            conv3 = tf.nn.relu(instance_norm(conv2d(conv2, 128, 3, 2, 'conv3')))

        with tf.variable_scope('res1'):
            res1 = residual(conv3, 128, 3, 1, 'res1')
        with tf.variable_scope('res2'): 
            res2 = residual(res1, 128, 3, 1, 'res2')
        with tf.variable_scope('res3'):
            res3 = residual(res2, 128, 3, 1, 'res3')
        with tf.variable_scope('res4'):
            res4 = residual(res3, 128, 3, 1, 'res4')
        with tf.variable_scope('res5'):
            res5 = residual(res4, 128, 3, 1, 'res5')

        with tf.variable_scope('deconv1'):
            deconv1 = tf.nn.relu(instance_norm(resize_conv2d(res5, 64, 3, 1, training, 'deconv1')))
        with tf.variable_scope('deconv2'):
            deconv2 = tf.nn.relu(instance_norm(resize_conv2d(deconv1, 32, 3, 1, training, 'deconv2')))
        with tf.variable_scope('deconv3'):
            deconv3 = tf.nn.tanh(instance_norm(conv2d(deconv2, 3, 9, 1, 'deconv3')))

        # Revalue to [0, 255]
        out = (deconv3 + 1.0) * 127.5

        # Remove padding
        height = tf.shape(out)[1]
        width = tf.shape(out)[2]
	    out = tf.slice(y, [0, 10, 10, 0], tf.stack([-1, height - 20, width - 20, -1]))

        return out

