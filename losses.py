#coding: utf-8
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets

def loss_model(x):
    logits, endpoints_dict = nets.vgg.vgg_16(x, spatial_squeeze=False)
    return logits, endpoints_dict

def content_loss(endpoints_dict, cont_layers):
    cont_loss = 0
    for layer in cont_layers:
        g_img, c_img, _ = tf.split(endpoints_dict[layer], 3, 0)
        cont_loss += tf.nn.l2_loss(g_img - c_img) * 2 / tf.to_float(tf.size(g_img))
    return cont_loss

def gram(layer):
    shape = tf.shape(layer)
    num_images = shape[0]
    width = shape[1]
    height = shape[2]
    num_filters = shape[3]
    features = tf.reshape(layer, tf.stack([num_images, -1, num_filters]))
    grams = tf.matmul(features, features, transpose_a=True) / tf.to_float(width * height * num_filters)

    return grams
    
def style_loss(endpoints_dict, style_layers):
    style_loss = 0
    for layer in style_layers:
        _, g_img, s_img = tf.split(endpoints_dict[layer], 3, 0)
        style_loss += tf.nn.l2_loss(gram(g_img) - gram(s_img)) * 2 / tf.to_float(tf.size(g_img))
    return style_loss