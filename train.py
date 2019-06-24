#coding: utf-8
from __future__ import print_function
from __future__ import division
import tensorflow as tf
import tensorflow.contrib.slim as slim
import model
import data_process
import numpy as np
from os.path import join
from scipy.misc import imsave
import argparse

# BATCH_SIZE = 4
# IMAGE_SIZE = 256
# CHANNEL_NUM = 3
# CONTENT_LAYERS = ["vgg_16/conv3/conv3_3"]
# STYLE_LAYERS = ["vgg_16/conv1/conv1_2", "vgg_16/conv2/conv2_2",
#                 "vgg_16/conv3/conv3_3", "vgg_16/conv4/conv4_3"]

# Style weight and content weight for predefined styles
# PRESET_STYLES = {'candy' : (50.0, 1.0),
#                  'cubist' : (180.0, 1.0),
#                  'feathers' : (220.0, 1.0),
#                  'mosaic' : (100.0, 1.0),
#                  'scream' : (250.0, 1.0),
#                  'udnie' : (200.0, 1.0),
#                  'wave' : (220.0, 1.0)}
# STYLE_WEIGHT = PRESET_STYLES['wave'][0]
# CONTENT_WEIGHT = PRESET_STYLES['wave'][1]

# MODEL_PATH = 'model/vgg_16.ckpt'
# STYLE_IMAGE_PATH = 'style_images/wave.jpg'
# TRAIN_IMAGE_PATH = 'train2014/'
# TRAIN_CHECK_POINT = 'model/trained_model/wave/'
# TEST_IMAGE_PATH = 'test/test.jpg'
# IMAGE_SAVE_PATH = 'image/'
# CHECK_POINT_PATH = 'log/'
# LEARNING_RATE = 1e-3
# EPOCH_NUM = 1
# DATA_SIZE = 82783

with tf.Graph().as_default():
    image = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, CHANNEL_NUM])
    style_image = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, CHANNEL_NUM])
    #Obtain image features
    generated_image = model.generator(image, training=True)
    squeezed_generated_image = tf.image.encode_jpeg(tf.cast(tf.squeeze(generated_image, [0]), tf.uint8))
    #Obtain loss model layers for image, generated_image and style image.
    _, endpoints_mixed = model.loss_model(tf.concat([image, generated_image, style_image], 0))
    variables_to_restore = slim.get_variables_to_restore(include=['vgg_16'])
    restorer = tf.train.Saver(variables_to_restore)
    #Content loss
    content_loss = model.content_loss(endpoints_mixed, CONTENT_LAYERS)
    #Style loss
    style_loss = model.style_loss(endpoints_mixed, STYLE_LAYERS)

    loss = STYLE_WEIGHT * style_loss + CONTENT_WEIGHT * content_loss
    tf.summary.scalar('losses/content_loss', CONTENT_WEIGHT * content_loss)
    tf.summary.scalar('losses/style_loss', STYLE_WEIGHT * style_loss)
    tf.summary.scalar('losses/loss', loss)
    tf.summary.image('generated', generated_image)
    tf.summary.image('origin', image)    
    summary = tf.summary.merge_all()
	#Only train generator network
    variables_for_training = slim.get_variables_to_restore(include=['generator'])
    gradients = tf.gradients(loss, variables_for_training)
    grad_and_var = list(zip(gradients, variables_for_training))

    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    opt_op = optimizer.apply_gradients(grads_and_vars=grad_and_var)

    #Only save parameters of generator model
    variables_to_save = slim.get_variables_to_restore(include=['generator'])
    saver = tf.train.Saver(variables_to_save, max_to_keep=100)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(CHECK_POINT_PATH, sess.graph)
        # Restore variables from disk.
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        restorer.restore(sess, MODEL_PATH)
        #Obtain style image batch
        style_image_batch = data_process.read_style_image(STYLE_IMAGE_PATH, BATCH_SIZE)
        step = 0
		
		img_paths = []
        for epoch in range(EPOCH_NUM):
            batch_index = 0
            for i in range(DATA_SIZE // BATCH_SIZE):
                image_batch, batch_index, img_paths = data_process.read_batches(TRAIN_IMAGE_PATH, img_paths, batch_index, BATCH_SIZE)
                _, batch_ls, style_ls, content_ls, summary_str = sess.run([opt_op, loss, style_loss, content_loss, summary], feed_dict={image: image_batch,
                                                                  style_image: style_image_batch})
                step += 1
                if i % 10 == 0:
                    print('Epoch %d, Batch %d of %d, loss is %.3f, style loss is %.3f, content loss is %.3f'%(epoch + 1, i, DATA_SIZE // BATCH_SIZE, batch_ls, 220 * style_ls, content_ls))
                    train_writer.add_summary(summary_str, step)
                    test_image = data_process.read_test_image(TEST_IMAGE_PATH)
                    styled_image = sess.run(squeezed_generated_image, feed_dict={image: test_image})
                    #imsave(join(IMAGE_SAVE_PATH, 'epoch' + str(epoch + 1) + '.jpg'), styled_image)
                    with open('training_image/res.jpg', 'wb') as img_s:
                         img_s.write(styled_image)
                if i % 1000 == 0:   
            	    #save model parameters
                    saver.save(sess, join(TRAIN_CHECK_POINT, 'model.ckpt'), global_step=step)




