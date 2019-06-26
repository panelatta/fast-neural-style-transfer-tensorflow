#coding: utf-8
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.misc as misc
import numpy as np

import os
import argparse
import yaml
import time

import model
import reader

# Receive arguments from command
def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--conf", default = "conf/wave.yml", help = "The path to the conf file")
    args = parser.parse_args()
    return args

def main():
    with tf.Graph().as_default():        
        # Load Dataset
        style_img_batch, _ = reader.load_image(imgarg.style_image_path, 
                                               imgarg.batch_size, 
                                               imgarg.image_size, 
                                               True)

        # Hold place for images
        image = tf.placeholder(tf.float32, [None, 
                                            imgarg.image_size, 
                                            imgarg.image_size, 
                                            imgarg.channel_num])
        style_image = tf.placeholder(tf.float32, [None, 
                                                  imgarg.image_size, 
                                                  imgarg.image_size, 
                                                  imgarg.channel_num])
        
        # Extract feature
        _, endpoints_dict = losses.loss_model(tf.concat([image, 
                                                         model.net_gen(image, training=True), 
                                                         style_image], 0))
        with tf.Session() as sess:
            # Building Losses
            cont_loss = losses.content_loss(endpoints_dict, imgarg.content_layers)
            style_loss = losses.style_loss(endpoints_dict, imgarg.style_layers)
            total_loss = imgarg.style_weight * style_loss + imgarg.content_weight * cont_loss

            # Preprocess before training
            var_for_train = slim.get_variables_to_restore(include=['net_gen'])
            optimizer = tf.train.AdamOptimizer(imgarg.learning_rate).apply_gradients(
                grads_and_vars=list(zip(tf.gradients(total_loss, var_for_train), var_for_train)))
            saver = tf.train.Saver(slim.get_variables_to_restore(include=['net_gen']), max_to_keep=100)
            restore_saver = tf.train.Saver(slim.get_variables_to_restore(include=[imgarg.loss_model]))
        
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

            # Restore vars
            if os.path.exists(imgarg.model_path):
                restore_saver.restore(sess, imgarg.model_path)

            # Start training
            start_time = time.time()
            prev_time = start_time
            for epoch in range(imgarg.epoch_num):
                imgarg.batch_index = 0
                iters = imgarg.data_size // imgarg.batch_size
                for step in range(iters):
                    step_time = time.time() - prev_time
                    tot_time = time.time() - start_time
                    prev_time = time.time()

                    img_batch = reader.get_images(imgarg.train_image_path, img_paths)
                    _, tot_loss, sty_loss, con_loss, = sess.run([optimizer, total_loss, style_loss, cont_loss], 
                                                feed_dict={image: img_batch, style_image: style_img_batch})
                    glob_step = epoch * iters + step + 1

                    # Logging
                    if step % 10 == 0:
                        tf.logging.info('Epoch: %d, Step: %d/%d, Total time: %f sec(s), Time/Step: %f sec(s) --- %.2f%%' 
                              % (epoch + 1, step, iters, tot_time, step_time, 100 * (glob_step / (imgarg.epoch_num * iters))))
                        tf.logging.info('Total loss: %f, Content loss: %f, Style loss: %f'
                              % (tot_loss, con_loss, sty_loss))

                    # Save checkpoints
                    if step % 1000 == 0:   
                        saver.save(sess, os.path.join(imgarg.train_check_point, 'model.ckpt'), 
                                   global_step=glob_step)
                        tf.logging.info('Saving checkpoints %d ...' % (glob_step))

            saver.save(sess, os.path.join(imgarg.train_check_point, 'model.ckpt-done'))
            tf.logging.info('Training completed --- Epoch limit achieved.')

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    
    # Read arguments
    args = build_parser()

    # Load in config file of style
    read_conf_file(args.conf)

    # Load in list of paths of dataset files
    img_paths = [os.path.join(imgarg.train_image_path, f) for f in os.listdir(imgarg.train_image_path)
                 if os.path.isfile(os.path.join(imgarg.train_image_path, f))]
    img_paths = sorted(img_paths)

    main()


