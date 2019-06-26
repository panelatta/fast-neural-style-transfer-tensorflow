#coding: utf-8
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import tensorflow.contrib.slim as slim
from scipy.misc import imsave
import numpy as np

import os
import argparse
import yaml

import model
import reader

# Receive arguments from command
def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--conf", default = "conf/wave.yml", help = "The path to the conf file")
    args = parser.parse_args()
    return args

def main():
    args = build_parser()
    read_conf_file(args.conf)

    # Load in list of paths of dataset files
    img_paths = [os.path.join(style_img_arg.TRAIN_IMAGE_PATH, f) for f in os.listdir(style_img_arg.TRAIN_IMAGE_PATH)
                 if os.path.isfile(os.path.join(style_img_arg.TRAIN_IMAGE_PATH, f))]
    img_paths = sorted(img_paths)

    with tf.Graph().as_default():
        image = tf.placeholder(tf.float32, [None, style_img_arg.IMAGE_SIZE, style_img_arg.IMAGE_SIZE, style_img_arg.CHANNEL_NUM])
        style_image = tf.placeholder(tf.float32, [None, style_img_arg.IMAGE_SIZE, style_img_arg.IMAGE_SIZE, style_img_arg.CHANNEL_NUM])
        output = model.net_gen(image, training=True)
        logits, endpoints_dict = losses.loss_model(tf.concat([image, output, style_image], 0))
        
        cont_loss = losses.content_loss(endpoints_dict, style_img_arg.CONTENT_LAYERS)
        style_loss = losses.style_loss(endpoints_dict, style_img_arg.STYLE_LAYERS)
        total_loss = style_img_arg.STYLE_WEIGHT * style_loss + style_img_arg.CONTENT_WEIGHT * cont_loss

        var_for_train = slim.get_variables_to_restore(include=['net_gen'])
        grad_and_var = list(zip(tf.gradients(total_loss, var_for_train), var_for_train))
        optimizer = tf.train.AdamOptimizer(style_img_arg.LEARNING_RATE).apply_gradients(grads_and_vars=grad_and_var)
        saver = tf.train.Saver(slim.get_variables_to_restore(include=['net_gen']), max_to_keep=100)
        var_to_res = slim.get_variables_to_restore(include=[style_img_arg.LOSS_MODEL])
        restorer = tf.train.Saver(var_to_res)
        
        with tf.Session() as sess:
            train_writer = tf.summary.FileWriter(style_img_arg.CHECK_POINT_PATH, sess.graph)

            # Restore from pretrained model
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            restorer.restore(sess, style_img_arg.MODEL_PATH)

            style_img_batch, _ = reader.load_image(style_img_arg.STYLE_IMAGE_PATH, style_img_arg.BATCH_SIZE, style_img_arg.IMAGE_SIZE, True)

            for epoch in range(style_img_arg.EPOCH_NUM):
                batch_index = 0
                batch_loop_time = style_img_arg.DATA_SIZE // style_img_arg.BATCH_SIZE
                for cnt in range(batch_loop_time):
                    img_batch, batch_index = reader.read_batches(style_img_arg.TRAIN_IMAGE_PATH, 
                                                                         img_paths, 
                                                                         batch_index, 
                                                                         style_img_arg.BATCH_SIZE, 
                                                                         style_img_arg.IMAGE_SIZE)
                    _, btc_ls, sty_ls, con_ls, sum_str = sess.run([optimizer, total_loss, style_loss, cont_loss, summary], 
                                                                  feed_dict={image: img_batch,
                                                                  style_image: style_img_batch})
                    if cnt % 10 == 0:
                        print('Epoch: %d, Batch %d of %d, Total loss: %.3f, Style loss: %.3f, Content loss: %.3f'%(epoch + 1, cnt, batch_loop_time, btc_ls, 220 * sty_ls, con_ls))
                        train_writer.add_summary(sum_str, epoch * batch_loop_time + cnt + 1)
                    if cnt % 1000 == 0:   
                        saver.save(sess, os.path.join(style_img_arg.TRAIN_CHECK_POINT, 'model.ckpt'), 
                                   global_step=epoch * batch_loop_time + cnt + 1)

if __name__ == '__main__':
    main()


