#coding: utf-8
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from scipy.misc import imread, imresize, imsave

import argparse
import os

import model
import reader
import losses

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--content_path', default = 'img/src.jpg', 
                        help = 'Path to the content image to be transferred')
    parser.add_argument('-m', '--model_path', default = 'model/trained_model/wave/model.ckpt-20001',
                        help = 'Path to the pretrained model')
    parser.add_argument('-s', '--save_path', default = 'img/',
                        help = 'Path to save the style-transferred image')
    return parser.parse_args()

def main():
    args = build_parser()
    src_img, src_shape = load_image(args.content_path, train=False)

    with tf.Graph().as_default():
        con_img = tf.placeholder(tf.float32, 
                                    [None, src_shape[0], src_shape[1], src_shape[2]])
        gen_img = tf.squeeze(model.net_gen(con_img, training=False), [0])
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            saver.restore(sess, args.model_path)
            styled_image = sess.run(gen_img, feed_dict={con_img: src_img})

            savepath = os.path.join(args.save_path, 'gen.jpg')
            # Check if savepath's exist
            # If not, create it
            if not os.path.exists(savepath):
                if not os.path.exists(args.save_path):
                    os.makedirs(args.save_path)
                os.mknod(savepath)

            imsave(os.path.join(args.save_path, 'gen.jpg'), np.squeeze(styled_image))

if __name__ == '__main__':
    main()
