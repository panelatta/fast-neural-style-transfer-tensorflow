#coding: utf-8
from __future__ import division
from __future__ import print_function

import os

from scipy.misc import imread, imresize
import numpy as np

#Mean pixel for all images in the set. It is provided by https://github.com/lengstrom/fast-style-transfer
MEAN_PIXEL = np.array([[123.68, 116.779, 103.939]])

class style_img_arg:
    style_image_path = None
    model_path = None
    train_image_path = None
    content_weight = None
    style_weight = None
    image_size = None
    batch_size = None
    epoch_num = None
    loss_model = None
    content_layers = None
    style_layers = None
    channel_num = None
    train_check_point = None
    test_image_path = None
    image_save_path = None
    check_point_path = None
    learning_rate = None
    data_size = None

    def __init__(self):
        pass

def load_image(filename, bat_size=64, img_size=256, train=True):
    img = imresize(imread(filename), (img_size, img_size, 3))

    img_batch = []
    if not train:
        bat_size = 1
    for _ in range(bat_size):
        img_batch.append(img)
    img_batch = np.array(img_batch).astype(np.float32)

    return img_batch, img.shape
	

def read_conf_file(filename):
    with open(filename) as f :
        options = **f.yaml.load(f)
    style_img_arg.__dict__.update(options)
    print('Style config file ' + filename + ' loaded.')

#Get batches from the data set.
def read_batches(filepath, img_paths, bat_idx, bat_size=4, img_size=256):
    img_list = []
    for x in img_paths[bat_idx: bat_idx + bat_size]:
        img = imresize(imread(x), (img_size, img_size, style_img_arg.CHANNEL_NUM))

        # Substracting mean value of each channel from input images
        for channels in range(style_img_arg.CHANNEL_NUM):
            img[:, :, channels] -= MEAN_PIXEL[0, channels]

        img_list.append(img)

    images = np.array(images).astype(np.float32)
    bat_idx += bat_size

    return images, bat_idx