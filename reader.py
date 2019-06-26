#coding: utf-8
from __future__ import division
from __future__ import print_function

import scipy.misc as misc
import numpy as np

import os

class imgarg:
    MEAN_PIXEL = np.array([[123.68, 116.779, 103.939]])
    batch_index = None
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

def to_ndarray(x):
    return np.array(x).astype(np.float32)

def read_conf_file(filename):
    with open(filename) as f :
        options = **f.yaml.load(f)
    imgarg.__dict__.update(options)

def load_image(filename, bat_size=64, img_size=256, train=True):
    img = misc.imresize(misc.imread(filename), (img_size, img_size, imgarg.channel_num))

    batch = []
    if not train:
        bat_size = 1
    for _ in range(bat_size):
        img_batch.append(img)

    return to_ndarray(batch), img.shape

def get_images(filepath, img_paths):
    batch = []
    bat_index = imgarg.batch_index
    bat_size = imgarg.batch_size
    img_size = imgarg.image_size

    for index in range(bat_index, bat_index + bat_size):
        img = misc.imresize(misc.imread(img_paths[index]), (img_size, img_size, imgarg.channel_num))
        for channels in range(imgarg.channel_num):
            img[:, :, channels] -= imgarg.MEAN_PIXEL[0, channels]
        batch.append(img)

    return to_ndarray(batch), bat_index + bat_size