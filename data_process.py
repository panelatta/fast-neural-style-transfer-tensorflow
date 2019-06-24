from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from scipy.misc import imread, imresize
import numpy as np
import cv2
import os

COCO_image_path = 'train2014/'
IMG_SIZE = 256
CHANNEL_NUM = 3
#Mean pixel for all images in the set. It is provided by https://github.com/lengstrom/fast-style-transfer
MEAN_PIXEL = np.array([[123.68, 116.779, 103.939]])

#Read image function for style image or test image 
def read_style_image(filename, batch_size=64):
	img = imread(filename)
    img = imresize(img, (IMG_SIZE, IMG_SIZE, CHANNEL_NUM))
    img_batch = []
    for _ in range(batch_size):
        img_batch.append(img)
    img_batch = np.array(img_batch).astype(np.float32)
    return img_batch
	
def read_test_image(filename):
    img = imread(filename)
    img = imresize(img, (IMG_SIZE, IMG_SIZE, CHANNEL_NUM))
    img_batch = []
    img_batch.append(img)
    img_batch = np.array(img_batch).astype(np.float32)
    return img_batch

#Get batches from the data set. I know there are better methods in Tensorflow to get input data, but I just read them from the prepared list
#for simplicity.
def read_batches(filepath, img_paths, batch_index, batch_size=4):
    images = []
    image_indices = range(len(img_paths))
    count = 0
    for i in image_indices[batch_index: batch_index + batch_size]:
        if count >= batch_size:
            break
        count += 1
        dirname = img_paths[i].strip('\n').split()
        img = imread(dirname[0])
        img = imresize(img, (IMG_SIZE, IMG_SIZE, CHANNEL_NUM))
        #The only process for input images is subtracting mean value of each channel.
        if len(img.shape) < 3:
           timg = img
           img = np.zeros((IMG_SIZE, IMG_SIZE, CHANNEL_NUM)).astype(np.float32)
           img[:, :, 0] = timg -  MEAN_PIXEL[0, 0]
           img[:, :, 1] = timg -  MEAN_PIXEL[0, 1]
           img[:, :, 2] = timg -  MEAN_PIXEL[0, 2]
        else:
           img[:, :, 0] = img[:, :, 0] - MEAN_PIXEL[0, 0]
           img[:, :, 1] = img[:, :, 1] - MEAN_PIXEL[0, 1]
           img[:, :, 2] = img[:, :, 2] - MEAN_PIXEL[0, 2]
        images.append(img)

    images_np = np.array(images).astype(np.float32)
    batch_index = batch_index + batch_size
    return images_np, batch_index
