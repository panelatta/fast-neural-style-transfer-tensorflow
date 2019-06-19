#!/bin/sh

if [ ! -d "vggmodel" ]; then
    echo "*** Downloading pretrained VGG-16 model ***"
    sudo mkdir vggmodel
    sudo wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
    sudo tar -xzvf vgg_16_2016_08_28.tar.gz
    sudo cp vgg_16.ckpt vggmodel/
    sudo rm -f vgg_16_2016_08_28.tar.gz
    echo "*** VGG-16 model downloading completed. ***"
else
    echo "*** VGG-16 model has already been found. ***"
fi

if [ ! -d "train2014" ]; then
    echo "*** Downloading COCO dataset ***"
    sudo wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip
    sudo unzip train2014.zip
    sudo rm -f train2014.zip
    echo "*** COCO dataset downloading completed. ***"
else
    echo "*** COCO dataset has already been found. ***"
fi
