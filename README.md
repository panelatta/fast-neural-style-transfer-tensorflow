## Fast Neural Style Transfer Tensorflow

A tensorflow implement of [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155) , referenced [OlavHN/fast-neural-style](https://github.com/OlavHN/fast-neural-style), [hzy46/fast-neural-style-tensorflow](https://github.com/hzy46/fast-neural-style-tensorflow) and [lengstrom/fast-style-transfer](https://github.com/lengstrom/fast-style-transfer).

### Requirement

- Python 2.7.x
- Tensorflow >= 1.4

Also make sure that you've installed numpy, scipy and pyyaml:

```python
pip install numpy
pip install scipy
pip install pyyaml
```

The code works well on Ubuntu 18.04, Python 2.7.15 and tensorflow 1.13.1.

### Usage

You should download [Pretrained VGG-16 Model](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz) of tensorflow-slim and unpack it to folder `model/`.

```bash
cd <this repo>/model
sudo wget -c http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
sudo tar -xzvf vgg_16_2016_08_28.tar.gz
sudo rm vgg_16_2016_08_28.tar.gz
```

Then, you should download [COCO2014 Dataset](http://msvocds.blob.core.windows.net/coco2014/train2014.zip) , unpack it, and create a symbol link to the folder `train2014/` .

```bash
sudo wget -c http://msvocds.blob.core.windows.net/coco2014/train2014.zip
sudo unzip train2014.zip
(If you've not installed unzip, use 'sudo apt install unzip' to install it)
cd <this repo>
sudo ln -s <path to train2014/> train2014
```

The repo has included a pretrained `wave` model checkpoint in `model/trained_model/wave/model.ckpt-20001`. To transfer images with a trained model, type that

```bash
cd <this repo>
python eval.py -c <path to source image> -m <path to model> -s <path to save result>
```

Default value of the para `-m` is defined as the pretrained model, so you can omit it.

To get more pretrained model you can download them [here](https://drive.google.com/open?id=1O8Hicm5PCLPeuS0OHCdIe7HSLOjaZBWG) .

The folder `conf/` has included some defined config file of different styles. As an example, to train the model `cubist` , type that

```bash
cd <this repo>
python train.py -c conf/cubist.yml
```

Then checkpoints could be found at `model/trained_model/<folder of your style>` .