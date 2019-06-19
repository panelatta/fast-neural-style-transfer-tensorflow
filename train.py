from __future__ import print_function
from __future__ import division
import tensorflow as tf
import argparse
import os
import utils

def main(confFile):


if __name__ == '__main__':
    # Reading parser from command line
    # confFilePath: The path to the model config file
    parser = argparse.ArgumentParser();
    parser.add_argument('confFilePath', type=str, help='The path to the model config file')
    main(utils.ReadConfFile(parser.confFilePath))
