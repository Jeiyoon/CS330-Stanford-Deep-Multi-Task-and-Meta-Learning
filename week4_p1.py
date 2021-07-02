# author: https://github.com/LecJackS and https://github.com/Luvata
# reimplementation and comments: jeiyoon
import setproctitle
import os

# pip install googledrivedownloader
from google_drive_downloader import GoogleDriveDownloader as gdd

import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import imageio

import tensorboard

import datetime

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# mirrored_strategy = tf.distribute.MirroredStrategy(devices = ["/gpu:0",
#                                                               "/gpu:1",
#                                                               "/gpu:2",
#                                                               "/gpu:3"])

setproctitle.setproctitle("[k4ke] meta_learning_test")

# multi gpu
# tf.debugging.set_log_device_placement(True)

# Need to download the Omniglot dataset
if not os.path.isdir('./omniglot_resized'):
    gdd.download_file_from_google_drive(file_id = '1iaSFXIYC3AB8q9K_M-oVMa4pmB7yKMtI',
                                        dest_path = './omniglot_resized.zip',
                                        unzip = True)

assert os.path.isdir('./omniglot_resized')

"""Utility functions. """
def cross_entropy_loss(pred, label, k_shot):
    # tf.reduce_mean
    # https://www.tensorflow.org/api_docs/python/tf/math/reduce_mean
    # tf.nn.softmax_cross_entropy_with_logits: Computes softmax cross entropy between logits and labels.
    # https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits
    # tf.stop_gradient: Stops gradient computation.
    # https://data-newbie.tistory.com/438
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred,
                                                                  labels = tf.stop_gradient(label)) / k_shot)


