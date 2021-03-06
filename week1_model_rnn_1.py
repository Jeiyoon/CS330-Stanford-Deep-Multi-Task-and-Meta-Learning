import tensorflow as tf

import os
import setproctitle

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

setproctitle.setproctitle("[k4ke] tf_test")

print(tf.test.is_gpu_available())

# batch * length * size of data
inputs = tf.random.normal([32, 10, 8])

cell = tf.keras.layers.LSTMCell(4)
# initialize cell state
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/RNN
# method that creates a tensor meant to be fed to call() as the initial state, if the user didn't specify any initial state via other means
state = cell.get_initial_state(batch_size = 32, dtype = tf.float32)
# process data one at a time
# inputs[:,0]: (32, 8) <- first one of em
# https://rfriend.tistory.com/290
output, state = cell(inputs[:, 0], state)

print(output.shape)
print(state[0].shape)
print(state[1].shape)
