import tensorflow as tf

import os
import setproctitle

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

setproctitle.setproctitle("[k4ke] tf_test")

print(tf.test.is_gpu_available())

# batch * length * size of data
inputs = tf.random.normal([32, 10, 8])

lstm = tf.keras.layers.LSTM(4)
output = lstm(inputs)

print(output.shape)

lstm = tf.keras.layers.LSTM(4,
                            return_sequences = True,
                            return_state = True)

whole_seq_output, final_memory_state, final_carry_state = lstm(inputs)

print(whole_seq_output.shape)
print(final_memory_state.shape)
print(final_carry_state.shape)
