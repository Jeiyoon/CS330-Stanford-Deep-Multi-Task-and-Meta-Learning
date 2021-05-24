import os
import setproctitle

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

setproctitle.setproctitle("[k4ke] tf_test")

import tensorflow as tf

w = tf.Variable(tf.random.normal((3, 2)), name = 'w')
b = tf.Variable(tf.zeros(2, dtype = tf.float32), name = 'b')
x = [[1., 2., 3.]]

# Gradient tape tracks differentiable operations
# persistent = True keeps compute graph after tape.gradient
with tf.GradientTape(persistent = True) as tape:
    y = x @ w + b
    loss = tf.reduce_mean(y**2)

# computes tracked variable grads
[dl_dw, dl_db] = tape.gradient(loss, [w, b])

print(y)
print(dl_db)

