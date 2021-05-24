import os
import setproctitle

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

setproctitle.setproctitle("[k4ke] tf_test")

import tensorflow as tf

# ResourceVariable + constant = EagerTensor (diff = None)
x0 = tf.Variable(3.0, name = 'x0')
x1 = tf.Variable(3.0, name = 'x1', trainable = False)
x2 = tf.Variable(2.0, name = 'x2') + 1.0
x3 = tf.constant(3.0, name = 'x3')

print(type(x0))
print(type(x2))

with tf.GradientTape() as tape:
    y = (x0**2) + (x1**2) + (x2**2)

grad = tape.gradient(y, [x0, x1, x2, x3])

print(grad)

