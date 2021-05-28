"""
TO DO:
colab -> ssh
"""

# https://github.com/tensorflow/probability/tree/master/tensorflow_probability/examples

# tfd = tfp.distributions
# Define a batch of two scalar valued Normals.
# The first has mean 1 and standard deviation 11, the second 2 and 22.
# dist = tfd.Normal(loc=[1, 2.], scale=[11, 22.])

import tensorflow as tf
import tensorflow_probability as tfp

mean  = tf.Variable([1.0, 2.0, 3.], name = 'mean')
std = tf.Variable([0.1, 0.1, 0.1], name = 'std')
var = tf.constant([3.0, 0.1, 2.0], name = 'var')

# Gradient tape tracks differentiable operations
# "persistent = True" keeps compute graph after tape.gradient
with tf.GradientTape(persistent = True) as tape:
  dist = tfp.distributions.Normal(loc = mean, scale = std)
  s = dist.sample() # s.shape = (3, )
  
  # tf.reduce_mean: (1/N)Σ[N,i]
  loss1 = tf.reduce_mean(s ** 2)
  loss2 = tf.reduce_mean(dist.log_prob(var))
  loss3 = tf.reduce_mean(dist.log_prob(s))
  

grad1 = tape.gradient(loss1, [mean])
grad2 = tape.gradient(loss2, [mean])
grad3 = tape.gradient(loss3, [mean])


