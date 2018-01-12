import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
#Simple RNN
n_input     = 3
n_neuron    = 5
n_step      = 2

X = tf.placeholder(tf.float32, shape = [None, n_step, n_input])
# X_seqs = tf.unstack(tf.transpose(X, perm=[1, 0, 2]))
basic_cell = tf.contrib.rnn.BasicRNNCell(num_units = n_neuron)
# output_seqs, state = tf.contrib.rnn.static_rnn(basic_cell, X_seqs,dtype = tf.float32)
output_seqs, state = tf.nn.dynamic_rnn(basic_cell, X,dtype = tf.float32)
outputs = tf.transpose(tf.stack(output_seqs), perm=[1, 0, 2])

X_batch = np.array([
    [[0, 1, 2], [9, 8, 7]],
    [[3, 4, 5], [0, 0, 0]],
    [[6, 7, 8], [6, 5, 4]],
    [[9, 0, 1], [3, 2, 1]],
])
init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    output_val = outputs.eval(feed_dict = {X: X_batch})

print(output_val)
# output_seqs, states = tf.contrib.

# https://github.com/roatienza/Deep-Learning-Experiments/blob/master/Experiments/Tensorflow/RNN/rnn_words.py
# https://github.com/roatienza/Deep-Learning-Experiments/blob/master/Experiments/Tensorflow/RNN/belling_the_cat.txt