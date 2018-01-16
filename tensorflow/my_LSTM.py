from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import collections
import time
# tf.reset_default_graph()

start_time = time.time()

def elapsed(sec):
    if sec<60:
        return str(sec) + " sec"
    elif sec<(60*60):
        return str(sec/60) + " min"
    else:
        return str(sec/(60*60)) + " hr"


# Target log path
# logs_path = './tmp/tensorflow/rnn_words'
# writer = tf.summary.FileWriter(logs_path)

# Text file containing words for training
# training_file = 'wonderland_small.txt'
# training_file = 'train_set_small.txt'
training_file = 'belling_the_cat.txt'

def read_data(fname):
    output = []
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    for i in range(len(content)):
        split = content[i].split()
        output = output + split
    # content = [content[i].split() for i in range(len(content))]
    output = np.array(output)
    output = np.reshape(output, [-1, ])
    return output

training_data = read_data(training_file)
print("Loaded training data...")

def build_dataset(words):
    count = collections.Counter(words).most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

# dictionary, reverse_dictionary = build_dataset(training_data)

import pickle
with open('dictionary.pickle', 'rb') as handle:
    dictionary = pickle.load(handle)
with open('reverse_dict.pickle', 'rb') as handle:
    reverse_dictionary = pickle.load(handle)

vocab_size = len(dictionary)

# Parameters
# learning_rate = 0.001
training_iters = 50 * len(training_data)
# # training_iters = 50000
display_step = 1000
n_input = 3
max_seq = 20
# number of units in RNN cell
n_hidden = 512

# tf Graph input
x = tf.placeholder("float", [None, max_seq, 1])
y = tf.placeholder("float", [None, vocab_size])

# RNN output node weights and biases
weights = tf.Variable(tf.random_normal([n_hidden, vocab_size]),name='w')

biases = tf.Variable(tf.random_normal([vocab_size]),name = 'b')
def RNN(x, weights, biases):

    # reshape to [1, n_input]
    x = tf.reshape(x, [-1, max_seq])

    # Generate a n_input-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    x = tf.split(x,max_seq,1)

    # 2-layer LSTM, each layer has n_hidden units.
    # Average Accuracy= 95.20% at 50k iter
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden),rnn.BasicLSTMCell(n_hidden)])

    # 1-layer LSTM with n_hidden units but with lower accuracy.
    # Average Accuracy= 90.60% 50k iter
    # Uncomment line below to test but comment out the 2-layer rnn.MultiRNNCell above
    # rnn_cell = rnn.BasicLSTMCell(n_hidden)

    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    # there are n_input outputs but
    # we only want the last output
    return tf.matmul(outputs[-1], weights) + biases
# def RNN(x, weights, biases):
#
#     # reshape to [1, n_input]
#     x = tf.reshape(x, [-1, n_input])
#
#     # Generate a n_input-element sequence of inputs
#     # (eg. [had] [a] [general] -> [20] [6] [33])
#     x = tf.split(x,n_input,1)
#
#     # 2-layer LSTM, each layer has n_hidden units.
#     # Average Accuracy= 95.20% at 50k iter
#     rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden),rnn.BasicLSTMCell(n_hidden)])
#
#     # 1-layer LSTM with n_hidden units but with lower accuracy.
#     # Average Accuracy= 90.60% 50k iter
#     # Uncomment line below to test but comment out the 2-layer rnn.MultiRNNCell above
#     # rnn_cell = rnn.BasicLSTMCell(n_hidden)
#
#     # generate prediction
#     outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
#
#     # there are n_input outputs but
#     # we only want the last output
#     result = tf.matmul(outputs[-1], weights) + biases
#     return result

pred = RNN(x, weights, biases)

# Loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
# optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)
#
# Model evaluation
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
# init = tf.global_variables_initializer()

# tf.reset_default_graph()
# model_checkpoint = './model/LSTM_model.ckpt'
saver = tf.train.Saver()
with tf.Session() as session:
    # session.run(init)
    # saver = tf.train.Saver()
    # saver = tf.train.import_meta_graph('./model/LSTM_model_n=3.ckpt.meta')
    # saver.restore(session, './model/LSTM_model_n=3.ckpt')
    saver.restore(session, tf.train.latest_checkpoint('./model/'))


    while True:
        prompt = "%s words: " % n_input
        sentence = input(prompt)
        sentence = sentence.strip()
        words = sentence.split(' ')
        try:
            symbols_in_keys = [[dictionary[str(words[i])]] for i in range(len(words))]
            zero_pad = [[0]] * (max_seq - len(words))
            symbols_in_keys = zero_pad + symbols_in_keys
            keys = np.reshape(np.array(symbols_in_keys), [-1, max_seq, 1])
            num_next_word = 1
            for i in range(num_next_word):
                onehot_pred = session.run(pred, feed_dict={x: keys})
                onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
                sentence = "%s %s" % (sentence, reverse_dictionary[onehot_pred_index+1])
                symbols_in_keys = symbols_in_keys[1:]
                symbols_in_keys.append(onehot_pred_index)
            print(sentence)
        except:
            print("Word not in dictionary")

    session.close()




