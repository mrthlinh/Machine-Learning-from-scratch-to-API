'''
A Recurrent Neural Network (LSTM) implementation example using TensorFlow..
Next word prediction after n_input words learned from text file.
A story is automatically generated if the predicted word is fed back as input.
Author: Rowel Atienza
Project: https://github.com/roatienza/Deep-Learning-Experiments
'''
# Tips for training
# https://danijar.com/tips-for-training-recurrent-neural-networks/

from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import collections
import time

start_time = time.time()
def elapsed(sec):
    if sec<60:
        return str(sec) + " sec"
    elif sec<(60*60):
        return str(sec/60) + " min"
    else:
        return str(sec/(60*60)) + " hr"


# Target log path
logs_path = './tmp/rnn_words'
writer = tf.summary.FileWriter(logs_path)

# Text file containing words for training
# training_file = 'wonderland_small.txt'
training_file = 'train_set_small.txt'
# training_file = 'belling_the_cat.txt'
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
        dictionary[word] = len(dictionary) + 1
        # dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

dictionary, reverse_dictionary = build_dataset(training_data)
vocab_size = len(dictionary)

# Write dictionary and reverse_dictionary
import pickle
# Store data
with open('dictionary.pickle', 'wb') as handle:
    pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('reverse_dict.pickle', 'wb') as handle:
    pickle.dump(reverse_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Load data
# with open('dictionary.pickle', 'rb') as handle:
#     load_dict = pickle.load(handle)

# Parameters
learning_rate = 0.001
training_iters = 200 * len(training_data)
# training_iters = 50000
display_step = 1000
n_input = 3
max_seq = 20

# number of units in RNN cell
n_hidden = 512

# tf Graph input
x = tf.placeholder("float", [None, max_seq, 1])
y = tf.placeholder("float", [None, vocab_size])

# x = tf.placeholder("float", [None, n_input, 1])
# y = tf.placeholder("float", [None, vocab_size])

# RNN output node weights and biases
weights = tf.Variable(tf.random_normal([n_hidden, vocab_size]),name='w')

biases = tf.Variable(tf.random_normal([vocab_size]),name = 'b')

# weights = {
#     'out': tf.Variable(tf.random_normal([n_hidden, vocab_size]),name='w')
# }
# biases = {
#     'out': tf.Variable(tf.random_normal([vocab_size]),name = 'b')
# }

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
#     return tf.matmul(outputs[-1], weights) + biases

pred = RNN(x, weights, biases)

# Loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

# Model evaluation
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as session:
    session.run(init)
    step = 0
    # offset = random.randint(0,n_input+1)
    end_offset = n_input + 1
    acc_total = 0
    loss_total = 0

    writer.add_graph(session.graph)
    offset = 0
    # print("Before Training")
    # print("Weight: ",weights.eval())
    # print("Bias: ", biases.eval())

    while step < training_iters:
        # Generate a minibatch. Add some randomness on selection process.
        if offset > (len(training_data)-end_offset):
            # offset = random.randint(0, n_input+1)
            offset = 0
        # print("offset: ",offset)
        symbols_in_keys = [ [dictionary[ str(training_data[i])]] for i in range(offset, offset+n_input) ]
        zero_pad = [[0]] * (max_seq - n_input)
        symbols_in_keys = zero_pad + symbols_in_keys
        symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, max_seq, 1])
        # symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])

        symbols_out_onehot = np.zeros([vocab_size], dtype=float)
        symbols_out_onehot[dictionary[str(training_data[offset+n_input])]-1] = 1.0
        symbols_out_onehot = np.reshape(symbols_out_onehot,[1,-1])

        # symbols_in = [training_data[i] for i in range(offset, offset + n_input)]
        # symbols_out = training_data[offset + n_input]
        # # symbols_out_pred = reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval())]
        # print("%s - [%s]" % (symbols_in, symbols_out))

        _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], \
                                                feed_dict={x: symbols_in_keys, y: symbols_out_onehot})
        loss_total += loss
        acc_total += acc
        if (step+1) % display_step == 0:
            print("Iter= " + str(step+1) + ", Average Loss= " + \
                  "{:.6f}".format(loss_total/display_step) + ", Average Accuracy= " + \
                  "{:.2f}%".format(100*acc_total/display_step))
            acc_total = 0
            loss_total = 0
            symbols_in = [training_data[i] for i in range(offset, offset + n_input)]
            symbols_out = training_data[offset + n_input]
            symbols_out_pred = reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval())+1]
            print("%s - [%s] vs [%s]" % (symbols_in,symbols_out,symbols_out_pred))
        step += 1
        offset += 1
        # offset += (n_input+1)

        # offset += 1
    print("Optimization Finished!")
    print("Elapsed time: ", elapsed(time.time() - start_time))
    print("Run on command line.")
    print("\ttensorboard --logdir=%s" % (logs_path))
    print("Point your web browser to: http://localhost:6006/")

    # Save model
    saver = tf.train.Saver()
    save_path = saver.save(session, "./model/LSTM_model_n="+str(n_input)+".ckpt")
    #
    #
    # W = weights.eval()
    # b = biases.eval()

    # Examine
    # step = 0
    # # offset = random.randint(0,n_input+1)
    # end_offset = n_input + 1
    # while step < 30:
    #     # Generate a minibatch. Add some randomness on selection process.
    #     if offset > (len(training_data) - end_offset):
    #         offset = 0
    #         # offset = random.randint(0, n_input+1)
    #
    #     symbols_in_keys = [[dictionary[str(training_data[i])]] for i in range(offset, offset + n_input)]
    #     zero_pad = [[0]] * (max_seq - n_input)
    #     symbols_in_keys = zero_pad + symbols_in_keys
    #     symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, max_seq, 1])
    #     # for i in range(0,n_input):
    #     #     print(reverse_dictionary[symbols_in_keys[0,i,0]])
    #     symbols_out_onehot = np.zeros([vocab_size], dtype=float)
    #     symbols_out_onehot[dictionary[str(training_data[offset + n_input])]] = 1.0
    #     symbols_out_onehot = np.reshape(symbols_out_onehot, [1, -1])
    #     # print(reverse_dictionary[int(tf.argmax(symbols_out_onehot, 1).eval())])
    #     # print(reverse_dictionary[symbols_out_onehot[0, 1]])
    #     acc, loss, onehot_pred = session.run([accuracy, cost, pred],
    #                                          feed_dict={x: symbols_in_keys, y: symbols_out_onehot})
    #     loss_total += loss
    #     acc_total += acc
    #
    #     symbols_in = [training_data[i] for i in range(offset, offset + n_input)]
    #     symbols_out = training_data[offset + n_input]
    #     symbols_out_pred = reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval())]
    #     print("%s - [%s] vs [%s] - [%s]" % (symbols_in, symbols_out, symbols_out_pred, acc))
    #
    #     # if (step+1) % display_step == 0:
    #     #     print("Iter= " + str(step+1) + ", Average Loss= " + \
    #     #           "{:.6f}".format(loss_total/display_step) + ", Average Accuracy= " + \
    #     #           "{:.2f}%".format(100*acc_total/display_step))
    #     #     acc_total = 0
    #     #     loss_total = 0
    #     #     symbols_in = [training_data[i] for i in range(offset, offset + n_input)]
    #     #     symbols_out = training_data[offset + n_input]
    #     #     symbols_out_pred = reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval())]
    #     #     print("%s - [%s] vs [%s]" % (symbols_in,symbols_out,symbols_out_pred))
    #
    #     step += 1
    #     offset += (n_input + 1)

    # Query next word

    while True:
        prompt = "%s words: " % n_input
        sentence = input(prompt)
        sentence = sentence.strip()
        words = sentence.split(' ')
        # if len(words) != n_input:
        #     continue
        symbols_in_keys = [[dictionary[str(words[i])]] for i in range(len(words))]
        zero_pad = [[0]] * (max_seq - len(words))
        symbols_in_keys = zero_pad + symbols_in_keys
        keys = np.reshape(np.array(symbols_in_keys), [-1, max_seq, 1])
        # onehot_pred = session.run(pred, feed_dict={x: keys})
        # onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
        # sentence = "%s %s" % (sentence, reverse_dictionary[onehot_pred_index + 1])
        # symbols_in_keys = symbols_in_keys[1:]
        # symbols_in_keys.append(onehot_pred_index)
        # print(sentence)
        num_next_word = 1
        for i in range(num_next_word):
            # symbols_in_keys = [[dictionary[str(training_data[i])]] for i in range(offset, offset + n_input)]
            # zero_pad = [[0]] * (max_seq - n_input)
            # symbols_in_keys = zero_pad + symbols_in_keys
            # symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, max_seq, 1])

            # keys = np.reshape(np.array(symbols_in_keys), [-1, max_seq, 1])
            # keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])
            onehot_pred = session.run(pred, feed_dict={x: keys})
            onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
            sentence = "%s %s" % (sentence, reverse_dictionary[onehot_pred_index+1])
            symbols_in_keys = symbols_in_keys[1:]
            symbols_in_keys.append(onehot_pred_index)
        print(sentence)

        # try:
        #     symbols_in_keys = [dictionary[str(words[i])] for i in range(len(words))]
        #     zero_pad = [[0]] * (max_seq - n_input)
        #     symbols_in_keys = zero_pad + symbols_in_keys
        #     num_next_word = 1
        #     for i in range(num_next_word):
        #         keys = np.reshape(np.array(symbols_in_keys), [-1, max_seq, 1])
        #         # keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])
        #         onehot_pred = session.run(pred, feed_dict={x: keys})
        #         onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
        #         sentence = "%s %s" % (sentence,reverse_dictionary[onehot_pred_index]+1)
        #         symbols_in_keys = symbols_in_keys[1:]
        #         symbols_in_keys.append(onehot_pred_index)
        #     print(sentence)
        # except:
        #     print("Word not in dictionary")
    session.close()