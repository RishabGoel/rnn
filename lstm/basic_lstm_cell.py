import tensorflow as tf
import numpy as np



def convert_2_bow(mapping, word, bow_size):
    bow = np.zeros([1,bow_size])
#     print bow
    bow[0][mapping[word]] = 1
#     print bow
    return bow

# Generate test data
from collections import Counter
data = open("data.txt","r").read()
data1 = data.strip().split()
data = Counter(data1)
pred_data1 = np.roll(data1,-1)
map_id = {word : idx for idx, word in enumerate(data.keys())}

# Dimension Variables
bow_size = len(data)
hidden_units = 200

x_inp = [convert_2_bow(map_id, word, bow_size) for word in data1]
y_out = [convert_2_bow(map_id, word, bow_size) for word in pred_data1]



# Weghts initialization
W_i_x = tf.Variable(tf.random_normal([bow_size, hidden_units]),name = "w_i_x")
W_i_h = tf.Variable(tf.random_normal([hidden_units, hidden_units]),name = "w_i_h")
W_f_x = tf.Variable(tf.random_normal([bow_size, hidden_units]),name = "w_f_x")
W_f_h = tf.Variable(tf.random_normal([hidden_units, hidden_units]),name = "w_f_h")
W_o_x = tf.Variable(tf.random_normal([bow_size, hidden_units]),name = "w_o_x")
W_o_h = tf.Variable(tf.random_normal([hidden_units, hidden_units]),name = "w_o_h")
W_c_x = tf.Variable(tf.random_normal([bow_size, hidden_units]),name = "w_c_x")
W_c_h = tf.Variable(tf.random_normal([hidden_units, hidden_units]),name = "w_c_h")

# Biases
b_i = tf.Variable(tf.random_normal([hidden_units]),name = "b_i")
b_f = tf.Variable(tf.random_normal([hidden_units]),name = "b_f")
b_o = tf.Variable(tf.random_normal([hidden_units]),name = "b_o")
b_c = tf.Variable(tf.random_normal([hidden_units]),name = "b_c")


def rnn_step(X, c, h):
    i = tf.sigmoid(tf.matmul(X, W_i_x) + tf.matmul(h, W_i_h) + b_i)
    o = tf.sigmoid(tf.matmul(X, W_o_x) + tf.matmul(h, W_o_h) + b_o)
    f = tf.sigmoid(tf.matmul(X, W_f_x) + tf.matmul(h, W_f_h) + b_f)
    c_hat = tf.tanh(tf.matmul(X, W_c_x) + tf.matmul(h, W_c_h) + b_c)
    c = f*c + i*c_hat
    h = o * tf.tanh(c)
    h_drop = tf.nn.dropout(h,0.5)
    return c, h
    # return c, h_drop


X = tf.placeholder("float32", [1, bow_size])
Y = tf.placeholder("float32", [1, bow_size])
h = tf.Variable(tf.zeros([1, hidden_units]))
c = tf.Variable(tf.zeros([1, hidden_units]))
c, h = rnn_step(X, c,h)
W_o = tf.Variable(tf.random_normal([hidden_units, bow_size]))
b__o = tf.Variable(tf.random_normal([1, bow_size]))
output = tf.nn.softmax(tf.matmul(h, W_o) + b__o)
cross_entropy = -tf.reduce_sum(Y * tf.log(output))

optimizer = tf.train.AdamOptimizer()

gvs = optimizer.compute_gradients(cross_entropy)
capped_gvs = [(tf.clip_by_value(grad, -3., 3.), var) for grad, var in gvs]
optimizer.apply_gradients(capped_gvs)

minimize = optimizer.minimize(cross_entropy)
mistakes = tf.not_equal(tf.arg_max(output,1), tf.arg_max(Y,1))
error = tf.cast(mistakes, "float32")


init_op = tf.initialize_all_variables()
correct = 0
with tf.Session() as sess:
    sess.run(init_op)
    for i in range(100):
        for i in range(len(x_inp)):
    #         print "train ",i
            sess.run(minimize, feed_dict = {X : x_inp[i], Y : y_out[i]})
    for i in range(len(x_inp)):
        if sess.run(error, feed_dict = {X : x_inp[i], Y : y_out[i]})[0] != 1:
            correct += 1
    
print correct/float(len(x_inp))
