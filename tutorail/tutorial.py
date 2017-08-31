import tensorflow as tf
import numpy as np 
from numpy.random import RandomState

x_input = tf.constant(2, tf.int16)
w_input = tf.constant(4, tf.float16)

graph = tf.Graph()
with graph.as_default():
    a = tf.Variable(8, tf.int16)
    b = tf.Variable(tf.zeros([2,2]))

with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()
    print(graph)
    print(sess.run(a))
    print(sess.run(b))

a = tf.constant(2, tf.int16)
b = tf.constant(4, tf.int16)
x = tf.Variable(6, tf.int16)
y = tf.Variable(8, tf.int16)
sess1 = tf.Session()
result = a+b
print(result)
print(sess1.run(result))
sess1.run(tf.global_variables_initializer())
print(x, y)
print(sess1.run(x), sess1.run(y))
sess1.close()

#---------------------------------------------------------------------------------------------
x = tf.Variable(tf.truncated_normal([256*256, 10]))
w = tf.Variable(tf.ones([10, 10]))
print(x.get_shape().as_list())
print(w.get_shape().as_list())

x = tf.placeholder(tf.float32, [1, 2])
x1 = tf.constant([[0.7, 0.9]])
w = tf.Variable(tf.random_normal([1, 2], mean=1.0, stddev=2.0))
y = x + w
y1 = x1 + w

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(y, feed_dict={x:[[0.6, 0.8]]}))
print(sess.run(y1))

#-----------------------------------------------------------------------------------------------
points_1 = [[1,2], [2,3], [3,4], [4,5]]
points_2 = [[3,4], [4,5], [5,6], [6,7]]
input_x = np.array([np.array(elem).reshape(1,2) for elem in points_1])
input_y = np.array([np.array(elem).reshape(1,2) for elem in points_2])

graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(tf.float16, [1,2])
    y = tf.placeholder(tf.float16, [1,2])
    def calc(xi, yi):
        diff = tf.subtract(xi, yi)
        pows = tf.pow(diff, tf.constant(2.0, tf.float16, [1,2]))
        adds = tf.reduce_sum(pows)
        result = tf.sqrt(adds)
        return result
    ret = calc(x, y)

with tf.Session(graph=graph) as sess:
    for i in range(len(points_1)):
        x_i = input_x[i]
        y_i = input_y[i]
        feed_dict = {x:x_i, y:y_i}
        print('The %d result is %f' % (i, sess.run(ret, feed_dict=feed_dict)))

#---------------------------------------------------------------------------------------------
batch_size = 10
iters = 20000
data_size = 512
rst = RandomState(1)
X = rst.rand(data_size, 2)
Y = [[int(x1+x2<1)] for (x1, x2) in X]

graph = tf.Graph()
with graph.as_default():
    w1 = tf.Variable(tf.random_normal([2, 3]))
    w2 = tf.Variable(tf.random_normal([3, 1]))
    input_x = tf.placeholder(tf.float32, [None, 2])
    input_y = tf.placeholder(tf.float32, [None, 1])

    a = tf.matmul(input_x, w1)
    relu_a = tf.nn.sigmoid(a)
    b = tf.matmul(relu_a, w2)
    relu_b = tf.nn.sigmoid(b)
    cross_entr_loss = -tf.reduce_mean(input_y*tf.log(tf.clip_by_value(relu_b, 1e-10, 1.0)))
    train = tf.train.AdamOptimizer().minimize(cross_entr_loss)

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(w1))
    print(sess.run(w2))
    for i in range(iters):
        start = i*batch_size%data_size
        end = min(start+batch_size, data_size)
        feed_dict = {input_x:X[start:end], input_y:Y[start:end]}
        sess.run(train, feed_dict=feed_dict)
        if i % 500 == 0:
            print('The %d iter, loss is %f'%(i, sess.run(cross_entr_loss, feed_dict=feed_dict)))