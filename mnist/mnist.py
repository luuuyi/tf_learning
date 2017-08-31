import tensorflow as tf
import numpy as np 
from numpy.random import RandomState
from tensorflow.examples.tutorials.mnist import input_data

def inference(input_t, aver, w1, b1, w2, b2):
    if aver == None:
        a = tf.matmul(input_x, w1) + b1
        relu_a = tf.nn.relu(a)
        b = tf.matmul(relu_a, w2) + b2
        return b
    else:
        a = tf.matmul(input_x, aver.average(w1)) + aver.average(b1)
        relu_a = tf.nn.relu(a)
        b = tf.matmul(relu_a, aver.average(w2)) + aver.average(b2)
        return b

def train_v1():
    mnist_ds = input_data.read_data_sets('./data', one_hot=True)
    print(mnist_ds.train.num_examples)
    print(mnist_ds.validation.num_examples)
    print(mnist_ds.test.num_examples)

    input_node = 784
    output_node = 10
    layer1_node = 500
    batch_size = 128

    LEARNING_RATE_BASE = 0.8      
    LEARNING_RATE_DECAY = 0.99    
    REGULARAZTION_RATE = 0.0001   
    TRAINING_STEPS = 10000        
    MOVING_AVERAGE_DECAY = 0.99

    graph = tf.Graph()
    with graph.as_default():
        w1 = tf.Variable(tf.truncated_normal([input_node, layer1_node], stddev=0.1))
        b1 = tf.Variable(tf.constant(0.1, shape=[1, layer1_node]))
        w2 = tf.Variable(tf.truncated_normal([layer1_node, output_node], stddev=0.1))
        b2 = tf.Variable(tf.constant(0.1, shape=[1, output_node]))
        input_x = tf.placeholder(tf.float32, [None, input_node])
        input_y = tf.placeholder(tf.float32, [None, output_node])

        global_step = tf.Variable(0, False)
        aver = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        aver_ops = aver.apply(tf.trainable_variables())

        y_inf = inference(input_x, None, w1, b1, w2, b2)
        y_aver = inference(input_x, aver, w1, b1, w2, b2)
        cross_entr_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_inf, labels=tf.arg_max(input_y, 1)))
        regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
        loss = cross_entr_loss + regularizer(w1) + regularizer(w2)

        learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist_ds.train.num_examples / batch_size,
        LEARNING_RATE_DECAY,
        True)
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step)
        with tf.control_dependencies([train_step, aver_ops]):
            train = tf.no_op('train')

        accuray = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(input_y, 1), tf.arg_max(y_aver, 1)), tf.float32))

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        valid_feed = {input_x:mnist_ds.validation.images, input_y:mnist_ds.validation.labels}
        test_feed = {input_x:mnist_ds.test.images, input_y:mnist_ds.test.labels}
        for i in range(TRAINING_STEPS):
            xs, ys = mnist_ds.train.next_batch(batch_size)
            train_feed = {input_x:xs, input_y:ys}
            sess.run(train, feed_dict=train_feed)
            if i % 100 == 0:
                print('The %d iters, loss is %f'%(i, sess.run(loss, feed_dict=train_feed)))

            if i % 1000 == 0:
                accuray_rate = sess.run(accuray, feed_dict=valid_feed)
                print('The %d validation, accuray is %f'%(1/1000, accuray_rate))
        
        test_accuray = sess.run(accuray, feed_dict=test_feed)
        print('After %d iters train, test accuray is %f'%(TRAINING_STEPS, test_accuray))

if __name__ == '__main__':
    train_v1()

'''import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
 


#加载MNIST数据集
mnist = input_data.read_data_sets("./data", one_hot=True)


INPUT_NODE = 784     
OUTPUT_NODE = 10     
LAYER1_NODE = 500         

BATCH_SIZE = 100 
 
       

# 模型相关的参数
LEARNING_RATE_BASE = 0.8      
LEARNING_RATE_DECAY = 0.99    
REGULARAZTION_RATE = 0.0001   
TRAINING_STEPS = 10000        
MOVING_AVERAGE_DECAY = 0.99 
 


def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    # 使用滑动平均类
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2

    else:

        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)  


def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

    # 生成隐藏层的参数。
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

    # 生成输出层的参数。
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))


    # 计算不含滑动平均类的前向传播结果
    y = inference(x, None, weights1, biases1, weights2, biases2)


    # 定义训练轮数及相关的滑动平均类 
    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)

    # 计算交叉熵及其平均值
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 定义交叉熵损失函数加上正则项为模型损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    regularaztion = regularizer(weights1) + regularizer(weights2)
    loss = cross_entropy_mean + regularaztion

    # 设置指数衰减的学习率。
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)

    # 随机梯度下降优化器优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 反向传播更新参数和更新每一个参数的滑动平均值
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    # 计算准确度
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 初始化会话并开始训练过程。
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        test_feed = {x: mnist.test.images, y_: mnist.test.labels} 

        # 循环地训练神经网络。
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training step(s), validation accuracy using average model is %g " % (i, validate_acc))

            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op,feed_dict={x:xs,y_:ys})

        test_acc=sess.run(accuracy,feed_dict=test_feed)
        print(("After %d training step(s), test accuracy using average model is %g" %(TRAINING_STEPS, test_acc)))

if __name__ == '__main__':
    train(mnist)'''