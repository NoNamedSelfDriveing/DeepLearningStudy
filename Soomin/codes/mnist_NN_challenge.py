# MNIST Challenge - Soomin Lee (MagmaTart) #

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
tf.set_random_seed(9297)

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

class Model:
    def __init__(self, sess, ll):
        self.sess = sess
        self.learning_rate = ll
        self.build()

    def build(self):
        self.X = tf.placeholder(tf.float32)
        self.Y = tf.placeholder(tf.float32)
        self.dropout_rate = tf.placeholder(tf.float32)

        W1 = tf.get_variable("W1", shape=[28*28, 512], initializer=tf.contrib.layers.xavier_initializer())
        W2 = tf.get_variable("W2", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
        W3 = tf.get_variable("W3", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
        W4 = tf.get_variable("W4", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
        W5 = tf.get_variable("W5", shape=[512, 10], initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.Variable(tf.random_normal([512]))
        b2 = tf.Variable(tf.random_normal([512]))
        b3 = tf.Variable(tf.random_normal([512]))
        b4 = tf.Variable(tf.random_normal([512]))
        b5 = tf.Variable(tf.random_normal([10]))

        layer1 = tf.nn.relu(tf.matmul(self.X, W1) + b1)
        layer1 = tf.nn.dropout(layer1, keep_prob=self.dropout_rate)
        layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)
        layer2 = tf.nn.dropout(layer2, keep_prob=self.dropout_rate)
        layer3 = tf.nn.relu(tf.matmul(layer2, W3) + b3)
        layer3 = tf.nn.dropout(layer3, keep_prob=self.dropout_rate)
        layer4 = tf.nn.relu(tf.matmul(layer3, W4) + b4)
        layer4 = tf.nn.dropout(layer4, keep_prob=self.dropout_rate)

        logits = tf.matmul(layer4, W5) + b5

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.Y))
        self.trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        correct = tf.equal(tf.argmax(logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    def get_accuracy(self, testX, testY):
        return self.sess.run(self.accuracy, feed_dict={self.X:testX, self.Y:testY, self.dropout_rate:1.0})

    def train(self, trainX, trainY, dropout_rate):
        cost_v, _ = self.sess.run([self.cost, self.trainer], feed_dict={self.X:trainX, self.Y:trainY, self.dropout_rate:dropout_rate})
        return cost_v

sess = tf.Session()

epochs = 20
batch_size = 100
learning_rate = 0.001

M1 = Model(sess, learning_rate)

sess.run(tf.global_variables_initializer())

for epoch in range(epochs):
    batches = int(mnist.train.num_examples / batch_size)
    avg_cost = 0

    for step in range(batches):
        batchX, batchY = mnist.train.next_batch(batch_size)
        cost_v = M1.train(batchX, batchY, 0.7)
        avg_cost += cost_v / batches

    print(epoch, avg_cost)

print("Acc :", M1.get_accuracy(mnist.test.images, mnist.test.labels))
