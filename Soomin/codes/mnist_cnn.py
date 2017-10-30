import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(9297)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate = 0.001
training_epochs = 15
batch_size = 100
keep_prob = 0.7

X = tf.placeholder(tf.float32)
X_img = tf.reshape(X, [-1, 28, 28, 1])    # Make 4-dimensional image
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))          # 32 filters of [1, 3, 3]
Layer1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')  # Convolution with [1, 1] strides
Layer1 = tf.nn.relu(Layer1)
Layer1 = tf.nn.max_pool(Layer1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')    # Max pooling with [2, 2] size and [2, 2] strides 

W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))         # 64 filters of [32, 3, 3]
Layer2 = tf.nn.conv2d(Layer1, W2, strides=[1, 1, 1, 1], padding='SAME')
Layer2 = tf.nn.relu(Layer2)
Layer2 = tf.nn.max_pool(Layer2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))        # 128 filters of [64, 3, 3]
Layer3 = tf.nn.conv2d(Layer2, W3, strides=[1, 1, 1, 1], padding='SAME')
Layer3 = tf.nn.relu(Layer3)
Layer3 = tf.nn.max_pool(Layer3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
Layer3_flat = tf.reshape(Layer3, [-1, 4 * 4 * 128])    # flat the output feature map to [1,  4 * 4 * 128] size

W4 = tf.get_variable("W4", shape=[4*4*128, 625], initializer=tf.contrib.layers.xavier_initializer())   # Fully Connected Layer
b4 = tf.Variable(tf.random_normal([625]))
Layer4 = tf.matmul(Layer3_flat, W4) + b4
Layer4 = tf.nn.relu(Layer4)
Layer4 = tf.nn.dropout(Layer4, keep_prob=keep_prob)

W5 = tf.get_variable("W5", shape=[625, 10], initializer=tf.contrib.layers.xavier_initializer())    # Fully Connected Layer
b5 = tf.Variable(tf.random_normal([10]))

logits = tf.matmul(Layer4, W5) + b5    # Logits function

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))    # Softmax cost
trainer = tf.train.AdamOptimizer(0.001).minimize(cost)    # Use Adam optimizer

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batchX, batchY = mnist.train.next_batch(batch_size)
            feed_dict = {X:batchX, Y:batchY}
            c, _ = sess.run([cost, trainer], feed_dict=feed_dict)
            avg_cost += c / total_batch

        print(epoch + 1, avg_cost)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Acc :', sess.run(accuracy, feed_dict={X:mnist.test.images, Y:mnist.test.labels}))

