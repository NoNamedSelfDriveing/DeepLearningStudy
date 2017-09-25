# Tensorflow로 Linear Regression 구현

실제 선형 회귀 알고리즘을, Tensorflow를 이용하여 구현해보자.



```
import tensorflow as tf
import numpy as np

tf.set_random_seed(9297)

dataX = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
dataY = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_normal([1]))		# Weight : 1 크기
b = tf.Variable(tf.random_normal([1]))		# Bias : 1 크기

logits = W*X+b		# 가설 함수
cost = tf.reduce_mean(tf.square(logits-Y))		# 손실 함수

trainer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)		# 경사하강법 사용

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(1001):
        cost_v, n = sess.run([cost, trainer], feed_dict={X:dataX, Y:dataY})

        if step % 100 == 0:
            print(step, cost_v)

    print("200 :", sess.run(logits, feed_dict={X:[200]}))
```
