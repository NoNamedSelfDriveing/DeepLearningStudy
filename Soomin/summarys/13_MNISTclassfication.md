# MNIST 분류의 구현

SOFTMAX 분류를 이용하여, 실제 MNIST 데이터를 분류하는 작업을 구현해보자.

이전에 보았듯이, MNIST 데이터는 아래와 같은 28 * 28 크기 손글씨 데이터셋이다.

![](../image/mnist.PNG)

이전에 구현해 본 일반 SOFTMAX와 동일한 방법을 사용해서 MNIST 손글씨 데이터의 분류를 구현해보자. 데이터셋은, `tensorflow.examples.tutorials.mnist` 에 있는 `input_data` 를 이용하여 가져오도록 하겠다.

저 라이브러리에서 mnist 데이터를 가지고 오면, 그 과정에서 자동으로 데이터를 flatten하게 펴 준다. 따라서 구현은 28 * 28 = 784의 길이를 가지고 있는 1차원 데이터들의 행렬로써 구현하면 된다.

숫자의 분류 클래스는 0부터 9까지 총 10개이다. 이를 이용해서 X, W, Weight, Bias의 모양을 결정해 주었다.

또 이 구현에서는, 일정 크기로 데이터를 끊어서 학습하는 __배치 학습__ 을 구현하였다. 이 또한 tensorflow 라이브러리의 도움을 받았지만, 스스로 구현하기도 어렵지 않을 것이다.

실제 코드로 보자.

```
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import random
tf.set_random_seed(9297)

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)    # MNIST 데이터셋 다운로드와 로딩

X = tf.placeholder(tf.float32, [None, 28*28])    # X 입력 데이터 : 28*28 = 784 크기
Y = tf.placeholder(tf.float32, [None, 10])       # Y 정답 데이터 : 분류 클래스 = 10 크기

W = tf.Variable(tf.random_normal([28*28, 10]))   # 가중치 : 입력 노드 개수 * 출력 노드 개수
b = tf.Variable(tf.random_normal([10]))          # 편향 : 출력 노드 개수

logits = tf.nn.softmax(tf.matmul(X, W) + b)      # 가설 함수 : 선형 함수 + SOFTMAX 활성화
cost = -tf.reduce_mean(tf.reduce_sum(Y * tf.log(logits), axis=1))     # 손실 함수 : 교차 엔트로피 오차

trainer = tf.train.GradientDescentOptimizer(0.1).minimize(cost)       # 경사 하강법 사용, 학습률 : 0.1

epochs = 15        # 배치 학습 : 15번으로 나눠서 학습
batch_size = 100   # 한번의 배치는 100 크기

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):

	# 총 데이터의 개수를 배치의 사이즈만큼 나눠서, 결국 모든 데이터를 학습할 수 있도록 함
        for step in range(int(mnist.train.num_examples / batch_size)):
            batchX, batchY = mnist.train.next_batch(batch_size)    # 배치 데이터 불러오기
            cost_v,_ = sess.run([cost, trainer], feed_dict={X:batchX, Y:batchY})    # 배치 데이터 학습

        print("Epoch :", epoch, "Loss :", cost_v)    # 한 Epoch을 거친 후 손실 파악

    # 임의로 하나의 테스트 이미지를 설정한 후, 잘 예측하는지 확인
    r = random.randint(0, mnist.test.num_examples-1)
    print("Label :", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
    print("Prediction :", sess.run(tf.argmax(logits, 1), feed_dict={X:mnist.test.images[r:r+1]}))
```
