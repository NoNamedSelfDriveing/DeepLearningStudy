# SOFTMAX 분류

SOFTMAX 분류는, 세 개 이상의 클래스를 분류하기 위해 사용되는 분류법이다.

기본적인 아이디어는, 세 개 이상의 클래스로 분류되는 데이터들을 여러 개의 Binary Classfication을 이용해 분류하자는 이야기이다.

![](https://latex.codecogs.com/gif.latex?%5Cbegin%7Bbmatrix%7D%20w_1%20%5C%20w_2%20%5C%20w_3%20%5Cend%7Bbmatrix%7D%20%5Ccdot%20%5Cbegin%7Bbmatrix%7D%20x_1%20%5C%5C%20x_2%20%5C%5C%20x_3%20%5Cend%7Bbmatrix%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20w_1x_1%20&plus;%20w_2x_2%20&plus;%20w_3x_3%20%5Cend%7Bbmatrix%7D)

위의 식은, Binary Classfication에서의 행렬을 이용한 가설 함수 계산이다. 이제 여러 개의 클래스를 분류하려면, 행렬의 출력이 클래스의 개수만큼 나와줘야 된다. 그러기 위해서, 출력 클래스의 개수만큼 Weight의 크기를 맞춰줘야 한다. 예를 들어 위의 상황에서 출력 클래스의 갯수가 4개라면, 다음과 같은 모양으로 가설 함수의 행렬 연산이 정의되어야 한다.

![](https://latex.codecogs.com/gif.latex?%5Cbegin%7Bbmatrix%7D%20w_a_1%20%5C%20w_a_2%20%5C%20w_a_3%20%5C%5C%20w_b_1%20%5C%20w_b_2%20%5C%20w_b_3%20%5C%5C%20w_c_1%20%5C%20w_c_2%20%5C%20w_c_3%20%5C%5C%20w_d_1%20%5C%20w_d_2%20%5C%20w_d_3%20%5C%5C%20%5Cend%7Bbmatrix%7D%20%5Ccdot%20%5Cbegin%7Bbmatrix%7D%20x_1%20%5C%5C%20x_2%20%5C%5C%20x_3%20%5Cend%7Bbmatrix%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20w_a_1x_1%20&plus;%20w_a_2x_2%20&plus;%20w_a_3x_3%20%5C%5C%20w_b_1x_1%20&plus;%20w_b_2x_2%20&plus;%20w_b_3x_3%20%5C%5C%20w_c_1x_1%20&plus;%20w_c_2x_2%20&plus;%20w_c_3x_3%20%5C%5C%20w_d_1x_1%20&plus;%20w_d_2x_2%20&plus;%20w_d_3x_3%20%5C%5C%20%5Cend%7Bbmatrix%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20y_1%20%5C%5C%20y_2%20%5C%5C%20y_3%20%5C%5C%20y_4%20%5Cend%7Bbmatrix%7D)

각 클래스마다 따로 정의된 Weight를 이용해서, 출력 4개를 도출하고 있다. 이제 출력 ![](https://latex.codecogs.com/gif.latex?%5Cbegin%7Bbmatrix%7D%20y_1%20%5C%5C%20y_2%20%5C%5C%20y_3%20%5C%5C%20y_4%20%5Cend%7Bbmatrix%7D) 에 SOFTMAX 활성화 함수를 적용해, 총합이 1인 확률로 변환 후, One-hot Encoding을 통해 최고 확률만 1로 설정해주는 작업을 거쳐 분류를 완성할 수 있다.

이제 이를 Tensorflow로 구현해 보자.

```
import tensorflow as tf
tf.set_random_seed(9297)

dataX = [[1, 1, 2, 2], [1, 2, 3, 3], [3, 1, 3, 3], [2, 2, 4, 4], [1, 7, 5, 3], [9, 9, 8, 8], [6, 6, 6, 6], [7, 7, 7, 7]]
dataY = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]

X = tf.placeholder(tf.float32, [None, 4])
Y = tf.placeholder(tf.float32, [None, 3])

W = tf.Variable(tf.random_normal([4, 3]))
b = tf.Variable(tf.random_normal([3]))

logit = tf.nn.softmax(tf.matmul(X, W) + b)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(logit), axis=1))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    for step in range(10001):
        session.run(trainer, feed_dict={X:dataX, Y:dataY})
        if step % 200 == 0:
            print(step, session.run(cost, feed_dict={X:dataX, Y:dataY}))
```

