# Deep and Wide XOR

이전 글에서, XOR 문제를 해결하기 위한 신경망을 만들고 학습시켰었다. 이번에는 그 신경망을 더 넓게, 또는 더 깊게 구성해서 어떻게 성능의 변화가 있는지 직접 확인해보자.

이 글에서는, 신경망의 구성 부분을 제외한 나머지 코드들은 모두 ![14번 글](https://github.com/MagmaTart/DeepLearningStudy/blob/master/Soomin/summarys/14_XORproblem.md)과 동일하게 사용한다. 또한 신경망의 기본적인 구조도 동일하게 사용한다.

### Wide Network

먼저 신경망을 더 넓게 구성해보자. 첫 번째 히든 레이어의 퍼셉트론을 원래대로 4개로도 해 보고, 더 넓게 20, 40개로 구성한 후 학습시켜보자.

먼저 20개로 구성해보자.
```
W1 = tf.Variable(tf.random_normal([2, 20]))
b1 = tf.Variable(tf.random_normal([20]))

W2 = tf.Variable(tf.random_normal([20, 2]))
b2 = tf.Variable(tf.random_normal([2]))

W3 = tf.Variable(tf.random_normal([2, 1]))
b3 = tf.Variable(tf.random_normal([1]))
```

그 다음, 40개로 구성해보자.
```
W1 = tf.Variable(tf.random_normal([2, 40]))
b1 = tf.Variable(tf.random_normal([40]))

W2 = tf.Variable(tf.random_normal([4, 2]))
b2 = tf.Variable(tf.random_normal([2]))

W3 = tf.Variable(tf.random_normal([2, 1]))
b3 = tf.Variable(tf.random_normal([1]))
```

그 후, 세 개의 모델을 각각 5000번씩 반복하여 학습시켜, 손실 함수를 비교해보자. 각각 돌려보니, 다음과 같은 결과가 나왔다.

```
4개 : 0.152915
20개 : 0.0291716
40개 : 0.482668
```

확실히 차이가 드러난다. 4개로 구성한 것보다는 훨씬 넓은 20개로 구성한 것이 성능이 크게 좋았고, 40개는 너무 넓었다. 이 결과가 말해주는 것은, __적당히 넓은__ 네트워크의 구성이 성능 향상에 큰 영향을 미친다는 것이다.

### Deep Network

이제는 신경망을 더 깊게 구성해보자. 신경망이 더 깊다는 것은, 중간에 Hidden Layer들을 더 많이 끼워넣는다는 의미이다.

원래대로 3층짜리 신경망과, Hidden Layer를 2층 더 추가한 5층 신경망을 구성해서, 30000번의 반복으로 각각 학습시킨 후 손실 함수를 비교해 보자.

먼저 3층짜리 신경망을 구성해 보자.

```
W1 = tf.Variable(tf.random_normal([2, 4]))
b1 = tf.Variable(tf.random_normal([4]))

W2 = tf.Variable(tf.random_normal([4, 2]))
b2 = tf.Variable(tf.random_normal([2]))

W3 = tf.Variable(tf.random_normal([2, 1]))
b3 = tf.Variable(tf.random_normal([1]))

layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)
layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)
hypothesis = tf.sigmoid(tf.matmul(layer2, W3) + b3)
```

다음으로, Hidden Layer를 2층 더 쌓아보자.

```
W1 = tf.Variable(tf.random_normal([2, 4]))
b1 = tf.Variable(tf.random_normal([4]))

W2 = tf.Variable(tf.random_normal([4, 4]))
b2 = tf.Variable(tf.random_normal([4]))

W3 = tf.Variable(tf.random_normal([4, 4]))
b3 = tf.Variable(tf.random_normal([4]))

W4 = tf.Variable(tf.random_normal([4, 2]))
b4 = tf.Variable(tf.random_normal([2]))

W5 = tf.Variable(tf.random_normal([2, 1]))
b5 = tf.Variable(tf.random_normal([1]))

layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)
layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)
layer3 = tf.sigmoid(tf.matmul(layer2, W3) + b3)
layer4 = tf.sigmoid(tf.matmul(layer3, W4) + b4)
hypothesis = tf.sigmoid(tf.matmul(layer4, W5) + b5)
```

이제 두 모델의 출력 결과를 비교해 보자.

```
3층 : 0.00181582
5층 : 0.00134461
```

실제로 학습을 돌려보면 느끼겠지만, 더욱 많은 층으로 구성된 모델이 최적값을 더 빠르게 찾고, 손실 함수를 더 줄인다. 따라서 __적당히 깊은__ 레이어를 구성하는 것도 모델의 성능에 큰 영향을 끼칠 것을 알 수 있다.

신경망 모델의 Wide함과 Deep함을 적절히 조절해서, 최고의 성능을 만들 수 있도록 튜닝해 보자!
