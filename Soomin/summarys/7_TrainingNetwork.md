# 신경망의 학습과 손실 함수

### one-hot encoding

먼저, one-hot encoding이란, 정답 또는 정답으로 추정되는 원소만 1, 나머지는 0을 가지게 하는 것을 말한다.

예를 들어, Softmax 함수를 거친 다음 리스트
```
[0.1, 0.1, 0.25, 0.15, 0.1, 0.2, 0.1]
```
를 one-hot encoding하면, 다음과 같이 된다.
```
[0, 0, 1, 0, 0, 0, 0]
```
이는 다중 레이블 분류 문제에서 예측 정답을 찾아내는 데에 활용된다.

### 손실 함수(Loss function)
신경망을 학습한다는 것은, 훈련 데이터로부터 Weight와 Bias를 자동으로 획득하는 과정을 거친다는 것을 말한다. 이 때, 손실 함수는 신경망을 훈련하는 데에 주된 지표가 된다.

손실 함수는 신경망이 현재 가지고 있는 가중치를 이용한 예측 결과가 실제 정답과 얼마나 다른지를 나타내는 함수로, 신경망의 학습은 이 손실 함수를 최소화하는 것을 목표로 한다.

대표적인 손실 함수 2개를 알아보자.

#### 평균 제곱 오차(Mean Square Error)
일단 이 손실 함수를 식으로 나타내면 다음과 같다.

![](https://latex.codecogs.com/gif.latex?E%20%3D%20%5Cfrac%7B1%7D%7B2%7D%20%5Csum_%7Bk%7D%5E%7B%20%7D%20%28y_k-t_k%29%5E2)

이 식에서, ![](https://latex.codecogs.com/gif.latex?t_k)는 ![](https://latex.codecogs.com/gif.latex?k)번째 정답이고, ![](https://latex.codecogs.com/gif.latex?y_k)는 ![](https://latex.codecogs.com/gif.latex?k)번째 예측값을 나타낸다.

제곱을 취하는 이유는, 연산의 결과로 음수가 나옴을 방지하고, 편차를 더 크게 만들어주기 위해서이다.

이를 실제로 구현해 보자.

```
import numpy as np

def ms_error(Y, T):
    return 0.5*np.sum(np.square(Y-T))

T = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
Y = np.array([0.05, 0.05, 0.05, 0.7, 0.05, 0.1, 0, 0, 0, 0])

print(ms_error(Y, T))
```

#### 교차 엔트로피 오차(Cross Entropy Error)

이 오차를 수식으로 나타내면 다음과 같다.

![](https://latex.codecogs.com/gif.latex?E%20%3D%20%5Csum_%7Bk%7D%5E%7B%20%7Dt_k%20%5Clog_e%7By_k%7D)

이제 이것을 실제로 구현해보자. 구현할 때 주의할 점은, log(0)에서 -inf가 발생하지 않게 아주 극미한 값을 더해주는 것이 필요하다는 것이다.


```
import numpy as np

def cross_entropy(Y, T):
    d = 1e-05
    return -np.sum(T * np.log(Y + d))

T = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
Y = np.array([0.1, 0, 0, 0.5, 0.1, 0, 0.1, 0.1, 0, 0])

print(cross_entropy(Y, T))
```
이 코드의 출력은 다음과 같다.
```
0.6931271807
```
이제 예측을 틀리게 만들어보자, 3번 원소가 가장 커야 예측이 맞는 거지만, 임의로 다른 원소를 가장 크게 해보고, 손실 함수의 값 변화를 보자.
```
T = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
Y = np.array([0.1, 0, 0, 0.1, 0.5, 0, 0.1, 0.1, 0, 0])
```
임의로 다른 원소를 정답으로 예측하게 하였다. 결과는 아래와 같이, 손실 함수가 크게 커지는 모습을 보였다.
```
2.30248509799
```

따라서 이 손실 함수의 출력을 최소화하는 것이 신경망 학습의 가장 큰 목표라는 것을 알 수 있다.