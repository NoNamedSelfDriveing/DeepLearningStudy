# 신경망과 활성화 함수

### 신경망

신경망이란, 하나의 입력 층, 여러 은닉 층, 하나의 출력 층으로 이루어진 퍼셉트론들의 네트워크를 말한다. 전형적인 신경망의 형태는 다음과 같다.

![eq](image/Network.png)

### 활성화 함수

[여기](https://github.com/NoNamedSelfDriveing/DeepLearningStudy/blob/master/Soomin/1_Perceptron.md)서 봤던 퍼셉트론의 기본적인 구조는 다음과 같다.

![eq](https://latex.codecogs.com/gif.latex?Y%20%3D%20%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%200%20%5C%20%5C%20%5C%20%28w_1x_1&plus;w_2x_2&plus;b%20%5Cleq%200%29%5C%5C%201%20%5C%20%5C%20%5C%20%28w_1x_1&plus;w_2x_2&plus;b%20%3E%200%29%20%5Cend%7Bmatrix%7D%5Cright.)

여기서, 다음 층으로 신호를 보낼 건지(1), 보내지 않을 건지(0)을 결정하는 분기를 함수로 나타낼 수 있다. ![eq](https://latex.codecogs.com/png.latex?x%20%3D%20w_1x_1&plus;w_2x_2&plus;b) 일때, 함수 ![eq](https://latex.codecogs.com/png.latex?h%28x%29) 를 다음과 같이 표현할 수 있다.

![eq](https://latex.codecogs.com/png.latex?h%28x%29%20%3D%20%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%201%20%5C%20%5C%20%5C%20%28x%5Cleq%200%29%20%5C%5C%200%20%5C%20%5C%20%5C%20%28x%20%3E%201%29%20%5Cend%7Bmatrix%7D%5Cright.)

함수 ![eq](https://latex.codecogs.com/png.latex?h%28x%29) 와 같이, 퍼셉트론의 출력을 결정하는 함수를 활성화 함수라고 한다. 정확히 말해, 입력 신호에 의해서 퍼셉트론이 연산한 결과를 출력 신호로 변환하는 함수이다.

#### 계단 함수

활성화 함수에는 여러 종류가 있는데, 일단 방금 전에 위에서 구현한 함수는 __계단 함수__ 이다. 값에 따라 정확히 계단과 같은 모습을 보인다고 해서 붙은 이름인데, 실제 그래프를 그려보면 다음과 같다.

```
import matplotlib.pyplot as plt
import numpy as np

def fc(x):
    return np.array(x > 0, dtype=np.int)

x = np.arange(-5.0, 5.0, 0.1)
y = fc(x)

plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()
```

![image](https://github.com/MagmaTart/DeepLearningStudy/blob/master/Soomin/image/stairs_function.PNG)

#### 시그모이드 함수

시그모이드(Sigmoid) 함수는 대표적인 __비선형 함수__ 이다. 이름의 뜻은 그저 알파벳 'S' 를 닮았다는 뜻이다. 사실 생긴 것은 계단 함수를 부드럽게 곡선으로 만들어 놓은 것과 비슷하다.

신경망으로 나아가는 중요한 열쇠가 바로 이 __비선형 함수__ 라고 한다. 선형 함수를 활성화 함수로 사용하면 네트워크 레이어를 쌓는 의미가 없다는 이유 때문이라고 한다.

시그모이드 함수의 식은 다음과 같다.

![eq](https://latex.codecogs.com/png.latex?h%28x%29%20%3D%20%5Cfrac%7B1%7D%7B1&plus;e%5E%7B-x%7D%7D)

파이썬으로 이를 직접 구현해서, 그래프를 그려 모양을 확인해 보자.

```
import matplotlib.pyplot as plt
import numpy as np

def fc(x):
    return np.array(1/(1+np.exp(-x)))

x = np.arange(-5.0, 5.0, 0.1)
y = fc(x)

plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()
```

![image](https://github.com/MagmaTart/DeepLearningStudy/blob/master/Soomin/image/sigmoid_function.PNG)

#### ReLU 함수
ReLU(Rectified Linear Unit) 함수는 정말 간단하지만 효과가 크다고 한다. 입력값이 0보다 작으면 0을, 그렇지 않으면 입력값을 그대로 출력하는 간단한 함수이다. 수식으로 나타내면 다음과 같다.

![eq](https://latex.codecogs.com/png.latex?h%28x%29%20%3D%20%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%200%20%5C%20%5C%20%5C%20%28x%5Cleq%200%29%20%5C%5C%20x%20%5C%20%5C%20%5C%20%28x%20%3E%200%29%20%5Cend%7Bmatrix%7D%5Cright.)

직접 Python으로 구현해서, 그래프를 그려 보자.

```
import matplotlib.pyplot as plt
import numpy as np

def fc(x):
    return np.array(np.maximum(0, x))

x = np.arange(-5.0, 5.0, 0.1)
y = fc(x)

plt.plot(x, y)
plt.ylim(-2, 7)
plt.show()
```

![image](https://github.com/MagmaTart/DeepLearningStudy/blob/master/Soomin/image/ReLU_function.PNG)
