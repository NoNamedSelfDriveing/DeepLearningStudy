# 퍼셉트론

### 퍼셉트론

퍼셉트론 : 다수의 입력을 신호로 받아 하나의 출력을 발생

퍼셉트론은 입력을 받아서 정해진 수식 또는 규칙에 따라 계산을 한 뒤, 정해진 임계값을 기반으로 출력을 결정한다.
퍼셉트론 하나를 정의해보자.

![eq1](https://latex.codecogs.com/gif.latex?y%20%3D%20%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%200%20%5C%20%5C%20%5C%20%28w_1x_1%20&plus;%20w_2x_2%20%5Cleq%20%5Ctheta%29%20%5C%5C%201%20%5C%20%5C%20%5C%20%28w_1x_1%20&plus;%20w_2x_2%20%3E%20%5Ctheta%29%20%5Cend%7Bmatrix%7D%5Cright.)

이 퍼셉트론에서는 입력 ![eq](https://latex.codecogs.com/png.latex?x_1) 과 ![eq](https://latex.codecogs.com/png.latex?x_2) 를 받아, 가중치 ![eq](https://latex.codecogs.com/png.latex?w_1), ![eq](https://latex.codecogs.com/png.latex?w_2) 를 이용해 퍼셉트론 안에서 연산하여 결과 ![eq](https://latex.codecogs.com/png.latex?y) 를 내어놓고 있다. 이 때, ![eq](https://latex.codecogs.com/png.latex?\theta) 의 값과 연산 결과를 비교하여 출력이 결정되는데, ![eq](https://latex.codecogs.com/png.latex?\theta) 를 __임계값__ 이라고 한다.

가중치는 각 입력 신호가 출력에 주는 영향력을 나타낸 값이다. 즉, 가중치가 클 수록 해당 입력이 더 큰 영향을 미친다는 말이 된다.

### 퍼셉트론을 이용한 논리 게이트
입력 A, B와 출력 C에 대한 AND 게이트의 진리표는 다음과 같다.

A|B|C
-|-|-
0|0|0
0|1|0
1|0|0
1|1|1

이 게이트를 구현하기 위하여, ![eq](https://latex.codecogs.com/png.latex?w_1), ![eq](https://latex.codecogs.com/png.latex?w_2), ![eq](https://latex.codecogs.com/png.latex?\theta) 의 값을 정해야 한다. 일례를 들면, ![eq](https://latex.codecogs.com/png.latex?w_1=0.5), ![eq](https://latex.codecogs.com/png.latex?w_2=0.5), ![eq](https://latex.codecogs.com/png.latex?\theta=0.7) 일 때 알맞은 출력을 내놓는다.

이를 파이썬 코드로 구현해보자.

```
def AND(x1, x2):
	w1, w2, t = 0.5, 0.5, 0.7
    if (w1*x1 + w2*x2) <= t :
    	return 0
    else:
    	return 1
```

OR 게이트도 구현해보자. OR 게이트의 진리표는 다음과 같다.

A|B|C
-|-|-
0|0|0
0|1|1
1|0|1
1|1|1


이 게이트에서는, ![eq](https://latex.codecogs.com/png.latex?w_1=0.5), ![eq](https://latex.codecogs.com/png.latex?w_2=0.5), ![eq](https://latex.codecogs.com/png.latex?\theta=0.1) 일 때 알맞은 출력을 내놓는다.
이도 파이썬 코드로 구현하면 다음과 같다.

```
def OR(A, B):
	w1, w2, t = 0.5, 0,5, 0,1
    if (w1*x1 + w2*x2) <= t:
    	return 0
    else:
    	return 1
```

### 가중치 도입

![eq](https://latex.codecogs.com/png.latex?\theta) 를 ![eq](https://latex.codecogs.com/png.latex?-b) 로 치환하면, 다음과 같이 퍼셉트론을 나타낼 수 있다.

![eq](https://latex.codecogs.com/png.latex?y%20%3D%20%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%200%20%5C%20%5C%20%5C%20%28b&plus;w_1x_1%20&plus;%20w_2x_2%20%5Cleq%200%29%20%5C%5C%201%20%5C%20%5C%20%5C%20%28b&plus;w_1x_1%20&plus;%20w_2x_2%20%3E%200%29%20%5Cend%7Bmatrix%7D%5Cright.)

이를 OR 게이트에서의 코드로 나타내면 다음과 같다.
```
def OR(A, B):
	w1, w2, b = 0.5, 0.5, -0,1
    if (w1*x1 + w2*x2 + b) <= 0:
    	return 0
    else:
    	return 1
```

### 다층 퍼셉트론을 이용한 XOR 게이트

XOR 게이트의 진리표는 다음과 같다.

A|B|C
-|-|-
0|0|0
0|1|1
1|0|1
1|1|0

XOR 게이트를 만들기 위한 ![eq](https://latex.codecogs.com/png.latex?w_1), ![eq](https://latex.codecogs.com/png.latex?w_2), ![eq](https://latex.codecogs.com/png.latex?\theta) 값을 찾으려고 해도, 직선 하나로는 찾을 수 없다. XOR 게이트의 논리식을 보면, 다음과 같다.

![eq](https://latex.codecogs.com/png.latex?C%20%3D%20%28A%20%5Ccdot%20%5Cbar%20B%29%20&plus;%20%28%5Cbar%20A%20%5Ccdot%20B%29)

이미 구현한 AND, OR 게이트를 이용해서 XOR 게이트를 구현할 수 있다. 추가로 NOT 게이트 또한 구현해야 한다. 이 다층 퍼셉트론의 구조를 보면 다음과 같다.

![](../image/XOR_perceptron_network.png)

이를 파이썬으로 직접 구현해 보자.

```
def NOT(x):
	if x==0:
    	return 1
    else
    	return 0
        
def XOR(x1, x2):
	A = AND(x1, NOT(x2))
    B = AND(NOT(x1), x2))
    return OR(A, B)
```

오늘은 퍼셉트론의 개념과, 다층 퍼셉트론을 이용한 간단한 네트워크의 구성을 진행했다.
