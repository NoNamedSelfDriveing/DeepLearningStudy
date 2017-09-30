# 경사 하강법

경사 하강법(Gradient Descent Method)는, 손실 함수를 최소화 하는 대표적인 방법이다.

함수의 기울기가 가리키는 방향은, 그 지점에서 함수의 출력 값을 가장 크게 줄이는 방향이다. 경사 하강법은 이 특징을 이용하는데, 함수의 현재 위치에서의 기울기를 구해서, 그 방향으로 일정 거리만큼 이동하는 작업을 반복하는 알고리즘이다.

손실 함수의 모든 구간에서의 최솟값을 Global Minimum이라고 한다. 신경망을 학습 할 때, 경사 하강법을 이용하여 손실 함수를 최소화하는 작업은 곧 Global Minimum을 찾는 일과 동일하다. 

변수 ![](https://latex.codecogs.com/gif.latex?x_0), ![](https://latex.codecogs.com/gif.latex?x_1)을 가지고 있는 손실 함수에서의 경사 하강법을 식으로 나타내면 다음과 같다.

![](https://latex.codecogs.com/gif.latex?x_0&space;=&space;x_0-\eta&space;\frac{\partial&space;f}{\partial&space;x_0})

![](https://latex.codecogs.com/gif.latex?x_0&space;=&space;x_0-\eta&space;\frac{\partial&space;f}{\partial&space;x_1})

한 마디로, 각 변수의 현재 기울기를 구해서, 그 값이 손실 함수 ![](https://latex.codecogs.com/gif.latex?f)에 얼마나 영향을 미칠지를 결정하는 함수이다. 여기서 ![](https://latex.codecogs.com/gif.latex?%5Ceta)는, 기울기가 실제로 손실 함수에 미칠 영향을 결정하는 Hyper Parameter로, 매개변수의 변화 크기를 결정한다.

그럼, 신경망에서의 기울기는 어떻게 될 지 알아보자.

가중치가 ![](https://latex.codecogs.com/gif.latex?W), 손실 함수가 ![](https://latex.codecogs.com/gif.latex?L)일때, 이는 실제로

![](https://latex.codecogs.com/gif.latex?W&space;=&space;\begin{pmatrix}&space;w_1_1&space;\&space;w_2_1&space;\&space;w_3_1&space;\\&space;w_1_2&space;\&space;w_2_2&space;\&space;w_3_2&space;\end{pmatrix}&space;,&space;\&space;\&space;\&space;\&space;\&space;\frac{\partial&space;L}{\partial&space;W}&space;=&space;\begin{pmatrix}&space;\frac{\partial&space;L}{\partial&space;w_1_1}&space;\&space;\frac{\partial&space;L}{\partial&space;w_2_1}&space;\&space;\frac{\partial&space;L}{\partial&space;w_3_1}&space;\\&space;\\&space;\frac{\partial&space;L}{\partial&space;w_1_2}&space;\&space;\frac{\partial&space;L}{\partial&space;w_2_2}&space;\&space;\frac{\partial&space;L}{\partial&space;w_3_2}&space;\end{pmatrix}

이다.

이 때 ![](https://latex.codecogs.com/gif.latex?\frac{\partial&space;L}{\partial&space;W})의 각 원소는, ![](https://latex.codecogs.com/gif.latex?w_x_y) 각각이 변화함에 따라 ![](https://latex.codecogs.com/gif.latex?L)에 미치는 영향을 나타낸다.
