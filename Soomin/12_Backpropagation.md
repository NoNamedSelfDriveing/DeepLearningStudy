# 역전파

역전파(Backpropagation)은, 딥러닝의 기반을 만들어준 중요한 알고리즘 중 하나이다. 이는 다층 신경망이 학습할 수 있도록 기반을 만들어준다. 현재 신경망의 학습 상태를 모든 노드에 전달하고, 또 각각의 변수들이 신경망의 출력에 얼마나 어떻게 영향을 미치는 지 알 수 있도록 한다.

### 합성함수의 미분
함수 ![](https://latex.codecogs.com/gif.latex?f)와 ![](https://latex.codecogs.com/gif.latex?g)가 있을 때, 이 둘의 합성 함수는 ![](https://latex.codecogs.com/gif.latex?%28f%20%5Ccirc%20g%29)로 나타낼 수 있다. 또 합성함수 ![](https://latex.codecogs.com/gif.latex?%28f%20%5Ccirc%20g%29)의 미분 ![](https://latex.codecogs.com/gif.latex?%28f%20%5Ccirc%20g%29%20%5Cprime) 는, 두 함수의 미분의 곱, 즉 ![](https://latex.codecogs.com/gif.latex?f%20%5Cprime%20%5Ctimes%20g%20%5Cprime) 으로 나타낼 수 있다.

![](https://latex.codecogs.com/gif.latex?f%28g%28x%29%29%20%5Cprime%20%3D%20f%28g%28x%29%29%20%5Cprime%20%5Ccdot%20g%28x%29%20%5Cprime)

### 순전파
다음과 같은 계산 그래프가 있다고 하자.

![](image/BackProp1.PNG)

이 그래프의 계산 방향은 다음과 같이 왼쪽의 입력에서부터 오른쪽의 출력으로 흐른다.

![](image/BackProp2.PNG)

그리고 중간 중간 계산 노드의 연산자를 이용해 노드들을 다시 작성해보면, 다음과 같은 함수들의 계산 그래프라는 것을 알 수 있다.

![](image/BackProp3.PNG)

이러한 방향으로 순서대로 진행이 되는 계산을 순전파(Forward Propagation)라고 한다.

### 역전파

위에서 본 순전파로는, 신경망의 학습을 수행할 수 없다. 계산의 흐름이 뒤에서 앞으로 흐르기 때문에, 최종적인 출력에 중간 연산들이 미치는 영향을 계산할 수가 없기 때문이다. 그래서 1987년, 그 영향들을 계산하기 위한 새로운 알고리즘이 고안되었는데, 그것이 바로 역전파(Backpropagation)이다.

역전파는, 계산 그래프의 방향을 거꾸로 거슬러 올라가면서, 각각 계산 그래프 출력이 신경망의 최종 출력에 미치는 영향을 계산하는 것이다. 계산 그래프 내에서 하나의 노드는 마치 하나의 함수와 같다. 순전파와 달리, 계산의 방향은 아래와 같이 진행된다.

![](image/BackProp4.PNG)

이제 맨 끝 출력에서부터, 각 간선의 값이 최종 출력에 미치는 영향을 계산한다. 이는 보통 최종 결과에 대한 해당 값의 편미분을 의미한다. 먼저 ![](https://latex.codecogs.com/gif.latex?R)이 최종 출력 ![](https://latex.codecogs.com/gif.latex?R)에 미치는 영향인 ![](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20R%7D%7B%5Cpartial%20R%7D) 을 계산한다.

![](image/BackProp5.PNG)

이제, 더 앞에 있는 노드들로 순차적으로 거슬러 올라간다. 이 때, __연쇄 법칙(Chain Rule)__ 이라는 방법을 사용한다.

위의 그래프에서, 덧셈 노드의 출력 ![](https://latex.codecogs.com/gif.latex?w)가 최종 출력에 미치는 영향은 __덧셈 함수와 제곱근 함수의 합성 함수의 미분__ 일 것이다. ![](https://latex.codecogs.com/gif.latex?w)가 입력된 덧셈 함수의 출력이 다시 제곱근 함수로 입력되기 때문이다.

글의 맨 처음에서 보았듯이, 합성 함수의 미분은 두 함수의 미분의 곱이다. 따라서 덧셈 노드의 출력 ![](https://latex.codecogs.com/gif.latex?w)가 제곱근 노드의 출력 ![](https://latex.codecogs.com/gif.latex?R)에 미치는 영향은 ![](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20R%7D%7B%5Cpartial%20w%7D%5Ccdot%20%5Cfrac%7B%5Cpartial%20R%7D%7B%5Cpartial%20R%7D%20%3D%20%5Cfrac%7B%5Cpartial%20R%7D%7B%5Cpartial%20w%7D) 처럼 계산할 수 있다.

이와 마찬가지로, 모든 간선의 값이 출력에 미치는 영향을 다음과 같이 계산할 수 있다.

![](image/BackProp6.PNG)

![](https://latex.codecogs.com/gif.latex?a)의 예를 들어보자. ![](https://latex.codecogs.com/gif.latex?a)가 출력 ![](https://latex.codecogs.com/gif.latex?R)에 미치는 영향은, 입력 노드, 덧셈 노드, 제곱근 노드의 합성 함수의 미분으로 생각할 수 있다. 이는 덧셈 노드의 출력 ![](https://latex.codecogs.com/gif.latex?w)가 ![](https://latex.codecogs.com/gif.latex?R)에 미치는 영향인 ![](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20R%7D%7B%5Cpartial%20w%7D%20%5Ccdot%20%5Cfrac%20%7B%5Cpartial%20R%7D%7B%5Cpartial%20R%7D) 에 ![](https://latex.codecogs.com/gif.latex?a)가 ![](https://latex.codecogs.com/gif.latex?w)에 미치는 영향인 ![](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20w%7D%7B%5Cpartial%20a%7D)를 곱한 ![](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20w%7D%7B%5Cpartial%20a%7D%20%5Ccdot%20%5Cfrac%20%7B%5Cpartial%20R%7D%7B%5Cpartial%20w%7D%20%5Ccdot%20%5Cfrac%20%7B%5Cpartial%20R%7D%7B%5Cpartial%20R%7D%20%3D%20%5Cfrac%20%7B%5Cpartial%20R%7D%7B%5Cpartial%20a%7D) 로 나타낼 수 있다.

연쇄 법칙을 적용해 얻을 수 있는 하나의 사실은, 각 노드들의 국소적인 미분들을 이용하여 신경망 전체의 미분을 모두 계산할 수 있다는 점이다. 그리고 이는, 여러 값들과 연계된 신경망의 출력 하나를 이용하여 모든 값들을 최적화시킬 수 있다는 것을 의미한다.
