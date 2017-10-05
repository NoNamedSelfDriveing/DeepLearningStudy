# 최적화 알고리즘 2. AdaGrad, RMSProp, Adam
신경망의 학습에는 학습률(learning rate)이 중요하다. 학습률 값이 너무 작으면 학습 시간이 너무 길어지고, 너무 크면 학습이 제대로 이뤄질 수 없기 떄문이다. 하이퍼 파라미터인 이 학습률을 효과적으로 정하는 기법 중 하나가 __학습률 감소__ 이다. 학습이 반복될수록 학습률을 점차 낮춰가는 방법을 말한다. 이 방법은 실제 신경망 학습에 자주 쓰인다. 그리고 가중치의 최적화 과정에서 학습률 감소 방법을 반영한 알고리즘들이 오늘 소개할 알고리즘들이다.

### AdaGrad (Adaptive Gradient)
AdaGrad가 학습률 감소를 도입한 기본적인 매개변수 최적화 알고리즘이다. 간단하게, 가중치 매개변수 전체의 학습률을 일괄적으로 낮추는 것이다, AdaGrad는 이를 구현하여, 가중치 매개변수의 각각 원소에 맞는 맞춤형 학습률 값을 만들어 준다. AdaGrad를 수식으로 보면 다음과 같다.

![](https://latex.codecogs.com/gif.latex?h%20%5Cleftarrow%20h&plus;%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20W%7D%20%5Codot%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20W%7D)

![](https://latex.codecogs.com/gif.latex?W%20%5Cleftarrow%20W%20-%20%5Ceta%20%5Cfrac%7B1%7D%7B%5Csqrt%7Bh%7D%7D%20%5Codot%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20W%7D)

![](https://latex.codecogs.com/gif.latex?h)는 일종의 메모리같은 역할을 하며, 이전 기울기들을 제곱한 값을 계속 더해 나간다. 이 때 제곱은 원소간의 제곱이다(Element-wise). 그리고 실제 가중치 매개변수를 수정할 때, 학습률에 ![](https://latex.codecogs.com/gif.latex?%5Cfrac%20%7B1%7D%7B%5Csqrt%7Bh%7D%7D)를 곱해준 값을 실제 학습률로 사용한다. 이 때 학습률은 행렬 형태가 되므로, 기울기 값과 연산하였을 때 모든 원소가 동일한 학습률을 적용받지 않고, 각 원소별로 최적화된 학습률을 적용받을 수 있게 된다.

![](https://latex.codecogs.com/gif.latex?h)의 변화에 따라 각 원소마다 변화량이 달라지는데, 이전 기울기 ![](https://latex.codecogs.com/gif.latex?%5Cfrac%20%7B%5Cpartial%20L%7D%7B%5Cpartial%20W%7D)에서 어떤 원소의 변화가 컸을 경우 ![](https://latex.codecogs.com/gif.latex?h)에서 해당 원소의 값이 커져 해당 매개변수의 학습률이 더 크게 감소하고, 원소의 기울기 변화가 작았을 경우에는 해당 매개변수의 학습률이 더 작게 감소한다.

AdaGrad는 좋은 알고리즘이지만, ![](https://latex.codecogs.com/gif.latex?h)의 값이 계속 커지기만 하므로, 언젠가는 학습률이 0에 수렴해 학습이 진행되지 않는 상황이 발생할 수 있다는 것이다.

### RMSProp
AdaGrad의 위와 같은 문제를 해결하기 위하여 만들어진 알고리즘이 __RMSProp__ 이다. RMSProp의 아이디어는, 먼 과거의 기울기를 서서히 잊고 최근의 새로운 기울기 정보를 학습률 갱신에 크게 반영하자는 것입니다. 그리고 이 과정에서, __지수이동평균(EMA)__ 를 사용한다.

지수이동평균을 잠깐 알아보고 가자. 최신의 정보(값)을 ![](https://latex.codecogs.com/gif.latex?x_n), 과거의 정보(값)들의 종합을 ![](https://latex.codecogs.com/gif.latex?s_n), 0과 1 사이의 값을 가지는 계수를 ![](https://latex.codecogs.com/gif.latex?%5Calpha)로 놓을 때, 업데이트되는 다음의 정보 ![](https://latex.codecogs.com/gif.latex?s_n_&plus;_1)는 다음과 같이 정의된다.

![](https://latex.codecogs.com/gif.latex?s_n_&plus;_1%20%3D%20%5Calpha%20x_n%20&plus;%20%281-%5Calpha%29s_n)

![](https://latex.codecogs.com/gif.latex?%5Calpha)가 0에 가까울 수록 과거의 값만 반영하고, 1에 가까울수록 최신의 값만 반영하게 된다. 보통 0.8이나 0.9에 가까운 값을 사용하는 것 같다.

값이 업데이트될수록 과거의 값들은 ![](https://latex.codecogs.com/gif.latex?%281-%5Calpha%29)라는 계수의 영향을 계속 누적해서 받게 되어 영향이 점점 줄어들게 된다. 하지만 영향이 아예 사라지지는 않는다

이 방법을 기울기와 학습률에 적용하는 것이 RMSProp이다. 아래의 식은, RMSProp의 작동 방식을 보여준다.

![](https://latex.codecogs.com/gif.latex?h%20%5Cleftarrow%20%5Cgamma%20h%20&plus;%20%281-%5Cgamma%29%20%28%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20W%7D%20%5Codot%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20W%7D%29)

![](https://latex.codecogs.com/gif.latex?W%20%5Cleftarrow%20W%20-%20%5Ceta%20%5Cfrac%7B1%7D%7B%5Csqrt%7Bh%7D%7D%20%5Codot%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20W%7D)

학습률의 갱신에 지수이동평균을 적용하고 있다. 지수이동평균의 성질에 의해 과거의 값들은 서서히 잊혀지고, 새로 갱신된 기울기의 제곱 값이 상대적으로 크게 반영되고 있는 것을 볼 수 있다. RMSProp는 이 방법을 사용하여 ![](https://latex.codecogs.com/gif.latex?h)의 값이 무한정으로 커지는 문제를 해결했다.

### Adam
Adam은 직관적으로 보면 Momentum과 RMSProp의 특징을 합쳐놓은 것 같은 알고리즘이다. 매개변수의 탐색 속도를 가속시켜주는 Momentum의 특징과, 매개변수의 각 원소에 알맞는 학습률을 만들어주는 RMSProp의 특징을 합친 알고리즘이다. 자세한 수식은 아직 이해가 부족해서 적지 않지만, 지수이동평균을 RMSProp에 해당하는 변수에만 사용하는 것이 아니라 Momentum의 '속도' 역할 변수를 업데이트할때도 사용한다고 한다.

Adam 알고리즘에 대한 부분은 나중에 더 공부하는 대로 추가하겠다.

### Conclusion
지금까지 다양한 가중치 매개변수 최적화 알고리즘들을 알아보았는데, 대부분 이전의 알고리즘이 가지고 있던 문제를 해결하면서 개발된 알고리즘이라는 것을 알 수 있었다.

또, 적지 않은 상황에서 Adam Optimizer가 크게 활용된다는 사실도 알 수 있었다.
