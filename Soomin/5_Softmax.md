# Softmax 함수

Softmax 함수는, 다중 분류 문제에서 많이 활용되는 활성화 함수이다.

이 함수는 다음과 같은 식으로 나타낼 수 있다. ![](https://latex.codecogs.com/png.latex?n%) : 출력층 노드 수, ![](https://latex.codecogs.com/gif.latex?y_k) : 그중 ![](https://latex.codecogs.com/png.latex?k%)번째 출력, ![](https://latex.codecogs.com/gif.latex?a_k) : ![](https://latex.codecogs.com/png.latex?k%) 번째 입력일 때, 이 함수는
![](https://latex.codecogs.com/png.latex?y_k%3D%5Cfrac%7Be%5E%7Ba_k%7D%7D%7B%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%20e%5E%7Ba_i%7D%7D) 로 나타내어진다.

이를 직접 구현해보자.

```
import numpy as np

def softmax(X):
    return np.array(np.exp(X)/(np.sum(np.exp(X))))

X = np.array([0.2, 0.3, 0.5])
print(softmax(X))
```

결과는 다음과 같다.
```
[ 0.28943311  0.31987306  0.39069383]
```

위 구현의 문제점은, 지수 함수의 특성에 따른 Overflow를 해결하지 않았다는 것이다. 이를 위해, 현재 입력 ![](https://latex.codecogs.com/png.latex?a_k) 에 입력 행렬 중 최대 값을 가지는 원소 ![](https://latex.codecogs.com/png.latex?a_%7Bmax%7D) 를 빼서 다음과 같이 해결해 보자.

![](https://latex.codecogs.com/gif.latex?y_k%20%3D%20%5Cfrac%7Be%5E%7Ba_k-a_%7Bmax%7D%7D%7D%7B%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%20e%5E%7Ba_i-a_%7Bmax%7D%7D%7D)
```
def softmax(X):
    x_exp = X-np.max(X)
    return np.array(np.exp(x_exp)/(np.sum(np.exp(x_exp))))
```

Softmax 함수의 구현 결과에서, 우리는 이 함수의 특성을 발견할 수 있다. Softmax 함수의 결과 행렬의 원소를 모두 더하면 1이 된다. 이것의 의미는, 각 원소의 값을 __그 출력이 선택될 확률__ 로 해석할 수 있다는 뜻이다. 이 때문에 Softmax 함수는 다중 레이블 분류 문제에 많이 사용된다.
