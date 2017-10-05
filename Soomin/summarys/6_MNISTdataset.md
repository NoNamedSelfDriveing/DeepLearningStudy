# MNIST �����ͼ�

MNIST�� ������ ����, �ձ۾��� �� ���� �����ͼ�Ʈ�̴�. 0���� 9������ �ձ۾� ���� �����Ͱ� 60,000�� ������, �׽�Ʈ�� �����Ͱ� 10,000�� �ִ�. MNIST�� �����ʹ� ![](https://latex.codecogs.com/gif.latex?%2828%20%5Ctimes%2028%29) ũ���� Gray Scale �̹����̸�, �� �ȼ��� 0~255 ������ ���� ������. �� �� �̹������� �� ���ڸ� �ǹ��ϴ� ���̺��� �پ� �ִ�.

�׷� �� �����͸� ������ ��� ����. �����ͼ��� å�� ������ �޾Ҵ�.

```
import numpy as np
import matplotlib.pyplot as plt
from mnist import load_mnist

(trainX, trainY), (testX, testY) = load_mnist(flatten=True, normalize=False)

print(trainX.shape)
print(trainY.shape)
print(testX.shape)
print(testY.shape)
```

å���� �̸� ������ �� mnist ������ �ٿ�ε� ����� ����ؼ� ����� ��� ���Ҵ�. ����� ������ ����.
```
(60000, 784)
(60000,)
(10000, 784)
(10000,)
```
28 x 28 = 784�̹Ƿ�, �̹����� ��Ȯ�� 60,000���� 10,000���� �ִ� ���� �� �� �ִ�.
�� �� �ϳ��� ���� �׷� ����.

flatten ���ڸ� False�� �༭ 1���� �迭���� ������ �ʰ� �ϰ�, �迭�� reshape�Ͽ� ������ �ٿ���.

```
(trainX, trainY), (testX, testY) = load_mnist(flatten=False, normalize=False)

trainX = np.reshape(trainX, (len(trainX), 28, 28))
print(trainX[0].shape)

plt.imshow(trainX[np.random.randint(60000)], 'gray')
plt.show()
```

60,000 ���� �н� ������ �� �����ϰ� �ϳ��� �����͸� �Ʒ�ó�� �׸����� �����ش�.
![](../image/mnist.PNG)

�� �����ͼ��� �̿��Ͽ� �Ű���� �����, �Է��� 784���̰� ����� 10���� �Ű������ ���� �� ���� ���̴�.
�׸��� Softmax �Լ��� ����Ͽ� ���� �з��� ������ �� ���� ���̴�.