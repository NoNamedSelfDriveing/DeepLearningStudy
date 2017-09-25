# ��� �ϰ���

��� �ϰ���(Gradient Descent Method)��, �ս� �Լ��� �ּ�ȭ �ϴ� ��ǥ���� ����̴�.

�Լ��� ���Ⱑ ����Ű�� ������, �� �������� �Լ��� ��� ���� ���� ũ�� ���̴� �����̴�. ��� �ϰ����� �� Ư¡�� �̿��ϴµ�, �Լ��� ���� ��ġ������ ���⸦ ���ؼ�, �� �������� ���� �Ÿ���ŭ �̵��ϴ� �۾��� �ݺ��ϴ� �˰����̴�.

�ս� �Լ��� ��� ���������� �ּڰ��� Global Minimum�̶�� �Ѵ�. �Ű���� �н� �� ��, ��� �ϰ����� �̿��Ͽ� �ս� �Լ��� �ּ�ȭ�ϴ� �۾��� �� Global Minimum�� ã�� �ϰ� �����ϴ�. 

���� ![](https://latex.codecogs.com/gif.latex?x_0), ![](https://latex.codecogs.com/gif.latex?x_1)�� ������ �ִ� �ս� �Լ������� ��� �ϰ����� ������ ��Ÿ���� ������ ����.

![](https://latex.codecogs.com/gif.latex?x_0%20%3D%20x_0%20-%20%5Ceta%20%5Cfrac%7B%5Cdelta%20f%7D%7B%5Cdelta%20x_0%7D)

![](https://latex.codecogs.com/gif.latex?x_1%20%3D%20x_1%20-%20%5Ceta%20%5Cfrac%7B%5Cdelta%20f%7D%7B%5Cdelta%20x_1%7D)

�� �����, �� ������ ���� ���⸦ ���ؼ�, �� ���� �ս� �Լ� ![](https://latex.codecogs.com/gif.latex?f)�� �󸶳� ������ ��ĥ���� �����ϴ� �Լ��̴�. ���⼭ ![](https://latex.codecogs.com/gif.latex?%5Ceta)��, ���Ⱑ ������ �ս� �Լ��� ��ĥ ������ �����ϴ� Hyper Parameter��, �Ű������� ��ȭ ũ�⸦ �����Ѵ�.

�׷�, �Ű�������� ����� ��� �� �� �˾ƺ���.

����ġ�� ![](https://latex.codecogs.com/gif.latex?W), �ս� �Լ��� ![](https://latex.codecogs.com/gif.latex?L)�϶�, �̴� ������

![](https://latex.codecogs.com/gif.latex?W%20%3D%20%5Cbegin%7Bpmatrix%7D%20w_1_1%20%5C%20w_2_1%20%5C%20w_3_1%20%5C%5C%20w_1_2%20%5C%20w_2_2%20%5C%20w_3_2%20%5Cend%7Bpmatrix%7D%2C%20%5C%20%5C%20%5C%20%5C%20%5C%20%5Cfrac%7B%5Cdelta%20L%7D%7B%5Cdelta%20W%7D%20%3D%20%5Cbegin%7Bpmatrix%7D%20%5Cfrac%7B%5Cdelta%20L%7D%7B%5Cdelta%20w_1_1%7D%20%5C%20%5Cfrac%7B%5Cdelta%20L%7D%7B%5Cdelta%20w_2_1%7D%20%5C%20%5Cfrac%7B%5Cdelta%20L%7D%7B%5Cdelta%20w_3_1%7D%20%5C%5C%20%5C%5C%20%5Cfrac%7B%5Cdelta%20L%7D%7B%5Cdelta%20w_1_2%7D%20%5C%20%5Cfrac%7B%5Cdelta%20L%7D%7B%5Cdelta%20w_2_2%7D%20%5C%20%5Cfrac%7B%5Cdelta%20L%7D%7B%5Cdelta%20w_3_2%7D%20%5Cend%7Bpmatrix%7D)

�̴�.

�� �� ![](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cdelta%20L%7D%7B%5Cdelta%20W%7D)�� �� ���Ҵ�, ![](https://latex.codecogs.com/gif.latex?w_x_y) ������ ��ȭ�Կ� ���� ![](https://latex.codecogs.com/gif.latex?L)�� ��ġ�� ������ ��Ÿ����.