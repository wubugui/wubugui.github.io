---
layout: post
title: "机器学习初步"
modified:
categories: blog
excerpt:
tags: [数学,概率与统计]
image:
  feature:
date: 2017-01-31T08:08:50-04:00
---

>都是https://www.coursera.org/learn/machine-learning的笔记。

# 基本学习分类

**监督学习（Supervised  Learning）**：给定一个数据集，并且已经知道正确输出看起来是什么样子，以及知道输入和输出之间的关系，也就是研究一个特定的问题。对比于无监督学习看则更加清晰。
-  回归问题：映射输入到连续函数。
-  分类问题：映射输入到离散函数。


**无监督学习（Unsupervised  Learning）**：更像是一个开放性问题，没有正确的答案来座位反馈，所有的关系都需要在学习过程中进行归纳。
-  Clustering：相同类型分组。
-  Non-clustering：自行在数据中提取结构。（比如：在鸡尾酒环境中提取出清晰的人声和音乐。）


# 梯度下降方法

一张图解释原理：
![](http://www.52ml.net/wp-content/uploads/2016/11/1042406-20161017221342935-1872962415.png)

梯度下降不一定能够找到全局的最优解，有可能是一个局部最优解。当然，如果损失函数是凸函数，梯度下降法得到的解就一定是全局最优解。

## 代数描述

设一个线性模型：$h_\theta(x_1, x_2, …x_n) = \theta_0  \theta_{1}x_1 …  \theta_{n}x_{n}$.

cost function为：
$$J(\theta_0, \theta_1…, \theta_n) = \sum\limits_{i=0}^{m}(h_\theta(x_0, x_1, …x_n) – y_i)^2$$

在没有任何先验知识的时候我们暂且将所有$\theta$初始化为0，步长$\alpha$初始化为1.

算法过程：
-  对所有$\theta$准备梯度：$\frac{\partial}{\partial\theta_i}J(\theta_0, \theta_1…, \theta_n)$.
-  用步长乘以上述结果
-  确定下降距离是否都小于预先设定的$\epsilon$.如果小于，精度已经足够，终止算法，当前的所有$\theta$就是结果。否则进入下一步。
-  **同步更新**（注意要同步更新，不能将更新过的$\theta$带入计算接下来的$\theta$）:
$$\theta_i = \theta_i – \alpha\frac{\partial}{\partial\theta_i}J(\theta_0, \theta_1…, \theta_n)$$
结束后，返回第一步。

## 矩阵描述

...

## 梯度下降变种

### Batch Gradient Descent（BGD）

$$\theta_i = \theta_i – \alpha\sum\limits_{j=0}^{m}(h_\theta(x_0^{j}, x_1^{j}, …x_n^{j}) – y_j)x_i^{j}$$

### Stochastic Gradient Descent（SGD）

随机选择样本j：
$$\theta_i = \theta_i – \alpha h_\theta(x_0^{j}, x_1^{j}, …x_n^{j}) – y_j)x_i^{j}$$

### Mini-batch Gradient Descent（MBGD）

此变种调和了BGD和SGD矛盾，得到了更加中庸的方法，也就是对于m个样本，我们采用k个样子来迭代，1<k<m。一般可以取x=10，当然根据样本的数据，可以调整这个x的值。对应的更新公式是：
$$\theta_i = \theta_i – \alpha \sum\limits_{j=t}^{t_k-1}(h_\theta(x_0^{j}, x_1^{j}, …x_n^{j}) – y_j)x_i^{j}$$


# Normal Equation

使用矩阵表示方程，可以使用解析方法来直接解出目标值：$\theta = (X^T X)^{-1}X^T y$

例子：
$m=4$

|$x_0$|Size($feet^2$) $x_1$|Number of bedrooms $x_2$|Number of floors $x_3$|Age of home(years) $x_4$|Price(1000 刀) $y$|
|:----:|:----:|:------:|:------:|:-----:|:------:|
|1|2104|5|1|45|460|
|1|1416|3|2|40|232|
|1|1534|3|2|30|315|
|1|852|2|1|36|178|

$$X=\begin{bmatrix}1 &2104&5&1&45 \newline 1&1416&3&2&40 \newline 1&1534&3&2&30 \newline 1&852&2&1&36\end{bmatrix}$$

$$y = \begin{bmatrix}460 \newline 232 \newline 315 \newline 178\end{bmatrix}$$

$$\theta = (X^T X)^{-1}X^T y$$


如果一个矩阵$(X^T X)$不可逆：
-  Redundant features, where two features are very closely related (i.e. they are linearly dependent)
-  Too many features (e.g. m ≤ n). In this case, delete some features or use "regularization" (to be explained in a later lesson).

**Solutions** to the above problems include deleting a feature that is linearly dependent with another or deleting one or more features when there are too many features.

# 多元线性回归

>$x^(i)_j = $value of feature $j$ in the $i^th$ training example
$x^(i) =$the column vector of all the feature inputs of the $i^th$ training example
  $m=$the number of training examples
  $n=∣∣x^(i)∣∣;$(the number of features)

hypothesis function:

$$h_\theta (x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_3 + \cdots + \theta_n x_n$$

设有$x_0^(i)=1$，则：

$$\begin{align*}h_\theta(x) =\begin{bmatrix}\theta_0 \hspace{2em} \theta_1 \hspace{2em} ... \hspace{2em} \theta_n\end{bmatrix}\begin{bmatrix}x_0 \newline x_1 \newline \vdots \newline x_n\end{bmatrix}= \theta^T x\end{align*}$$

训练样例以row-wise储存在X中，$x_0^(i)=1$使得下列计算相对方便一些:

>
$$\begin{align*}X = \begin{bmatrix}x^{(1)}_0 & x^{(1)}_1 \newline x^{(2)}_0 & x^{(2)}_1 \newline x^{(3)}_0 & x^{(3)}_1 \end{bmatrix}&,\theta = \begin{bmatrix}\theta_0 \newline \theta_1 \newline\end{bmatrix}\end{align*}$$

$$h_\theta(X) = X \theta$$

## 多元线性回归的梯度下降

$$\begin{align*}& \text{repeat until convergence:} \; \lbrace \newline \; & \theta_j := \theta_j - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)} \; & \text{for j := 0...n}\newline \rbrace\end{align*}$$

### Feature Scaling

如果样本项之间的范围相差过大，可能造成cost function的轮廓图呈扁平状态，从而导致梯度下降效率低下，下降路线可能会成密集的锯齿状。于是，我们是用Feature Scaling来将项目缩放到相似的区域：

**feature scaling**：dividing the input values by the range (i.e. the maximum value minus the minimum value) of the input variable, resulting in a new range of just 1.（1是不重要的，只要在这个数量级均可。）

**mean normalization** : 使用下列公式缩放：
$$x_i := \dfrac{x_i - \mu_i}{s_i}$$
$\mu_i$是所有特征i的平均值，$s_i$是范围或者标准差。



### Feature Combine & Polynomial Regression

有时候不一定要使用某一个给定的feature，而是可以将若干个feature结合起来，比如将$x_1、x_2$结合为$x_1·x_2$。

hypothesis function不一定要是线性的，可以使用quadratic, cubic or square root function (or any other form).
比如：

quadratic function：$h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_1^2$

cubic function：$h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_1^2 + \theta_3 x_1^3$

square root function:$h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 \sqrt{x_1}$

使用Feature Combine，feature范围就很重要了

eg. if $x_1$ has range $1 - 1000$ then range of $x^2_1$ becomes $1 - 1000000$ and that of $x^3_1$ becomes $1 - 1000000000$.

# 逻辑回归

对于分类问题，有时候可以使用线性回归来进行拟合，通过大于0.5是1，小于等于就是0来实现。但是这种方法对于非线性函数工作的不好。

把y变成若干离散值，成为逻辑回归问题，它和线性回归很类似。

## Binary classification problem

### Hypothesis function

我们使用"Sigmoid Function," also called the "Logistic Function":
![](https://github.com/wubugui/FXXKTracer/raw/master/pic/Logistic_function.png)

$$\begin{align*}& h_\theta (x) = g ( \theta^T x ) \newline \newline& z = \theta^T x & (dot(\theta,x)) \newline& g(z) = \dfrac{1}{1 + e^{-z}}\end{align*}$$

$h_\theta(x)$给出了输出1的概率:

$$\begin{align*}& h_\theta(x) = P(y=1 | x ; \theta) = 1 - P(y=0 | x ; \theta) \newline& P(y = 0 | x;\theta) + P(y = 1 | x ; \theta) = 1\end{align*}$$

hypothesis function指使了下列性质：

$$\begin{align*}& h_\theta(x) \geq 0.5 \rightarrow y = 1 \newline& h_\theta(x) < 0.5 \rightarrow y = 0 \newline\end{align*}$$

并且：

$$\begin{align*}& g(z) \geq 0.5 \newline& when \; z \geq 0\end{align*}$$

同时：

$$\begin{align*}z=0, e^{0}=1 \Rightarrow g(z)=1/2\newline z \to \infty, e^{-\infty} \to 0 \Rightarrow g(z)=1 \newline z \to -\infty, e^{\infty}\to \infty \Rightarrow g(z)=0 \end{align*}$$

因此我们可以检查$g(x)$中的x，来得到函数结果，也就是：
$$\begin{align*}& \theta^T x \geq 0 \Rightarrow y = 1 \newline& \theta^T x < 0 \Rightarrow y = 0 \newline\end{align*}$$

如果我们指定$g(x)$中的$x=0$，则得到了decision boundary方程，这条边界将数据区域分为$y=0$,$y=1$两个区域。

### Cost Function

$$\begin{align*}& J(\theta) = \dfrac{1}{m} \sum_{i=1}^m \mathrm{Cost}(h_\theta(x^{(i)}),y^{(i)}) \newline & \mathrm{Cost}(h_\theta(x),y) = -\log(h_\theta(x)) \; & \text{if y = 1} \newline & \mathrm{Cost}(h_\theta(x),y) = -\log(1-h_\theta(x)) \; & \text{if y = 0}\end{align*}$$

![](https://github.com/wubugui/FXXKTracer/raw/master/pic/Logistic_regression_cost_function_positive_class.png)

![](https://github.com/wubugui/FXXKTracer/raw/master/pic/Logistic_regression_cost_function_negative_class.png)

几条性质：

$$\begin{align*}& \mathrm{Cost}(h_\theta(x),y) = 0 \text{ if } h_\theta(x) = y \newline & \mathrm{Cost}(h_\theta(x),y) \rightarrow \infty \text{ if } y = 0 \; \mathrm{and} \; h_\theta(x) \rightarrow 1 \newline & \mathrm{Cost}(h_\theta(x),y) \rightarrow \infty \text{ if } y = 1 \; \mathrm{and} \; h_\theta(x) \rightarrow 0 \newline \end{align*}$$

可以把两个Cost函数合成一个：

$$\mathrm{Cost}(h_\theta(x),y) = - y \; \log(h_\theta(x)) - (1 - y) \log(1 - h_\theta(x))$$

最终整个J就如下：

$$J(\theta) = - \frac{1}{m} \displaystyle \sum_{i=1}^m [y^{(i)}\log (h_\theta (x^{(i)})) + (1 - y^{(i)})\log (1 - h_\theta(x^{(i)}))]$$

向量化：

$$\begin{align*} & h = g(X\theta)\newline & J(\theta) = \frac{1}{m} \cdot \left(-y^{T}\log(h)-(1-y)^{T}\log(1-h)\right) \end{align*}$$

此函数的梯度下降公式和线性回归的梯度下降公式具有相同的形式：

$$\begin{align*} & Repeat \; \lbrace \newline & \; \theta_j := \theta_j - \frac{\alpha}{m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)} \newline & \rbrace \end{align*}$$

向量化：

$$\theta := \theta - \frac{\alpha}{m} X^{T} (g(X \theta ) - \vec{y})$$

We still have to simultaneously update all values in theta.

### 实现

使用matlab实现以下cost函数和梯度(注意正确的向量化)：
```matlab
%X包含1，theta包含x0
ldm =  lambda/m;
dm = ldm/lambda;
m1 = length(theta);
sig = sigmoid(X*theta);
grad = dm.*X'*(sig - y) + ldm.*theta;
 %第一项求导没有尾巴，需要把尾巴减掉
grad(1,:) = grad(1,:) -  ldm*theta(1,:);
 %注意cost的最后规则化求和里面不包括第一项，所以theta0求导没有最后一个尾巴
J = -dm.*(log(sig)'*y+log(1-sig)'*(1-y)) + ldm*0.5* sum(theta(2:m1,:).^2);
%return J,grad

%训练一组预测num_label个数字的参数
function [all_theta] = oneVsAll(X, y, num_labels, lambda)
    m = size(X, 1);n = size(X, 2);
    all_theta = zeros(num_labels, n + 1);
    X = [ones(m, 1) X];
    initial_theta = zeros(n + 1, 1);
    options = optimset('GradObj', 'on', 'MaxIter', 50);
    for i=1:num_labels
         %y==i为逻辑数组
         [theta] = fmincg (@(t)(lrCostFunction(t, X, (y == i),  lambda)),initial_theta, options);
         %返回的是列向量
         all_theta(i,:) = theta';
    end
end

%预测
function p = predictOneVsAll(all_theta, X)
    m = size(X, 1);
    num_labels = size(all_theta, 1);
    p = zeros(size(X, 1), 1);
    X = [ones(m, 1) X];
    Mat = all_theta*X';
    for i=1:m
        [~,pos] = max(Mat(:,i));
        p(i,:) = pos;
    end
end

%Training Set Accuracy: 94.960000
```
>**向量化应该思考为：在矩阵里面逐元素操作。**

### 高级优化

首先需要提供函数计算：

$$\begin{align*} & J(\theta) \newline & \dfrac{\partial}{\partial \theta_j}J(\theta)\end{align*}$$


octave代码类似下面这样(这个没在matlab测试过，可能函数会有些不同)：
```matlab
function [jVal, gradient] = costFunction(theta)
  jVal = [...code to compute J(theta)...];
  gradient = [...code to compute derivative of J(theta)...];
end
```
使用"fminunc()"优化算法（这个函数用于最小值优化也就是求最小值）：

```matlab
options = optimset('GradObj', 'on', 'MaxIter', 100);
initialTheta = zeros(2,1);
   [optTheta, functionVal, exitFlag] = fminunc(@costFunction, initialTheta, options);
```

## Multiclass Classification: One-vs-all

把Binary classification拓展到Multiclass Classification，如果有$y={1,2,3,...,n}$，就将问题分成n个Binary classification.

$$\begin{align*}& y \in \lbrace0, 1 ... n\rbrace \newline& h_\theta^{(0)}(x) = P(y = 0 | x ; \theta) \newline& h_\theta^{(1)}(x) = P(y = 1 | x ; \theta) \newline& \cdots \newline& h_\theta^{(n)}(x) = P(y = n | x ; \theta) \newline& \mathrm{prediction} = \max_i( h_\theta ^{(i)}(x) )\newline\end{align*}$$

We are basically choosing one class and then lumping all the others into a single second class. We do this repeatedly, applying binary logistic regression to each case, and then use the hypothesis that returned the highest value as our prediction.

## Overfitting问题

![](https://github.com/wubugui/FXXKTracer/raw/master/pic/0cOOdKsMEeaCrQqTpeD5ng_2a806eb8d988461f716f4799915ab779_Screenshot-2016-11-15-00.23.30.png)

上图展示分别是underfitting，normal，overfitting的情况。

Underfitting, or high bias, is when the form of our hypothesis function h maps poorly to the trend of the data. It is usually caused by a function that is too simple or uses too few features. 

At the other extreme, overfitting, or **high variance**, is caused by a hypothesis function that fits the available data but does not generalize well to predict new data. It is usually caused by a complicated function that creates a lot of unnecessary curves and angles unrelated to the data.

This terminology is applied to both linear and logistic regression. There are two main options to address the issue of overfitting:

1) Reduce the number of features:

-  手动选择feature保留。
-  Use a model selection algorithm 

2) Regularization

-  保持所有的features, 但是减少参数θj的值.
-  Regularization works well 当我们有大量影响甚微的features时.

>Error due to Variance ： 在给定模型数据上预测的变化性，你可以重复整个模型构建过程很多次，variance 就是衡量每一次构建模型预测相同数据的变化性。图形上，如图所示，图形中心是模型完美正确预测数据值，当我们远离中心预测越来越差，我们可以重复整个模型构建过程多次，通过每一次命中图形来表示bias and variance：
![](http://img.blog.csdn.net/20141124233425122?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvaHVydXp1bg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)
??????????????????????????????????????


### 规则化后的线性回归

使用规则化后的Cost函数：

$$min_\theta\ \dfrac{1}{2m}\ \left[ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2 + \lambda\ \sum_{j=1}^n \theta_j^2 \right]$$

$\lambda$具有平滑原来函数的效果，并且越大平滑效果越好。

新的梯度下降：

$$\begin{align*} & \text{Repeat}\ \lbrace \newline & \ \ \ \ \theta_0 := \theta_0 - \alpha\ \frac{1}{m}\ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_0^{(i)} \newline & \ \ \ \ \theta_j := \theta_j - \alpha\ \left[ \left( \frac{1}{m}\ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)} \right) + \frac{\lambda}{m}\theta_j \right] &\ \ \ \ \ \ \ \ \ \ j \in \lbrace 1,2...n\rbrace\newline & \rbrace \end{align*}$$

后者也可以写为：
$$\theta_j := \theta_j(1 - \alpha\frac{\lambda}{m}) - \alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)}$$

Normal Equation变为：

$$\begin{align*}& \theta = \left( X^TX + \lambda \cdot L \right)^{-1} X^Ty \newline& \text{where}\ \ L = \begin{bmatrix} 0 & & & & \newline & 1 & & & \newline & & 1 & & \newline & & & \ddots & \newline & & & & 1 \newline\end{bmatrix}\end{align*}$$

### 规则化后的逻辑回归

Cost函数：

$$J(\theta) = - \frac{1}{m} \sum_{i=1}^m \large[ y^{(i)}\ \log (h_\theta (x^{(i)})) + (1 - y^{(i)})\ \log (1 - h_\theta(x^{(i)}))\large] + \frac{\lambda}{2m}\sum_{j=1}^n \theta_j^2$$

新加入的和式意味着惩罚bias term。

梯度下降：


$$\begin{align*} & \text{Repeat}\ \lbrace \newline & \ \ \ \ \theta_0 := \theta_0 - \alpha\ \frac{1}{m}\ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_0^{(i)} \newline & \ \ \ \ \theta_j := \theta_j - \alpha\ \left[  \frac{1}{m}\ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)} + \frac{\lambda}{m}\theta_j \right] &\ \ \ \ \ \ \ \ \ \ j \in \lbrace 1,2...n\rbrace\newline & \rbrace \end{align*}$$

# 神经网络

## 基本概念

神经输入电信号channel到输出。在神经网络中，输入的feature$x_1...x_n$就是神经的输入信号，输出信号是hypothesis function的输出，注意这里输出的是一个函数，和前面的内容差不多。

$x_0$叫做“bias unit”，总是等于1.
使用前面classification中相同的logistic function$\frac{1}{1+e^{\theta T_x}}$,这叫做sigmoid (logistic) activation.这里面的$\theta$就叫做weights.

神经网络一个非常简单的表示：

$$\begin{bmatrix}x_0 \newline x_1 \newline x_2 \newline \end{bmatrix}\rightarrow\begin{bmatrix}\ \ \ \newline \end{bmatrix}\rightarrow h_\theta(x)$$

输入节点（layer 1）叫做“输入层”，这一层输入到（layer 2），最终输出hypothesis function，叫做“输出层”。
$[]$所包含的是“隐藏层”。这些内部隐藏层表示为：

$$\begin{bmatrix}x_0 \newline x_1 \newline x_2 \newline x_3\end{bmatrix}\rightarrow\begin{bmatrix}a_1^{(2)}&...& a_1^{(n)} \newline a_2^{(2)}&...& a_2^{(n)} \newline a_3^{(2)}&...& a_2^{(n)} \newline \end{bmatrix}\rightarrow h_\theta(x)$$

其中，$a_i^{(j)}$叫做"activation units"，中的i表示第几个单位，j表示层数。

为了把地j层映射到j+1层，使用权重矩阵$\Theta^{j}$.
$$\begin{bmatrix}\Theta_{10}^{(1)} & \Theta_{11}^{(1)} & \Theta_{12}^{(1)} & \Theta_{13}^{(1)}\newline \Theta_{20}^{(1)} & \Theta_{21}^{(1)} & \Theta_{22}^{(1)} & \Theta_{23}^{(1)} \newline \Theta_{30}^{(1)} & \Theta_{31}^{(1)} & \Theta_{32}^{(1)} & \Theta_{33}^{(1)} \end{bmatrix}$$

>注意：左乘矩阵

每一项$\Theta_{ts}^l$中，t表示映射目标unit，s表示要作用的源输入，l表示来自哪一层。

神经网络
$$\begin{bmatrix}x_0 \newline x_1 \newline x_2 \newline x_3\end{bmatrix}\rightarrow\begin{bmatrix}a_1^{(2)} \newline a_2^{(2)} \newline a_3^{(2)} \newline \end{bmatrix}\rightarrow h_\theta(x)$$
的映射关系展开如下：

$$\begin{align} a_1^{(2)} = g(\Theta_{10}^{(1)}x_0 + \Theta_{11}^{(1)}x_1 + \Theta_{12}^{(1)}x_2 + \Theta_{13}^{(1)}x_3) \newline a_2^{(2)} = g(\Theta_{20}^{(1)}x_0 + \Theta_{21}^{(1)}x_1 + \Theta_{22}^{(1)}x_2 + \Theta_{23}^{(1)}x_3) \newline a_3^{(2)} = g(\Theta_{30}^{(1)}x_0 + \Theta_{31}^{(1)}x_1 + \Theta_{32}^{(1)}x_2 + \Theta_{33}^{(1)}x_3) \newline h_\Theta(x) = a_1^{(3)} = g(\Theta_{10}^{(2)}a_0^{(2)} + \Theta_{11}^{(2)}a_1^{(2)} + \Theta_{12}^{(2)}a_2^{(2)} + \Theta_{13}^{(2)}a_3^{(2)}) \newline \end{align}$$

如果神经网络的第$j$层有$s_j$个单位，第$j+1$层有$s_{j + 1}$个单位，那么映射矩阵维度为：$s_{j+1} \times (s_j + 1)$(也就是$s_{j+1}$行$s_j + 1$列)(**左乘情形下**).其中的$s_j + 1$的$+1$来自于"bias nodes",$x_0$ and $\Theta_0^{(j)}$.

>例子：
对如下网络使用权重矩阵$\Theta^{(1)} =\begin{bmatrix} -10 & 20 & 20\end{bmatrix}$可以得到$OR$函数。
$$\begin{align}\begin{bmatrix}x_0 \newline x_1 \newline x_2\end{bmatrix} \rightarrow\begin{bmatrix}g(z^{(2)})\end{bmatrix} \rightarrow h_\Theta(x)\end{align}$$
$\Theta^{(1)} =\begin{bmatrix}-30 & 20 & 20\end{bmatrix}$可以得到$AND$函数。

## Multiclass Classification

将输出的元素搞成不止一个单元，也就是映射到$R^n$来处理多维分类问题。

## 简单实现


![](https://github.com/wubugui/FXXKTracer/raw/master/pic/nn.png)

前面实现了使用逻辑回归的多类分类器，但是逻辑回归不能表示更复杂的hypothesis function，只能实现线性分类器。
现在先使用已经训练好的参数来实现前馈传播算法（feedforward propagation algorithm）。
```matlab
function p = predict(Theta1, Theta2, X)
    m = size(X, 1);
    num_labels = size(Theta2, 1);
    p = zeros(size(X, 1), 1);
    l = ones(m,1);
    X = [l,X];
    A = Theta1*X';
    A = sigmoid(A)';
    A = [l,A];
    A = Theta2*A';
    A = sigmoid(A);
    [~,p] = max(A',[],2);
    %就是做矩阵乘法然后在结果中挑一个值最大的，位置即为值，矩阵转置过来转置过去可能性能很低。
end
```

## Cost Function

$$J(\Theta) = -\frac{1}{m}\sum_{i=1}^m\sum_{k=1}^K [ y_k^{(i)}\log((h_{\Theta}(x^{(i)}))_k) + (1-y_k^{(i)}) \log(1-(h_{\Theta}(x^{(i)}))_k)]+ \frac{\lambda}{2m}\sum_{l=1}^{L-1}\sum_{i=1}^{s_l}\sum_{j=1}^{s_{l+1}}(\Theta_{j,i}^{(l)})^2$$


-   $L$:网络总层数.
-   $s_l$:第l层中不包括bias unit的单元数.
-   $K$:Number of output units/classes.

>
**Note**:
>-   the double sum simply adds up the logistic regression costs calculated for each cell in the output layer
>-   the triple sum simply adds up the squares of all the individual Θs in the entire network.
>-   the i in the triple sum does not refer to training example i.

### 后向传播求梯度

目标是计算$\min_\Theta J(\Theta)$，先计算$J(\Theta)$的梯度：
$$\dfrac{\partial}{\partial \Theta_{i,j}^{(l)}}J(\Theta)$$

使用下列Back propagation Algorithm：
-   Given training set $\lbrace (x^{(1)}, y^{(1)}) \cdots (x^{(m)}, y^{(m)})\rbrace$
-   set $\Delta^{(l)}_{i,j} := 0$ for all (l,i,j), (hence you end up having a matrix full of zeros)
-   For training example t =1 to m:
    -   Set $a^{(1)} := x^{(t)}$
    -   Perform forward propagation to compute $a^{(l)}$ for l=2,3,…,L.(已经给定了一组假定的权重)
    -   Using $y^{(t)}$,compute $\delta^{(L)} = a^{(L)}-y^{(t)}$（输出节点和训练数据的差就是$\delta$,然后逐步往前求每一层的$\delta$）
    -   Compute $\delta^{(L-1)}, \delta^{(L-2)},\dots,\delta^{(2)}$ using $\delta^{(l)} = ((\Theta^{(l)})^T \delta^{(l+1)})\ .＊ \ a^{(l)}\ .＊ \ (1 - a^{(l)})$(其中后面.＊的是$g'(z^{(l)}) = a^{(l)}\ .*\ (1 - a^{(l)})$)
    -   ![](https://github.com/wubugui/FXXKTracer/raw/master/pic/delta%E2%80%94%E2%80%94format.gif) or with vectorization,![](https://github.com/wubugui/FXXKTracer/raw/master/pic/CodeCogsEqn.gif)
    
-   Hence we update our new $\Delta$ matrix.
    -   ![](https://github.com/wubugui/FXXKTracer/raw/master/pic/CodeCogsEqn_1.gif),if j != 0
    -   ![](https://github.com/wubugui/FXXKTracer/raw/master/pic/CodeCogsEqn_2.gif),otherwise.
-   Finally,$\frac \partial {\partial \Theta_{ij}^{(l)}} J(\Theta) = D_{i,j}^{(l)}$.

### 理解
>If we consider simple non-multiclass classification (k = 1) and disregard regularization, the cost is computed with:
$$cost(t) =y^{(t)} \ \log (h_\Theta (x^{(t)})) + (1 - y^{(t)})\ \log (1 - h_\Theta(x^{(t)}))$$
$$\delta_j^{(l)} = \dfrac{\partial}{\partial z_j^{(l)}} cost(t)$$
Recall that our derivative is the slope of a line tangent to the cost function, so the steeper the slope the more incorrect we are. Let us consider the following neural network below and see how we could calculate some$\delta_j^{(l)}$.
![](https://github.com/wubugui/FXXKTracer/raw/master/pic/qc309rdcEea4MxKdJPaTxA_324034f1a3c3a3be8e7c6cfca90d3445_fixx.png)
In the image above, to calculate $\delta_2^{(2)}$,we multiply the weights 
$\Theta_{12}^{(2)}$ and $\Theta_{22}^{(2)}$ by their respective $\delta$ values found to the right of each edge. So we get $\delta_2^{(2)}=\Theta_{22}^{(2)}\delta_1^{(3)}+\Theta_{22}^{(2)}\delta_2^{(3)}$.To calculate every single possible $\delta_j^{(l)}$ ,we could start from the right of our diagram. We can think of our edges as our $\Theta_{ij}$.Going from right to left, to calculate the value of $\delta_j^{(l)}$ ,you can just take the over all sum of each weight times the $\delta$ it is coming from. Hence, another example would be $\delta_2^{(3)} = \Theta_{12}^{(3)}* \delta_1^{(4)}$
** “error term”$\delta_j^{(l)}$ 通俗点说表示这一个节点应该为最后输出的误差负多少责。**


### 实现

>将矩阵展平为向量`thetaVec = [Theta1(:);Theta2(:);Theta3(:)];`
>从向量中提取矩阵`reshape(thetaVec(1:110),10,11);`将thetaVec的前110个元素提取为10行，11列的矩阵。
**在实现的时候，将传递展平为向量的矩阵。**


神经网络方向传播算法比较复杂，计算出来的结果可以使用下列方法来Check梯度计算是否正确：

使用当时的$\Theta$带入下式：
$$\dfrac{\partial}{\partial\Theta_j}J(\Theta) \approx \dfrac{J(\Theta_1, \dots, \Theta_j + \epsilon, \dots, \Theta_n) - J(\Theta_1, \dots, \Theta_j - \epsilon, \dots, \Theta_n)}{2\epsilon}$$

一般${\epsilon = 10^{-4}}$.
这种方法可以近似计算梯度，但是及其的缓慢，所以check完之后注意关闭。

随机初始化$\Theta$的时候，需要打破对称，并且在区间$[-\epsilon_{init},\epsilon_{init}]$均匀随机选择。对于$\epsilon_{init}$的选择，一般可以准守如下规则：
> One effective strategy for choosing $\epsilon_{init}$ is to base it on the number of units in the network. A good choice of $\epsilon_{init}$ is $\epsilon_{init}=\frac{\sqrt{6}}{\sqrt{L_{in}+L_{out}}}$ ,where $L_{in}=s_l$ and $L_{out}=s_{l+1}$ are the number of units in the layers adjacent to $\Theta^{(l)}$.

```matlab
function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
m = size(X, 1);
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
l = ones(m,1);
%加一列bais
Xn = [l,X];
Z1 = Xn';
A1 = Xn';
%已经有了bais的参数
A = Theta1*Xn';
Z2 = A;
A = sigmoid(A);
A = [l,A'];
A2 = A';
A = Theta2*A';
Z3 = A;
A3 = A;
A = sigmoid(A);
%1列一组数据

ldm =  lambda/m;
dm = ldm/lambda;
%y为1x10
%m lines
%装换y格式
p = randperm(num_labels);
p = sort(p);
yn = p(ones(m,1),:);
for idx=1:m
    yn(idx,:) = yn(idx,:)==y(idx);
end
a = diag(yn*log(A));
b = diag((1.-yn)*log(1.-A));
J= -sum(a+b)/m;

[n1,n] = size(Theta1);
k1 = 0;
for i=1:n1
    for j=2:n
        k1= k1+Theta1(i,j)*Theta1(i,j);
    end
end

[n1,n] = size(Theta2);
k2 = 0;
for i=1:n1
    for j=2:n
        k2=k2+Theta2(i,j)*Theta2(i,j);
    end
end
J=J+(k1+k2)*ldm*0.5;

%Backpropagation
for i=1:m
    delta3 = A(:,i) - yn(i,:)';
    delta2 = Theta2'*delta3.*sigmoidGradient([1;Z2(:,i)]);
    Theta1_grad = Theta1_grad + (A1(:,i)*delta2(2:end)')';
    temp = A2(:,i)*delta3';
    Theta2_grad = Theta2_grad + temp';
end

Theta1_grad = Theta1_grad./m;
Theta2_grad = Theta2_grad./m;

[~,n2]=size(Theta1);
Theta1_grad(:,2:n2) = Theta1_grad(:,2:n2) + ldm*Theta1(:,2:n2);
[~,n2]=size(Theta2);
Theta2_grad(:,2:n2) = Theta2_grad(:,2:n2) + ldm*Theta2(:,2:n2);
% Unroll
grad = [Theta1_grad(:) ; Theta2_grad(:)];
end
```
# Advice for Applying Machine Learning & Machine Learning System Design
训练学习曲线

训练不同的数量的example得到的参数用来训练一组cross validation数据，并与自身对比误差。
```matlab
function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
    m = size(X, 1);
    error_train = zeros(m, 1);
    error_val   = zeros(m, 1);
    for i=1:m
        %learning theta
        t = trainLinearReg(X(1:i,:),y(1:i,:),lambda);
        %train
        [j,~]=linearRegCostFunction(X(1:i,:), y(1:i,:), t, 0);
        error_train(i)=j;
        %cross validation
        [j,~]=linearRegCostFunction(Xval, yval, t, 0);
        error_val(i)=j;
    end
end
```
……未完待续

# 支持向量机
用一个比较直观的图来理解支持向量机：
![](https://github.com/wubugui/FXXKTracer/raw/master/pic/mm_svm.png)
首先要注意：多项式$\Theta^TX$可以理解为向量的内积。向量的内积可以写为$|\Theta^T||X|cos<\Theta^T,X> = |\Theta^T| Prj_{\Theta^T}X$.
上图中$\Theta_n$代表训练参数，$L_{\Theta_n}$代表分类直线，其中$\Theta_n$指向直线的法向，可以看到，M点和G点分别做到各个$\Theta_n$的投影，得到$P_n,P_n'$.
此时SVM要求求解：
$$min_{\Theta}\frac{1}{2}\sum_{j=1}^n \theta_j^2=min_{\Theta} \frac{1}{2}|\Theta|^2$$
s.t.
$p^{(i)}|\theta|\ge 1  $if $y^{(i)}=1$
$p^{(i)}|\theta|\le -1  $if $y^{(i)}=0$
为了简化，已经令$\theta_0 = 0$.
这里的$p^{(i)}$就是上面$|X|$在$\Theta^T$的投影.
**现在，要解决这个最优化问题，就要令$\theta$最小，但又要满足上述条件，那么一方面就要$|p^{(i)}|$非常大，那么也就意味着投影要非常小，也就是说X必须距离$L_{\Theta_n}$这条线远越好，所以，这个条件将让$X$尽量地远离分割直线，也就形成了一个边缘。。另一方面结合上图$L_{\Theta_1}$和$L_{\Theta_2}$以及对应的两个数据$G,M$来看，可以获得数据是如何被分类的直觉。**

## 核方法
之前我们有hypothesis function多项式：
$$h_\theta(X) = X \theta = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_3 + \cdots + \theta_n x_n$$
后来可以将这些项变为：
quadratic function：$h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_1^2$
cubic function：$h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_1^2 + \theta_3 x_1^3$
square root function:$h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 \sqrt{x_1}$
现在这些可以写为：
$$h_\theta(f(X)) = \theta_0 + \theta_1 f(x_1) + \theta_2 f(x_2) + \theta_3 f(x_3) + \cdots + \theta_n f(x_n)$$
这样的写法获得更多灵活性，$f$被称为核(Kernel).
比如为了度量**相似度**可以使用高斯核：
$$f_1 = \exp(-\frac{\|x_1-l^{(1)}\|}{2\sigma^2})$$

所以优化函数变为：
$$min_{\theta}C\sum_{i=1}^m y^{(i)}cost_1(\Theta^Tf^{(i)})+(1-y^{(i)})cost_0(\Theta^Tf^{(i)})+\frac{1}{2}\sum_{j=1}^m\theta_j^2$$

## 参数选择

对于参数$C=\frac{1}{\lambda}$，大C低偏高方差，小C反之。
参数$\sigma^2$,大，$f_i$平滑，高偏，低方差。
大量example小feature用高斯函数，小数据直接线性，高斯容易overfit.
另外依然要注意数据feature的归一化。

**参数选择目标：选择参数和核函数，以便在cross-validation数据上执行得最好。**

## 选择

n=number of features,m=number of training examples

-   If n is large (relative to m):
    使用逻辑回归或者线性核SVM

-   if n is small ,m is intermediate:
    使用高斯核SVM

-   if n is small,m is large:
    创建/添加更多features，然后使用逻辑回归或者线性核SVM

-   神经网络可能对所有上述设定都工作的良好，但是可能会更慢。



# Ref
-  [https://www.coursera.org/learn/machine-learning](https://www.coursera.org/learn/machine-learning)
-  [http://www.52ml.net/20615.html](http://www.52ml.net/20615.html)
-  [http://blog.csdn.net/huruzun/article/details/41457433](http://blog.csdn.net/huruzun/article/details/41457433)
-   [深入理解机器学习：从原理到算法](https://www.amazon.cn/%E5%9B%BE%E4%B9%A6/dp/B01IN688PQ/ref=sr_1_1?ie=UTF8&qid=1488351471&sr=8-1&keywords=%E6%B7%B1%E5%85%A5%E7%90%86%E8%A7%A3%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%EF%BC%9A%E4%BB%8E%E5%8E%9F%E7%90%86%E5%88%B0%E7%AE%97%E6%B3%95)