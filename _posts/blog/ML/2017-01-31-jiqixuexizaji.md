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

# Principal Component Analysis

>摘自：[http://blog.codinglabs.org/articles/pca-tutorial.html](http://blog.codinglabs.org/articles/pca-tutorial.html)

## 原理

常用数据分析方法，通过线性变换将原始数据变换为一组各维度线性无关的表示，可用于提取数据的主要特征分量，常用于高维数据的降维。

一般的，如果我们有M个N维向量，想将其变换为由R个N维向量表示的新空间中，那么首先将R个基按行组成矩阵A，然后将向量按列组成矩阵B，那么两矩阵的乘积AB就是变换结果，其中AB的第m列为A中第m列变换后的结果。

$$\begin{pmatrix}  p_1 \newline  p_2 \newline \vdots \newline  p_R
\end{pmatrix} \begin{pmatrix}  a_1 & a_2 & \cdots & a_M \end{pmatrix}=\begin{pmatrix}  p_1a_1 & p_1a_2 & \cdots & p_1a_M \newline
 p_2a_1 & p_2a_2 & \cdots & p_2a_M \newline
  \vdots  & \vdots  & \ddots & \vdots
 \newline  p_Ra_1 & p_Ra_2 & \cdots & p_Ra_M
\end{pmatrix}$$

其中$p_i$是一个行向量，表示第$i$个基，$a_j$是一个列向量，表示第$j$个原始数据记录。

特别要注意的是，这里R可以小于N，而R决定了变换后数据的维数。也就是说，我们可以将一N维数据变换到更低维度的空间中去，变换后的维度取决于基的数量。因此这种矩阵相乘的表示也可以表示降维变换。

降维问题的优化目标：将一组N维向量降为K维（K大于0，小于N），其目标是选择K个单位（模为1）正交基，使得原始数据变换到这组基上后，各字段两两间协方差为0，而字段的方差则尽可能大（在正交的约束下，取最大的K个方差）。

假设我们只有a和b两个字段，那么我们将它们按行组成矩阵X：

$$X=\begin{pmatrix}
  a_1 & a_2 & \cdots & a_m \newline
  b_1 & b_2 & \cdots & b_m
\end{pmatrix}$$

然后我们用X乘以X的转置，并乘上系数1/m：

$$\frac{1}{m}XX^\mathsf{T}=\begin{pmatrix}
  \frac{1}{m}\sum_{i=1}^m{a_i^2}   & \frac{1}{m}\sum_{i=1}^m{a_ib_i} \newline
  \frac{1}{m}\sum_{i=1}^m{a_ib_i} & \frac{1}{m}\sum_{i=1}^m{b_i^2}
\end{pmatrix}$$

这个矩阵对角线上的两个元素分别是两个字段的方差，而其它元素是a和b的协方差。两者被统一到了一个矩阵的。

根据矩阵相乘的运算法则，这个结论很容易被推广到一般情况：

设我们有m个n维数据记录，将其按列排成n乘m的矩阵X，设$C=\frac{1}{m}XX^\mathsf{T}$,，则C是一个对称矩阵，其对角线分别个各个字段的方差，而第i行j列和j行i列元素相同，表示i和j两个字段的协方差。

设原始数据矩阵X对应的协方差矩阵为C，而P是一组基按行组成的矩阵，设Y=PX，则Y为X对P做基变换后的数据。设Y的协方差矩阵为D，我们推导一下D与C的关系：

$$\begin{array}{l l l}
  D & = & \frac{1}{m}YY^\mathsf{T} \newline
    & = & \frac{1}{m}(PX)(PX)^\mathsf{T} \newline
    & = & \frac{1}{m}PXX^\mathsf{T}P^\mathsf{T} \newline
    & = & P(\frac{1}{m}XX^\mathsf{T})P^\mathsf{T} \newline
    & = & PCP^\mathsf{T}
\end{array}$$

我们要找的P不是别的，而是能让原始协方差矩阵对角化的P。换句话说，优化目标变成了寻找一个矩阵P，满足$PCP^T$是一个对角矩阵，并且对角元素按从大到小依次排列，那么P的前K行就是要寻找的基，用P的前K行组成的矩阵乘以X就使得X从N维降到了K维并满足上述优化条件。

协方差矩阵C是一个是对称矩阵，实对称矩阵有一系列非常好的性质：
-  实对称矩阵不同特征值对应的特征向量必然正交。
-  设特征向量λ重数为r，则必然存在r个线性无关的特征向量对应于λ，因此可以将这r个特征向量单位正交化。

由上面两条可知，一个n行n列的实对称矩阵一定可以找到n个单位正交特征向量，设这n个特征向量为$e_1,e_2,\cdots,e_n$:

$$E=\begin{pmatrix}
  e_1 & e_2 & \cdots & e_n
\end{pmatrix}$$

则对协方差矩阵C有如下结论：

$$E^\mathsf{T}CE=\Lambda=\begin{pmatrix}
  \lambda_1 &             &         & \newline
              & \lambda_2 &         & \newline
              &             & \ddots & \newline
              &             &         & \lambda_n
\end{pmatrix}$$

P是协方差矩阵的特征向量单位化后按行排列出的矩阵，其中每一行都是C的一个特征向量。如果设P按照ΛΛ中特征值的从大到小，将特征向量从上到下排列，则用P的前K行组成的矩阵乘以原始数据矩阵X，就得到了我们需要的降维后的数据矩阵Y。

## 算法及实例

### PAC算法

设有m条n维数据。
-  将原始数据按列组成n行m列矩阵X
-  将X的每一行（代表一个属性字段）进行零均值化，即减去这一行的均值
-  求出协方差矩阵$C=\frac{1}{m}XX^\mathsf{T}$
-  求出协方差矩阵的特征值及对应的特征向量
-  将特征向量按对应特征值大小从上到下按行排列成矩阵，取前k行组成矩阵P
-  $Y=PX$即为降维到k维后的数据

### 实例

$$\begin{pmatrix}
  -1 & -1 & 0 & 2 & 0 \newline
  -2 & 0 & 0 & 1 & 1
\end{pmatrix}$$

为例，我们用PCA方法将这组二维数据其降到一维。

因为这个矩阵的每行已经是零均值，这里我们直接求协方差矩阵：

$$C=\frac{1}{5}\begin{pmatrix}
  -1 & -1 & 0 & 2 & 0 \newline
  -2 & 0 & 0 & 1 & 1
\end{pmatrix}\begin{pmatrix}
  -1 & -2 \newline
  -1 & 0  \newline
  0  & 0  \newline
  2  & 1  \newline
  0  & 1
\end{pmatrix}=\begin{pmatrix}
  \frac{6}{5} & \frac{4}{5} \newline
  \frac{4}{5} & \frac{6}{5}
\end{pmatrix}$$

然后求其特征值和特征向量，具体求解方法不再详述，可以参考相关资料。求解后特征值为：

$\lambda_1=2,\lambda_2=2/5$

其对应的特征向量分别是：

$$c_1\begin{pmatrix}
  1 \newline
  1
\end{pmatrix},c_2\begin{pmatrix}
  -1 \newline
  1
\end{pmatrix}$$

其中对应的特征向量分别是一个通解，$c_1$和$c_2$可取任意实数。那么标准化后的特征向量为：

$$\begin{pmatrix}
  1/\sqrt{2} \newline
  1/\sqrt{2}
\end{pmatrix},\begin{pmatrix}
  -1/\sqrt{2} \newline
  1/\sqrt{2}
\end{pmatrix}$$

因此我们的矩阵P是：

$$P=\begin{pmatrix}
  1/\sqrt{2}  & 1/\sqrt{2}  \newline
  -1/\sqrt{2} & 1/\sqrt{2}
\end{pmatrix}$$

可以验证协方差矩阵C的对角化：

$$PCP^\mathsf{T}=\begin{pmatrix}
  1/\sqrt{2}  & 1/\sqrt{2}  \newline
  -1/\sqrt{2} & 1/\sqrt{2}
\end{pmatrix}\begin{pmatrix}
  6/5 & 4/5 \newline
  4/5 & 6/5
\end{pmatrix}\begin{pmatrix}
  1/\sqrt{2} & -1/\sqrt{2}  \newline
  1/\sqrt{2} & 1/\sqrt{2}
\end{pmatrix}=\begin{pmatrix}
  2 & 0  \newline
  0 & 2/5
\end{pmatrix}$$

最后我们用P的第一行乘以数据矩阵，就得到了降维后的表示：

$$Y=\begin{pmatrix}
  1/\sqrt{2} & 1/\sqrt{2}
\end{pmatrix}\begin{pmatrix}
  -1 & -1 & 0 & 2 & 0 \newline
  -2 & 0 & 0 & 1 & 1
\end{pmatrix}=\begin{pmatrix}
  -3/\sqrt{2} & -1/\sqrt{2} & 0 & 3/\sqrt{2} & -1/\sqrt{2}
\end{pmatrix}$$

降维投影结果如下图：

![](http://blog.codinglabs.org/uploads/pictures/pca-tutorial/07.png)

# 讨论

PCA也存在一些限制，例如它可以很好的解除线性相关，但是对于高阶相关性就没有办法了，对于存在高阶相关性的数据，可以考虑Kernel PCA，通过Kernel函数将非线性相关转为线性相关。另外，PCA假设数据各主特征是分布在正交方向上，如果在非正交方向上存在几个方差较大的方向，PCA的效果就大打折扣了。
PCA是一种无参数技术，也就是说面对同样的数据，如果不考虑清洗，谁来做结果都一样，没有主观参数的介入，所以PCA便于通用实现，但是本身无法个性化的优化。