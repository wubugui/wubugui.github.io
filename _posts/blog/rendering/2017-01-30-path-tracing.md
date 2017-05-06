---
layout: post
title: "Path Tracing"
modified:
categories: blog
excerpt:
tags: [渲染,LTE,光线传输,GI]
image:
feature:
date: 2017-01-30T08:08:50-04:00
---

# 渲染方程的面积形式

渲染方程的面积形式：

$$L(p' \rightarrow p)=L_e(p' \rightarrow p)+\int_A f(p'' \rightarrow p' \rightarrow p)L(p'' \rightarrow p')G(p'' \leftrightarrow p')dA(p'')$$

其中：

$$G(x,y)=V(x,y)\frac{cos(N_x,\omega_i)cos(N_y,-\omega_i)}{r^2_xy}$$

A代表整个场景的所有表面。
稍后我们将使用ray trace来解算V(x,y).

>Recall solid angle form:
$$L_(p,\omega_o) = L_e(p,\omega_o) + \int_{\Omega} f(p.\omega_o,\omega_i)L_i(p,\omega_i)|cos\theta_i|d\omega_i$$

面积形式与原来等价，**但是却带来了不同的视角，以及解决方法**。对于面积形式，我们将会根据表面分布在表面上采样表面点。

对于面积形式，我们首先要明白一个事实，一个场景中的光照能量全部来自于表面的emit，并没有所谓的抽象光源。我们在tracing的时候，把整个场景看成一个大的surface，然后根据一些奇怪的依据在上面采样，采到哪个表面点，如果他emit了能量就是光源。然而，随机乱采样显然是不行的，比如你使用path tracing采样1000个点，1000个点击中的表面点都不emit能量，那整个图都是黑色的，方差高的吓人（这也解释了为什么path tracing对于某些小光源场景支持效果很差，因为你的path vertex难以击中光源）。所以，对于采样点的选择需要根据光源分布和brdf来选择。

接下来推到一个LTE的**N点形式**，上面的面积形式实际上是个**三点形式**。

设原方程的积分的积分为一个线性算子$\Gamma$.

$L_{i \rightarrow i-1}$代表面积元$i$到面积元$i-1$的Radiant.

其中:

$\Gamma L_{i\rightarrow i-1} = \int_A f(i \rightarrow i-1 \rightarrow i-2)L(i\rightarrow i-1)G(i \leftrightarrow i-1)dA(i)$

现在把原公式递归展开：

$$L(1 \rightarrow 0) = L_e(1\rightarrow 0) + \Gamma \Gamma \Gamma ... \Gamma L_{i \rightarrow i-1}$$

其中因为$\Gamma L(i \rightarrow i-1) = \Gamma (L_e(i \rightarrow i-1) + \Gamma L(i-1 \rightarrow i-2))$且$\Gamma$本身是线性算子，故：

$$\Gamma L(i \rightarrow i-1) = L_e(i \rightarrow i-1) + \Gamma \Gamma L(i-1 
\rightarrow i-2)$$

所以整个方程照此展开的最终形式如下：

$$L(1 \rightarrow 0) = L_e(1\rightarrow 0) + \Gamma 
(L_e(2 \rightarrow 1) +  \Gamma 
(L_e(3 \rightarrow 2) + \Gamma (
... \Gamma(
 L_e(i \rightarrow i-1) + \Gamma L(i-1 \rightarrow i-2))))$$

因为每一层积分其实都是单变量积分，所以外层相关的变量都可以看为常数，所以可以把外层的函数提到内层积分里面去。因为内层里面有个独立的$L_e$项不递归，所以可以把这一项拿出来作为整个积分式子单独的一项（越里面的$L_e$就有越多的从外层提进去的函数），**需要注意，因为所有的项都是外面提进来的，所以$L_e$极其本来的f、G积分项本身的下标要比提进来的项的下标大，且下标依次往下降1知道降到2和1为止**，剩下的项要继续递归，直到变为一个余项。
假设我们看$L_{2,1}$和$L_{3,2}$两个连续项，则有：

$$L_{2,1} = L_{e2,1} + \int_A f_{3,2,1}L_{3,2}G_{3,2}dA_3$$

$$L_{3,2} = L_{e3,2} + \int_A f_{4,3,2}L_{4,3}G_{4,3}dA_4$$

把后者代入前者有：

$$L_{2,1} = L_{e2,1}+\int_A f_{3,2,1} L_{e3,2} G_{3,2} dA_3 + \int_A \int_A f_{3,2,1}f_{4,3,2}G_{3,2}G_{4,3}L_{4,3} dA_3 dA_4$$

如此一来最终这个式子就化为了：

$$L_{1,0} = L_{e1,0}+\int_A f_{2,1,0} L_{e2,1} G_{2,1} dA_2 + $$

$$\int_A \int_A L_{e3,2}f_{3,2,1}G_{3,2}f_{2,1,0}G_{2,1}dA_3dA_2 +$$

$$\int_A \int_A \int_A L_{e4,3}f_{4,3,2}G_{4,3}f_{3,2,1}G_{3,2}f_{2,1,0}G_{2,1}dA_4dA_3dA_2 + $$

$$... +$$

$$\int_A \int_A \int_A ... f_{2,1,0}f_{3,2,1}f_{4,3,2}...G_{2,1}G_{3,2}G_{4,3}... dA_2dA_3 dA_4...$$

最后这一项和前面说的一样也是可以把这些在一起的项提出去的，所以相当于有无穷个没有L的项相乘，因为能量守恒单独积分$\int_A fG dA < 1$，所以余项收敛于0.
那么公式最后就变成了：

$$L_{1,0} = L_{e1,0}+\int_A f_{2,1,0} L_{e2,1} G_{2,1} dA_2 + $$

$$\int_A \int_A L_{e3,2}f_{3,2,1}G_{3,2}f_{2,1,0}G_{2,1}dA_3dA_2 +$$

$$\int_A \int_A \int_A L_{e4,3}f_{4,3,2}G_{4,3}f_{3,2,1}G_{3,2}f_{2,1,0}G_{2,1}dA_4dA_3dA_2 + ...$$

这个公式就是LTE的**N点形式**，也就是路径积分形式.简记为：

$$L(p1 \rightarrow p_0) = \sum_{n=1}^{\infty}P(\bar{p_n})$$

其中：$\bar{p_n}=p_0,p_1,...,p_n$

**其中每一项就是一条长度为n的Path。**

粗暴无脑地说，要解决LTE，**就是要把这个N点形式解出来。**

我们观察一下这个方程可以发现，解决他是可行的：

-   首先，每一项都足够简单，没有积分镶嵌积分的情形

-   其次，每一项与相邻的项都递增

-   多维积分可以使用蒙特卡洛方法来计算

-   无穷项可以用Russian roulette来解决，其中每一项都采用RR
    $$\frac{1}{1-q_1}(P(\bar{p_1})+\frac{1}{1-q_2}(P(\bar{p_2})+\frac{1}{1-q_3}(P(\bar{p_3}))))$$


# 实现

注意要解决每一项，要做的事情简单地说，就是在场景中采样n个顶点，然后从某一点开始计算链接这n个顶点的所有路径所携带的能量。

完全可以任意地采样这n个点，不过这样做可能引入相当高的方差，计算某些面积measure的PDF也并不方便。

## 递增构造path
所以可以从相机出发，递增地构造路径，这样可以消除G（相当于对G的重要性采样）；另外，可以使用rr来终结无限长的路径。对每一次相交对于贡献底的path都使用rr来决定是否要终结路径，如果终结就终结，不终结就继续，同时除以不终结概率，这个概率一般与当前的path贡献相关。

另外一个实现上的技巧是，观察原式，可以发现：$fG$是递增的，也就是说，前一term有的因子，后面也有，如果可以把先前的因子计算了保留下来，并强行假设这些保留的因子是随机，不想干的，那么就需要每个term都算一次了，可以一次性从头到尾都算完，代价只是增大一点点方差而已。

值得一说的是，最终的方案将PT的计算简化为主要是一个throughput的计算:

$$\frac{L_e(p_i->p_{i-1})f(p_i->p_{i-1}->p_{i-2})G(p_i->p_{i-1})}{p_A(p_i)}(
\prod_{j=1}^{i-2}\frac{f(p_{j+1}->p_j->p_{j-1})|\cos\theta_j|}{p_{\omega}(p_{j+1}-p_j)})$$

注意后面连乘项的分母，因为上下的$G$都被约掉了，所以只剩下一个solid angle measure的pdf，这为实现带来了方便（BDPT中因为要对path使用MIS要求每个顶点都记录面积measure的pdf，这要求知道前后两个顶点来计算面积measure的pdf，实现比PT更复杂一些）.

>这里为了把事情说得透彻以免以后进一步学习出现疑惑，必须要注意以下几个事实：
>
>1.  pt中的path是采样path，原则上可以在场景中任意选择点，连接成path。
>
>2.  **使用面积形式计算的时候，采样每一条path所使用的pdf分布必须都是面积measure的，也就是说，在某个点去采样一个半球来作为一个pdf是错误的。**
>
>3.  注意pdf从solid angle measure转换到面积measure时，实际上说的是，**顶点i**的solid采样pdf转换成了**顶点i+1**的面积采样pdf，也就是：
>
>$$p_A(p_{i+1})=p_w(p_{j+1}\leftarrow p_j)\frac{|cos\theta|}{||p_j-p_{j+1}||^2}$$
>
>-  于是throughput为化简前的形式应该是(注意顶点下标):
>
>   $$\frac{L_e(p_i\rightarrow p_{i-1})f(p_i \rightarrow p_{i-1}\rightarrow p_{i-2})G(p_i\rightarrow p_{i-1})}{p_A(p_i)}(
\prod_{j=1}^{i-2}\frac{f(p_{j+1}\rightarrow p_j\rightarrow p_{j-1})G(p_{j+1},p_j)}{p_A(p_{j+1})})$$
>
>-  其中采样点$p_1$是相机发射的第一根ray与场景的交点，位置确定不需要采样，所以从$p_2$开始采样，采样概率密度为$p_A(p_{j+1})$
>
>-  这一点也是面积形式(eg.pt积分器)和solid角形式(eg.直接光积分器)的显著差别，实现形式上看起来差不多，可能容易想当然把前后混淆，但这一点区别在双向方法中将会有更清晰。

## Next-Event Estimation

由于pt最终如果无法击中光源，那么整条path贡献都是0.于是将直接光与间接光分开，每次击中**漫反射**表面都计算一次直接光（随机采样所有光源中的一个的采样点），然后在最后一次Le的时候仅仅只计算**非漫反射**表面的Le即可。

## 最终结果

最后使用上述方案，可以渲染出下列图像：

![](https://github.com/wubugui/FXXKTracer/raw/master/pic/mat1.png)

# 补充

## 完整PT积分

实际上蒙特卡洛pt的解算目标是一个维度相当高的积分：

-   像素(x,y)的反走样积分
-   镜头(u,v)的景深积分
-   时间(t)的运动模糊积分
-   波长($\lambda$)的色散积分
-   这些维度随着递归深度提升指数depth


最终的维度为：

$$d = (x,y;u,v;t;\lambda)^depth$$


