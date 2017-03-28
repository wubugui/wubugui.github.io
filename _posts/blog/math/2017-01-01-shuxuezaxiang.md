---
layout: post
title: "数学杂项"
modified:
categories: blog
excerpt:
tags: [数学]
image:
  feature:
date: 2015-08-10T08:08:50-04:00
---

#   线性插值

[线性插值](https://en.wikipedia.org/wiki/Linear_interpolation)
[双线性插值](https://en.wikipedia.org/wiki/Bilinear_interpolation)
[三线性插值](https://en.wikipedia.org/wiki/Trilinear_interpolation)
[多线性插值](https://en.wikipedia.org/wiki/Multivariate_interpolation)

# Von Mises–Fisher分布

在directional statistic中，这是一个在p维空间中的p-1维球上的概率分布。如果p=2，这就是一个在circle上的[von Mises distribution](https://en.wikipedia.org/wiki/Von_Mises_distribution)。
>一个平面,一个球面,或不管什么面,都是二维空间,因为对于面上的任意一点,只要用两个数就可以描述.同理,线(直线或曲线)是一维的,因为只需一个数便可以描述线上各点的位置.

概率密度函数如下：
$$f_p(x;\mu,\kappa)=C_p(\kappa)e^{\kappa \mu^T x}$$
其中$\kappa \ge 0,||\mu||=1$,归一化常数$C_p(\kappa)$等于：
$$C_p(\kappa)=\frac{\kappa^{p/2-1}}{(2\pi)^{p/2}I_{p/2-1}(\kappa)}$$
$I_v$表示一个Bessel function，如果p=3,公式变为下面这样：
$$C_3(\kappa)=\frac{\kappa}{4\pi sinh\kappa}=\frac{\kappa}{2\pi(e^\kappa-e^{-\kappa})}$$
参数$\mu$和$\kappa$分别叫做平均方向和聚集系数。**其中后者越大在前者的方向上聚集的程度就越高。**The distribution is unimodal(单峰值) for $\kappa>0$, and is uniform on the sphere for $\kappa=0$.
如果p=3，也叫作Fisher distribution。我们要使用的就是p=3的情形，这时候是在三维空间中的二维sphere上的二维分布。类似下图：

![]({{ site.url }}/images/10-59-17.jpg)

Points sampled from three von Mises–Fisher distributions on the sphere (blue: $\kappa=1$, green: $\kappa=10$, red: $\kappa=100$). The mean directions $\mu$ are shown with arrows.


**Ref**
-   [Wiki](https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution)
-   [(应用)Frequency Domain Normal Map Filtering](http://www.cs.columbia.edu/cg/normalmap/)