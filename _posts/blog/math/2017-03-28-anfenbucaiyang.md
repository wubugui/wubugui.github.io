---
layout: post
title: "按分布采样"
modified:
categories: blog
excerpt:
tags: [数学,随机采样]
image:
  feature:
date: 2015-08-10T08:08:50-04:00
---

蒙特卡洛方法最重要的就是按照某一分布采样。

采样某一分布，主要有两种方法：
-   [拒绝法](http://claireswallet.farbox.com/post/math/gai-lu-lun-yu-shu-li-tong-ji-ji-chu-zhai-yao#toc_22)
-   [逆分布法](http://claireswallet.farbox.com/post/math/gai-lu-lun-yu-shu-li-tong-ji-ji-chu-zhai-yao#toc_21)


# 检验采样结果
为了在实现采样后检验采样结果是否真的满足分布，采用下列方法:
-   绘制**归一化**后的分布函数
-   搜集若干采样
-   统计采样数据绘制**归一化**的分布图（注意：在统计的时候，出来的统计数据的值代表——这个数据在x轴的哪个地方，在某个x0的领域内，统计出所有落在领域的数据量，除以总数据就是频率）

统计一组数据并且画出分布的代码大致如下（所有传入的采样必须在[0,1]）：
```cpp
//这里将采集的数据分成140个区间进行统计
void check(std::vector<float> v,NormalView* app,ofColor c=ofColor::black,float scale = 0.2)
{
	const int acc = 140;
	float a[acc]={0};
	for(int i=0;i<v.size();++i)
	{
		float t =0;
		int idx = 0;
		for(int j=0;j<acc;++j)
		{
			if(v[i]>=t&&v[i]<t+1.f/(float)acc) 
			{
				a[idx]++;
				break;
			}
			t +=1.f/acc;
			++idx;
		}
	}
	//定义域在[0,1],将
	//原来的频率图积分应该将dx设置为1，于是有sum f(x)dx = 1
	//现在，定义域被缩放到长度为1，所以为了保证积分为1，应该把dx修改为对应的大小，于是
	//sum c f(x) dx = 1
	//dx = 1/acc & sum f(x)=1 -> c = acc
	float intval = 1.f/acc;
	for(int i=0;i<acc;++i)
		a[i] = (float)a[i]/v.size()*acc;
	for(int i=0;i<acc;++i)
	{
		int idx = (float)i/acc*app->grid_panel->get_res_w();
		app->grid_panel->set_px(idx,a[i]*app->grid_panel->get_res_h()*scale+10,c);
	}
}
```

同样的，也有2维分布的check:
```cpp
//input normalized
void check2d(std::vector<ofVec2f>& v,GridCanvasPanel3D* grid_3d)
{
	if(v.empty()) return;
	int x_res = 50;
	int y_res = 50;
	float intervalx = 1.f/x_res;
	float intervaly = 1.f/y_res;
	float* a = new float[x_res*y_res];
	std::memset(a,0,sizeof(int)*x_res*y_res);
	for(auto i:v)
	{
		for(int j=0;j<x_res;++j)
		{
			if(i.x>j*intervalx&&i.x<(j+1)*intervalx)
			{
				for(int k=0;k<y_res;++k)
				{
					if(i.y>k*intervaly&&i.y<(k+1)*intervaly)
					{
						a[j+k*x_res]=a[j+k*x_res]+1.f;
						goto label;
					}
				}
			}
		}
		label:;
	}
	int rew = grid_3d->get_res_w();
	int reh = grid_3d->get_res_h();
	for(int i=0;i<y_res;++i)
	{
		for(int j=0;j<x_res;++j)
		{
			if(abs(a[j+i*x_res])<0.01)
				continue;
			float val =  ((float)a[j+i*x_res])/v.size()*x_res*y_res;
			float yp = ((float)i)/y_res;
			float xp = ((float)j)/x_res;
			grid_3d->set_px( xp*rew,yp*reh,val,ofColor::green,0.2);
		}
	}
}

```


# 分布间转换


两个分布之间的采样要相互转化，可设他们之间的关系：
设y(满足分布$p(y)$)和x(满足分布g(x))之间有关系
$y=f^{-1}(x)=t(x)$x,y均在[0,1],且函数单调，必须要求两个分布下的采样**等概率**，设在两个pdf定义域上有两个极小的区间$|\Delta x|/\|\Delta y|$，于是通过对函数t求导同时注意建立等概率关系不能为负值，就有$|\Delta x| |t'(x)|\approx|\Delta y|$，即有：
$$p(x)|\Delta x| \approx g(y)|\Delta y|$$
$$\frac{p(x)}{g(y)}\approx \frac{|\Delta y|}{|\Delta x|}$$
当$|\Delta x| \rightarrow 0$时：
$$\frac{p(x)}{g(y)}= |t'(x)|=|\frac{dy}{dx}|$$

为了求得函数t，现在有微分方程$p(x)=g(y)|\frac{dy}{dx}|$
为了解开绝对值，分类讨论：
-	当$\frac{dy}{dx} \le 0$时
	$$-p(x)dx=g(y)dy$$
	两端对x做变上限定积分
	$$-\int_0^x p(x)dx = \int_0^x g(y)dy \rightarrow$$
	根据牛顿莱布尼兹公式算定积分有：
	$$-P(x)|_0^x = G(y)|_0^x = G(t(x))-G(t(0))$$
	由于函数$y=t(x)$是单调递减的，又x、y 都在[0,1]，于是$t(0)=1$，所以：
	$$P(x)=1-G(y)$$
-	当大于0是，类似地，有$$P(x)=G(y)$$


另外，注意有另外一种更一般的方法：
$$P(y)=Pr{Y<=y}=Pr{t(X)<=y}$$
当t单调递增时，上式等于：
$$Pr{X<=f(y)}=G(f(y)) \rightarrow y = P^{-1}[G(x)]$$
单调递减时，上式为：
$$Pr{X>=f(y)}=1-G(f(y)) \rightarrow y = P^{-1}[1-G(x)]$$


>可参考：http://math.arizona.edu/~jwatkins/f-transform.pdf

对于多维分布，有$\vec{y}=T(\vec{x})=(T_1(\vec{x}),...,T_n(\vec{x}))$,$|\frac{dy}{dx}|$取雅各比矩阵：
$$\begin{bmatrix} \frac{\partial T_1}{\partial x_1} & \cdots & \frac{\partial T_1}{\partial x_n}\newline \vdots & \ddots & \vdots \newline \frac{\partial T_n}{\partial x_1} & \cdots & \frac{\partial T_n}{\partial x_n}\end{bmatrix}$$

## 例子1：极坐标与笛卡尔坐标

现在要求从极坐标上采样某个分布$p(r,\theta)$，然后转换成对应的笛卡尔坐标分布$p(x,y)$.

首先是变换关系：
$$
\begin{matrix}
x=r\cos(\theta)
\newline
y=r\sin(\theta)
\end{matrix}
$$

根据之前的推导，不妨设有：
$$p(r,\theta) D(r,\theta)=p(x,y) D(x,y) \rightarrow p(r,\theta) = p(x,y) \frac{D(x,y)}{D(r,\theta)} \rightarrow p(r,\theta) = p(x,y) |J_T| $$
其中:
$$J_T=\begin{bmatrix}  \cos(\theta) & -r\sin(\theta) \newline \sin(\theta) & r\cos(\theta)\end{bmatrix}$$
$$|J_T|=r(\cos^2(\theta)+\sin^2(\theta))=r$$
因此：
$$p(r,\theta)=rp(x,y)$$

## 例子2：球坐标与笛卡尔坐标

现在要求从极坐标上采样某个分布$p(r,\theta,\phi)$，然后转换成对应的笛卡尔坐标分布$p(x,y,z)$.

变换关系：
$$
\begin{matrix}
x=r\sin(\theta)\cos(\phi)
\newline
y=r\sin(\theta)\sin(\phi)
\newline
z=r\cos(\theta)
\end{matrix}
$$

这个雅克比矩阵写起来很麻烦，直接得到结果:$J_T=r^2\sin(\theta)|$

因此对应的转换关系为：
$$p(r,\theta,\phi)=r^2\sin(\theta)p(x,y,z)$$

## 例子3：球坐标到solid角

由例2知道：$p(r,\theta,\phi)=r^2\sin(\theta)p(x,y,z)$。
设有$(x,y,z)$表示**单位球**上的solid角$\vec{\omega}$，由此$r=1$.
由图可知：
![](https://github.com/wubugui/FXXKTracer/raw/master/pic/sp.png)

$$d\omega = \sin(\theta)d\theta d\phi$$

于是由概率相等有：
$$p(\omega)d\omega = p(\theta,\phi) d\theta d\phi \rightarrow p(\theta,\phi)=\sin(\theta)p(\omega)$$


# 采样实例

## 均匀采样单位半球

均匀采样半球意味着在半球上均匀地选择solid角，于是$p(\omega)=c$，同时满足归一化条件：
$$\int_{H^2}p(\omega)=1 \rightarrow c = \frac{1}{2\pi}$$

根据例3的结论有：
$$p(\theta,\phi)=\frac{\sin(\theta)}{2\pi}$$

考虑首先采样$\theta$，先求边缘分布强度函数：
$$p(\theta)=\int_0^{2\pi}\frac{\sin(\theta)}{2\pi}d\phi=\sin(\theta)$$
为了采样$\phi$计算条件分布密度函数：
$$p(\phi|\theta)=\frac{p(\theta,\phi)}{p(\theta)}=\frac{1}{2\pi}$$

然后可以使用逆分布法进行采样了，相应的CDF为：


$$
\begin{matrix}
P(\theta)=\int_a^{\theta}\sin(\theta ')d\theta ' = 1 - \cos(\theta)
\newline
P(\phi|\theta)=\int_0^{\phi}\frac{1}{2\pi}d\phi '=\frac{\phi}{2\pi}
\end{matrix}
$$

于是：
$$
\begin{matrix}
\theta=\cos^{-1}\xi_1
\newline
\phi=2\pi\xi_2
\end{matrix}
$$
换回xyz就得到：
$$
\begin{matrix}
x=\sin\theta\cos\phi=\cos(2\pi\xi_2)\sqrt{1-\xi^2_1}
\newline
y=\sin\theta\sin\phi=\sin(2\pi\xi_2)\sqrt{1-\xi_1^2}
\newline
z=\cos\theta=\xi_1
\end{matrix}
$$

>类似地，采样全球有：
$$
\begin{matrix}
x=\cos(2\pi\xi_2)\sqrt{1-z^2}
\newline
y=\sin(2\pi\xi_2)\sqrt{1-z^2}
\newline
z=1-2\xi_1
\end{matrix}
$$

## 均匀采样单位圆盘

不正确但是很直觉的方案：
$$r=\xi_1,\theta=2\pi\xi_2$$

这个方案并不均匀，采样点将会聚集在圆心附近。

首先求pdf：
$$\int_\Omega p(x,y)dxdy = 1 \rightarrow p(x,y)=\frac{1}{\pi}$$
于是：
$$p(r,\theta)=\frac{r}{\pi}$$

$$
\begin{matrix}
p(r) = \int_0^{2\pi}p(r,\theta)d\theta=2r \rightarrow P(r)=r^2
\newline
p(\theta|r)=\frac{p(r,\theta)}{p(r)}=\frac{1}{2\pi} \rightarrow P(\theta)=\frac{\theta}{2\pi}
\end{matrix}
$$

所以结果其实是：
$$r=\sqrt{\xi_1},\theta=2\pi\xi_2$$

## 均匀采样三角形

在均匀采样三角形之前，必须了解这样一个事实：
有很多随机变量都具有**放射不变性**，也就是说**他们的分布在做了仿射变换($Y = aX+b$)之后依然保持基本的分布（pdf）不变只是改变分布的参数**。而均匀分布就具有这一性质，其他的还有正态分布等等。
**这里按照前面说的变换分布可以推导出来。**

>这里要另外提一种叫做poison disc的分布，这种分布也是所谓“均匀”的，这里看一下，以免和均匀分布搞混。
Cook提出了使用Poisson Disk分布的Pattern会非常适合二维的采样，他指出人眼中的感光细胞也成Poisson Disk分布。所谓Poisson Disk Pattern即所有的采样点到其余所有的采样点的距离都大于一个阀值，可以认为未抖动的网格是Poisson Disk Pattern的其中一个特例。
这样Poisson Disk就要求任意两点之间的距离不小于阀值，比如10x10的区域内生成100个(以上)的采样点，阀值可以采用0.9（大于等于1将有可能使有的采样点不能放到区域内）。
传统生成Poisson Disk序列的方法为Dart-Throwing。可以把这个原始算法看作买六合彩。它不断“Throw”一个随机的采样点，然后和已有的采样点集合比较距离，若遇到小于阀值的就Discard，再重新“Throw”一个新的随机采样点；如果符合条件则“Dart”中了，添加到采样点集合里。就这样不断循环直到完全填满区域，或者生成的采样点“足够多”为止。如果采样区域非常大、或者采样点数目巨大，那么要计算完所有采样点的几率真的比中六合彩还要低得多。
可是Poisson Disk分布的确太好，太适合各种图像重构的采样了，所以很多人会预计算一个足够大的Pattern，再把它Tile到采样区域里。
https://www.cg.tuwien.ac.at/research/publications/2009/cline-09-poisson/cline-09-poisson-paper.pdf
https://bl.ocks.org/mbostock/dbb02448b0f93e4c82c3
http://devmag.org.za/2009/05/03/poisson-disk-sampling/
http://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph07-poissondisk.pdf
https://www.jasondavies.com/poisson-disc/


因为均匀采样具有仿射不变性，所以均匀采样一个三角形就可以直接采样一个特殊的**等腰直角腰长为1的三角形**了，这样生成的采样依然是均匀的，对于任意三角形，只要仿射变换后都是均匀的。

>如果三角形具有顶点ABC，其中有一个AB边上的点为$Q=(1-t)A+tB,where 0\le t\le1$，类似地，也有一个QC上的点$R=(1-s)Q+sC,where 0\le s\le 1$，于是我们有：
$$R=(1-s)(1-t)A+(1-s)B+sC$$
这个方程可以定义一个以ABC为顶点的三角形：
$$F:[0,1]x[0,1]\rightarrow R^2:(s,t) \mapsto (1-s)(1-t)A+(1-s)B+sC$$
直观地讲，这讲一个1x1的方形区域，压缩到了一个三角形中，其中一个边变成了顶点C。这种参数化的三角形在图形学中经常使用。
对于参数方程的系数，可以发现：
$$(1-s)(1-t)+(1-s)t+s = 1$$
于是可以有：
$$Q=\alpha A + \beta B + \gamma C ， where \alpha+\beta+\gamma = 1$$
$(\alpha,\beta,\gamma)$就叫做三角形的重心坐标，注意重心坐标只有两个自由度。

在一个等腰直角三角形（直角在左下）中，设重心坐标$(u,v)$，其斜边方程为$v=1-u$，均匀分布的概率$p(u,v)=\frac{1}{S}=2$，于是边沿概率：
$$p(u)=\int_0^{1-u}p(u,v)dv=2(1-u)$$(注意积分上下线为常数，积分v在v方向求和)
条件概率：
$p(v|u)=\frac{1}{1-u}$
积分得到CDF：
$$
\begin{matrix}
P(u)=\int_0^up(u')du'=2u-u^2
\newline
P(v)=\int_0^vp(v'|u)dv'=\frac{v}{1-u}
\end{matrix}
$$
由此得：
$$
\begin{matrix}
u=1-\sqrt{\xi_1}
\newline
v=\xi_2\sqrt{\xi_1}
\end{matrix}
$$
注意此两个变量不独立！
根据均匀分布的仿射不变性，这种方案可以适用于任意三角形。