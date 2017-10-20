---
layout: post
title: "实时渲染Notes"
modified:
categories: blog
excerpt:
tags: [游戏引擎,实时渲染]
image:
feature:
date: 2016-11-23T08:08:50-21:00
---

最近一个季度公司任务不太近所以一直在琢磨实时渲染和GPU编程，把做的几个东西记录一下，等明年再拉出来优化或者重写。

# 基于物理着色

因为之前一直在学习离线渲染，加上前段时间也从理论上仔细学习了一下基于物理着色]所以这个参考一些notes实现起来相对容易，主要要注意实时渲染中的效率问题。

第一步是着色模型的选取。

漫反射项我尝试了Disney的漫反射项

$$f_{d}=\frac{baseColor}{\pi}(1+(F_{D90}-1)(1-cos\theta_{l})^5)(1+(F_{D90}-1)(1-cos\theta_{v})^5)$$

where

$$F_{D90} = 0.5+2*roughness*cos^2\theta_{d}$$

和

[Lambertian](http://claireswallet.farbox.com/post/graphics/ji-yu-wu-li-zhao-se#toc_1).

两者效果差别不大，前者仅仅是因为菲涅尔在边缘有了一层非常不明显的亮环。这个现象UE4在他们的course notes[1]里面也有提到过。

高光项选择了微表面模型，DG、F分别采用了GGX（Smith）、Schlick（菲涅尔也做了以下，Schlick的近似效果差不多，但效率要高得多，只需要插值）.

着色时，同时使用了基于图像的环境漫反射和环境高光。其中漫反射环境纹理采用了预过滤的irradiance map，环境光也近似地使用了预过滤的irradiance map，不管是当做漫反射来预过滤的，结果肯定是不正确，但是效果还可以，正统的做法到处都有提到，其中GPUPro上就有。
为了达到一个良好的效果，还使用一定的方法根据着色参数调整了每个部分的权重。

```cpp
float3 derive_diffuse_color(float3 base_color, float metallic)
{
    return base_color - base_color * metallic;
}
float3 derive_specular_color(float3 base_color, float3 specular, float metallic)
{
    //from UE4
    return lerp(0.08 * specular.xxx, base_color, metallic);
}
float3 diffuse_ibl(float rs, float metallic, float3 diffuse, float cspec, float3 n, float3 v, float3 l, float3 clight)
{
    float3 base = diffuse; 
    float3 cdiff = derive_diffuse_color(base, metallic);
    float roughness = rs;
    float3 diff = diffuse_lambert(cdiff);
    cspec = derive_specular_color(base, cspec, metallic);
    return diff * clight / PI;
}
float3 specular_ibl(float rs, float metallic, float3 diffuse, float3 cspec, float3 n, float3 v, float3 l, float3 clight)
{
    float3 base = diffuse; 
    cspec = derive_specular_color(base, cspec, metallic);
    float3 h = normalize(v + l);
    return cspec * clight * (1 - rs) * metallic;
}
```

基于物理着色上实用的美术资源主要有：

Albedo纹理，Normal纹理以及Roughness和metallic纹理，植被铁链则还包含Mask纹理。

另外特别要注意的是，由于基于物理着色会积累辐射度，所以HDR渲染是必须的。

HDR的流程大概是：

-   渲染浮点结果
-   计算浮点结果的平均明度
-   根据平均明度使用filmic tonemapping operator调整图像
-   降采样原图截取高亮部分图像blur
-   混合结果

关于HDR渲染，Intel有个Demo很好地诠释了整个过程，值得参考。

## 实验结果

![](https://github.com/wubugui/FXXKTracer/raw/master/pic/17-25-41.jpg)
![](https://github.com/wubugui/FXXKTracer/raw/master/pic/QQ%E5%9B%BE%E7%89%8720161123162035.jpg)
![](https://github.com/wubugui/FXXKTracer/raw/master/pic/17-38-38.jpg)

注意图中吊着的火盆呈全黑，主要是因为金属性很强，但是又没有上envmap.



# 景观云与物理天穹

## 物理天穹

大气这个主要参考了[3].

大气强度衰减公式：

$$density(h)=density(0) e^{-\frac{h}{H}}$$

大气散射粒子分为气体原子散射和尘埃闪射。前者叫做Rayleigh scattering后者叫做Mie scattering。前者负责呈现天空的蓝色和日落日出时的红色，后者负责灰白色（大气污染那种颜色）。

注意单位差距，粒子的单位在毫米微米级，而星球的半径则在千米级。

### Rayleigh Scattering

在体渲染中，衰减系数决定散射量，相位函数决定散射方向。其中衰减系数为散射系数和吸收系数之和。

散射系数：

$$\beta_R^s  (h, \lambda) = \frac{8\pi^3(n^2-1)^2}{3N\lambda^4} e^{-\frac{h}{H_R}}$$

$h$是海拔高度，$N$是海平面气体原子密度，$\lambda$是波长，$n$是气体折射率，$H_R$是scale height。*我们使用$H_R=8km$*.

注意：我们完全可以使用一些测量值来计算N，n等等参数，但是我们这里将会使用预计算的Rayleigh散射系数（ $33.1 \mathrm{e^{-6}} \mathrm{m^{-1}}$,$13.5 \mathrm{e^{-6}} \mathrm{m^{-1}}$,$5.8 \mathrm{e^{-6}} \mathrm{m^{-1}}$ for wavelengths 440, 550 and 680 respectively.）

在大气渲染中，吸收可以忽略不计所以：

$$\beta_R^e = \beta_R^s$$

相位函数为：

$$P_R(\theta)=\frac{3}{16\pi}(1+\theta^2)$$

其中$\theta$是入射光线与视线的夹角。


### Mie Scattering

散射系数：

$$\beta_M^s(h,\lambda)=\beta_M^s(0,\lambda) e^{-\frac{h}{H_M}}$$

$H_M$是scale height 通常为1.2km.我们使用预计算值$\beta_S = 210 \mathrm{e^{-5}} \mathrm{m^{-1}}$。衰减系数是散射系数的约1.1倍。

相位函数：

$$P_M(\mu)=\frac{3}{8\pi}\frac{(1-g^2)(1+\mu^2)}{(2+g^2)(1+g^2-2g\mu)^{\frac{3}{2}}}$$

$g$用来决定各向异性，前向为[0,1],后向为[-1,0].我们取0.76，0代表各向同。

### 沿视线做Ray March

体渲染方程：

$$L(P_c, P_a)=\int_{P_c}^{P_a} T(P_c,X)L_{sun} (X)ds$$
$$T(P_a,P_b)=\frac{L_a}{L_b}=exp(-\sum_{P_a}^{P_b} \beta_e(h)ds)$$


$$\beta_s(h) = \beta_s(0)exp(-\frac {h}{H})$$
$$\beta_e(h)=\beta_s(h)+\beta_a(h)=\beta_s(h)+0=\beta_s(h)$$
$$T(P_a, P_b)=exp(-\beta_e(0) \sum_{P_a}^{P_b} exp(-\frac{h}{H})ds)$$

$$L_{sun}(X)=Sun Intensity * T(X,P_s) * P(V,L)*\beta_S(h)$$

最终：

$$Sky \: Color(P_c, P_a) = \int_{P_c}^{P_a} T(P_c, X) * Sun \: Intensity * P(V,L) * T(X,P_s)*\beta_s(h)ds$$

对于方程中的$SunIntensity$和$P(V,L)$都是常量可以提出来，两个$T$合在一起：

$$T(P_c,X) * T(X,P_s)=e^{ -\beta_{e0} } * e^{ -\beta_{e1} }=e^{ -(\beta_{e0} + \beta_{e1}) }$$

最后我们将分别计算Rayleigh和Mie各一次加到一起。

### GPU初步实现

我是用了一个全屏pass渲染的，基本就是一个后处理。但是渲染在frame的最前面。PS传入摄像机参数生成每个像素的View方向。

需要注意的地方是数据精度问题，宇宙级的半径和原子半径差距太大，再加上积分操作，很容易造成精度丢失。我最初的时候把单位统一为cm，最后输出了一片黑，大概算了一下，发现光学深度这个数据非常的小，因为拿取做指数衰减的项非常大，宇宙级的。

仔细观察不难发现那些很小的部分都是线性乘上去的，前面的计算和mm级的数据没有交集，所以前面的计算可以使用千米为单位，在某些预估可能会比较大的地方转化成m。在整个计算过程中根据计算项的意义不断地调整单位，最后可以把单位全部统一到m。

这个统一到m的量就可以使用一个scale来统一缩放了。

还有一点是，在计算view方向的积分的时候，最好是把有效的采样范围缩到最小，这样可以提高采样率，否则当摄像机在大气外时，地球半径级别的距离会导致采样率非常低。

```cpp
struct VS_Output
{  
    float4 Pos : SV_POSITION;              
    float2 Tex : TEXCOORD0;
};

VS_Output main_vs(uint id : SV_VertexID)
{
    VS_Output Output;
    Output.Tex = float2((id << 1) & 2, id & 2);
    Output.Pos = float4(Output.Tex * float2(2,-2) + float2(-1,1), 0, 1);
    return Output;
}

#define M_PI 3.141592653

int isc_sphere(float3 ro,float3 rd,float3 pos,float r,inout float t1,inout float t2)
{
	float t;
	float3 temp = ro - pos;
	float a = dot(rd, rd);
	float b = 2.f*dot(temp, rd);
	float c = dot(temp, temp) - r*r;
	float ds = b*b - 4 * a*c;

		
	int isc_n = 0;
	if (ds < 0.f)
	{
		return isc_n;
	}
	else
	{
		float e = sqrt(ds);
		float de = 2.f * a;
		t = (-b - e) / de;

		if (t > 0.00001)
		{
			t1 = t;
			isc_n++;
			//return isc_n;
		}

		t = (-b + e) / de;

		if (t>0.00001)
		{
			t2 = t;
			isc_n++;
			//return isc_n;
		}
	}
	return isc_n;
}

cbuffer CB
{
	//km
	float3 sphere_pos_v;
	float view_dist;
	float3 sun_dir;
	float sacle_factor;
	float tan_fov_x;
	float tan_fov_y;
	int n_sample;
	int n_sample_light;
	matrix mv_mat;
};

float4 main_ps(VS_Output pin): SV_TARGET0
{
	sphere_pos_v = mul(sphere_pos_v*1000,mv_mat);
	sphere_pos_v/=1000;
	sun_dir = mul(sun_dir,mv_mat);
	
	//km
	float er = 6360;
	float ar = 6420;
	//m
	float hr = 7994;
	float hm = 1200;
	
	//mm
	float3 betar= float3(0.0058,0.0135,0.0331);
	float3 betam = float3(0.021,0.021,0.021);
	
	float tx = 2*pin.Tex.x - 1;
	float ty = 2*(1-pin.Tex.y) - 1;
	float vx = tx*view_dist* tan_fov_x;
	float vy = ty*view_dist*tan_fov_y;
	
	//cm
	float3 view_dir = float3(vx,vy,view_dist);
	view_dir = normalize(view_dir);
	
	float t1 = -1;
	float t2 = -1;
	float t11 = -1;
	float t12 = -1;
	int ik = isc_sphere(float3(0,0,0),view_dir,sphere_pos_v,ar,t1,t2);
	//检测内球交点，提升采样率
	int ic = isc_sphere(float3(0,0,0),view_dir,sphere_pos_v,er,t11,t12);
	float3 pa,pb;
	if(ik==0)
		discard;
		//return float4(1,0,0,1);
	if(ik==1)
	{
		pa = float3(0,0,0);
		pb = view_dir*t2;
		//return float4(0,0,1,1);
	}
	else if(ik==2)
	{
		
		if(ic<=1)
		{
			pa = view_dir*t1;
			pb = view_dir*t2;
		}
		else if(ic==2 )
		{
			pa = view_dir*t1;
			pb = view_dir*t11;
		}
		//return float4(0,1,0,1);
	}
	
	
	//km
	float seg_len = length(pb-pa)/n_sample;
	float3 sumr=0,summ=0;
	float optiacal_depth_r=0,optical_depth_m=0;
	float theta = dot(view_dir,sun_dir);
	float phaser = 3.f / (16.f * M_PI) * (1 + theta * theta);
	float g = 0.76;
	float phasem = 3.f / (8.f * M_PI) * ((1.f - g * g) * (1.f + theta * theta)) / ((2.f + g * g) * pow(1.f + g * g - 2.f * g * theta, 1.5f));
		
	for(uint i=0;i<n_sample;++i)
	{
		//km
		float3 sample_pos = pa + seg_len*(i+0.5)*view_dir;
		float samlpe_height = length(sample_pos - sphere_pos_v) - er;
		//km to m
		samlpe_height*=1000;
		float h_r = exp(-samlpe_height/hr)*seg_len;
		float h_m = exp(-samlpe_height/hm)*seg_len;
		//to km
		optiacal_depth_r += h_r;
		optical_depth_m += h_m;
		float t0Light;
		float t1Light;
		int sn = isc_sphere(sample_pos,sun_dir,sphere_pos_v,ar,t0Light,t1Light);

		//km
		float seg_len_sun = t1Light/n_sample_light;
		float optiacal_depth_sun_r=0,optical_depth_sun_m=0;
		uint j = 0;
		for(j=0;j<n_sample_light;++j)
		{
			float3 sample_pos_sun = sample_pos + seg_len_sun*(j+0.5)*sun_dir;
			float sample_height_sun = length(sample_pos_sun - sphere_pos_v) - er;
			//km to m
			sample_height_sun*=1000;
			//km
			optiacal_depth_sun_r += exp(-sample_height_sun/hr)*seg_len_sun;
			optical_depth_sun_m += exp(-sample_height_sun/hm)*seg_len_sun;			
		}
		if(j==n_sample_light)
		{
			//mm*km=m^2
			float3 tau = betar*(optiacal_depth_r+optical_depth_m) + betam*1.1f*(optiacal_depth_sun_r+optical_depth_sun_m);
			float3 attenuation = float3(exp(-tau.x),exp(-tau.y),exp(-tau.z));
			sumr += attenuation*h_r;
			summ += attenuation*h_m;
		}
	}
	
	float3 sky_color = clamp((sumr*betar*phaser + summ*betam*phasem)*sacle_factor,0.0001,100000000);
	return float4(sky_color,1);
}
```
这只是一个试验级别的实现，想到达到一个稳定良好的效果还需要抽时间进一步优化。


## 景观云

景观云的实现主要参考自[2]，GPUPro和2017年的course上有更加详细的描述。

主要的思路就是使用FBM叠加不同频率的Perlin噪声和Worley噪声，其中Berlin模拟云的大体形状，Worley负责在大体形状上添加细节。叠加后的噪声被他们称为Perlin-Worley噪声。

噪声是离线生成的，生成两个3D纹理作为噪声源：

其中一个使用128分辨率1通道Perlin-Worley 2通道不同频率FBM后的Worley，这个纹理用作云的大体外形。

另外一个使用32分辨率，3个通道每个都是不一样频率的Worley，用于添加云的细节。

另外，Slide上提到使用了一个2D纹理的curl噪声来扰动云的边界，但是我暂时没有实现这个。

然后使用纹理来控制云的而生成高度，不同层的云具有不同的厚度和高度，这些都可以通过纹理来进行灵活的控制。我的实现中主要是通过几个参数来控制云的高度和厚度，灵活性不如纹理好，但是实现起来更简单。
最后使用一个称为天气纹理的纹理来控制每个区域内云的覆盖情况，根据覆盖情况还可以决定是否下雨等等。

云的渲染主要还是使用常见的体渲染方法Ray March来搞，不过他们做了很多优化，另外为了凸显云的各向异性使用了Henyey-Greenste相位函数。

目前只做了基本的实现，要做到相对完美，还需在未来更多的工作，包括建模细节以及优化上。另外，这个方案的强大之处在于给予艺术家的控制很足，在这方面来讲要真正实现出来还满不容的。

注意这个方法在2017的siggraph realtime course着重又讲过！

## 实验结果

![](https://github.com/wubugui/FXXKTracer/raw/master/pic/%E6%9C%AA%E6%A0%87%E9%A2%98-1.png)
![](https://github.com/wubugui/FXXKTracer/raw/master/pic/17-11-39.jpg)
![](https://github.com/wubugui/FXXKTracer/raw/master/pic/17-16-41.jpg)
![](https://github.com/wubugui/FXXKTracer/raw/master/pic/17-17-11.jpg)

# GPU SVO

过程大致就是体素化到链表，一层一层建立SVO，写入节点值到Brick纹理。

第一步体素化，首先要摆好模型位置，我将模型摆在原点，然后摄像机放在z轴上正对模型，距离就是体素化尺寸的一般，注意必须做到摄像机有各向同性，这样就不需要使用三个摄像机来分别计算透视矩阵了，只需要在Shader里面改变一下坐标轴，就可以使用同一个透视矩阵来透视。

```cpp
if(max_axis==sabs.x)
        idx = 1*sign(s.x);
    else if(max_axis==sabs.y)
        idx = 2*sign(s.y);
    else if(max_axis==sabs.z)
        idx = 3*sign(s.z);
    float4 new_vert[3];
    float3 posw[3];
    uint ma[3]={2,2,2};
    for(int j=0;j<3;++j)
    {
        idx = -abs(idx);ma[j] = -idx;
        if(idx==-1)
        {
           new_vert[j] = float4(gin[j].position.zyx,1);
            new_vert[j] = mul(new_vert[j],v);
            new_vert[j] = mul(new_vert[j],o);
            gin[j].position.xyz = new_vert[j].xyz;
           posw[j] = new_vert[j].xyz;
        }
        else if(idx==-2)
        {    
            new_vert[j] = float4(gin[j].position.xzy,1);
            new_vert[j] = mul(new_vert[j],v);
            new_vert[j] = mul(new_vert[j],o);
            gin[j].position.xyz = new_vert[j].xyz;
            posw[j] = new_vert[j].xyz;
        }
        else if(idx==-3)
        { 
           new_vert[j] = float4(gin[j].position.xyz,1);
           new_vert[j] = mul(new_vert[j],v);
            new_vert[j] = mul(new_vert[j],o);
            gin[j].position.xyz = new_vert[j].xyz;
            posw[j] = new_vert[j].xyz;
        }
```

当然这么做在shader中添加了很多分支语句也不直观并不一定好。

然后是保守保守光栅化，由于我的显卡并不支持保守光栅化，所以只能自己来实现，实现的摘要在[这里](http://claireswallet.farbox.com/post/graphics/2016-08-31#toc_3)，这个实现在深度和纹理坐标上还有些细小的问题，不过对体素化没有什么打的影响。
最后PS里面在grid里定位体素，然后写入体素链表。值得注意的是：如果三角形很小，或者三角形重叠，就会有相同位置的像素多次写入值，正确处理这种情形必须要混合这些值，否则就会丢失信息，这种情况在之后的遍历构建SVO的时候也会出现。

紧接着要对每个level执行flag、alloc、init操作，并在适当的时候更新参数。最后拿出一个专门的pass来遍历建立好的SVO，在目标叶节点链接并且写入Brick。

可视化这个结果采用cs生成一个顶点buffer，然后用gs生成方块渲染出来。

之前犯了一个小错误，在遍历SVO的时候需要计算下一个位置，由于把0.5判给了下一级节点，导致整个程序渲染结果都是错的。查了好些时候才查出来。

# 实现结果

![](https://github.com/wubugui/FXXKTracer/raw/master/pic/QQ%E5%9B%BE%E7%89%8720161123162223.png)
![](https://github.com/wubugui/FXXKTracer/raw/master/pic/QQ%E5%9B%BE%E7%89%8720161123162254.png)
(512X512X512体素化，128M体素链表，256MSVO结点，64M Brick Pool)

# Reference

-   [1] Real Shading in Unreal Engine 4
-   [2] THE REAL-TIME VOLUMETRIC CLOUDSCAPES OF HORIZON: ZERO DAWN,ADVANCES IN REAL-TIME RENDERING 2015
-   [3] http://www.scratchapixel.com/lessons/procedural-generation-vritual-worlds/simulating-sky
