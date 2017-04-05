---
layout: post
title: "KD-Tree Acceleration Structures"
modified:
categories: blog
excerpt:
tags: [Ray Tracing,Acceleration Structures,KD-Tree]
image:
  feature:
date: 2017-04-05T08:13:52-03:10
---

# 概述

kdtree可以使得ray tracing复杂度从$O(n)$减少到$O(logn)$.加速比率是非常之高的，但是萦绕而来的有两大问题——高效的构建和高效的tracing，这两个问题必须平衡，以似的每一帧的渲染时间达到最小。

对于静态的场景，可以采用预先构建并且储存kdtree的方案来削减kdtree构建的时间，从而每一帧的渲染时间取决于kdtree的遍历。但是对于动态场景来说，几乎每一帧都会有object的移动或者变形，每次操作都将会导致kdtree无效化，以至于必须花时间去处理kdtree。


有两种划分方法——按照空间划分和按照object划分。

按照划分方法来分类，加速结构可分为两类：
-	Adaptive结构（object划分）：一般是tree，以高效tracing为目标实现良好的orinitive空间排序。例如：Kd-trees，BVHs，bound ing interval hierarchies[1]，bounded kd-trees[2].
-	Uniform结构（空间划分）：选择划分位置是由结构自身决定的，虽然可能会有启发函数来决定划分到什么粒度。例如：grid,octree.

构建Adaptive结构的方法分为自顶向下和自底向上的。前者递归划分，可以根据整个数据集和附加信息来划分；后者则无此优点，他成组构建primitives集，然后对其分层。

递归构建一般有4中操作：
-	终结条件，比如最深深度，目标primitive个数
-	选择划分点，根据不同的启发函数，有不同的划分方式，最简单的比如根据当前volume的信息计算，复杂的比如SAH将会考虑primitive分布并且一般会基于某个cost模型。
-	节点创建
-	后处理，比如重排节点以获取更好的cache效率，或者类似基于Rope的kdtree，构建和优化rope。

>**SAH**( surface area heuristic)
>SAH是一种概率模型，目标是找到最小化cost函数的分割点：
>$$C(N)=\left\{\begin{matrix}C_{traversal}+p_l*C(N_l)+p_r*C(N_r)　for　inner　nodes\\C_{intersect}*|N|　for　leaf　nodes \end{matrix}\right.$$
>假设ray均匀分布，那么$p_l$,$p_r$等于ray通过父节点，且通过本节点的概率，使用几何概率进行计算，也就是包围盒的面积比上父节点包围盒的面积。
>自顶向下构建近似地贪婪计算cost函数即可找到分割点。
>使用SAH还可作为一个终结条件：
>$$Terminate(N)=\left\{\begin{matrix}true & minC(N)>|N|*C_{intersect}\\ false & otherwise\end{matrix}\right.$$

# 构建KD-Tree

自顶向下递归构建kdtree的算法复杂度为$O(nlogn)$($n+\frac{n}{2}+\frac{n}{2}+4* \frac{n}{4}$+..+n*\frac{n}{n}).

具体算法参考：
```cpp
class OLPBRKDTreeNode
{
public:
	OLPBRKDTreeNode(bool val):leaf(val),left(nullptr),right(nullptr),root(false),state(-1){}
	void init_as_leaf(std::vector<int> index);
	
	std::vector<int> index;
	OLPBRKDTreeNode* left;
	OLPBRKDTreeNode* right;
	RBAABB bound;
	int split_axis;
	float split_pos;

	//debug
	int depth;
  	//序列化根节点标记
  	bool root;
  	bool leaf;
};
//bd当前节点的AABB，prim图元储存实际储存，cur_index当前节点所属图元的index
static void build_kdtree(const RBAABB& bd, const std::vector<OLPBRGeometry*>& prim, std::vector<int>& cur_index, int depth, OLPBRKDTreeNode*& toparent)
{
  if (cur_index.size() < 1)
  {
    toparent = nullptr;
    return;
  }
  if (depth == 0 || cur_index.size() < single_max_n)
  {
    toparent = alloc_node(true);
    toparent->init_as_leaf(cur_index);
    toparent->bound = bd;
    toparent->depth = 0;
    toparent->split_axis = -1;
    toparent->split_pos = -1.f;
    return;
  }
  int axis = 0;
  std::vector<int> oa;
  std::vector<int> ob;
  f32 best_point;
  int best_aixs = -1;
  RBAABB bda, bdb;

  //寻找分割平面
  {
    float old_cost = MAX_F32;
    int best_index = -1;
    float inv_ta = 1.f / bd.get_surface_area();
    auto d = bd.max - bd.min;
    std::vector<split_point_t> tobesplitx;
    {
      //x
      for (int i = 0; i < cur_index.size(); ++i)
      {
        auto bdi = prim[cur_index[i]]->bound();
        tobesplitx.push_back(split_point_t(cur_index[i], true, bdi.min.x - bd.min.x));
        tobesplitx.push_back(split_point_t(cur_index[i], false, bdi.max.x - bd.min.x));
      }
      if (tobesplitx.size() > 0)
        std::sort(tobesplitx.begin(), tobesplitx.end());
      f32 aa = 0;
      f32 ab = 0;
      int na = 0, nb = tobesplitx.size()*0.5;
      for (int i = 0; i < tobesplitx.size(); ++i)
      {
        if (tobesplitx[i].low == false) nb--;
        aa = ((d.y*d.z) + tobesplitx[i].point*(d.z + d.y)) * 2;
        ab = ((d.y*d.z) + (d.x - tobesplitx[i].point)*(d.z + d.y)) * 2;
        f32 pa = aa * inv_ta;
        f32 pb = ab * inv_ta;
        f32 eb_ = (na == 0 || nb == 0) ? eb : 0.f;
        eb_ = 0;
        f32 cost = trav_t + isc_t*(1.f - eb_)*(pa*na + pb*nb);
        if (cost < old_cost)
        {
          old_cost = cost;
          best_aixs = 0;
          best_index = i;
        }
        if (tobesplitx[i].low == true) na++;
      }
    }
    std::vector<split_point_t> tobesplity;
    {
      //y
      for (int i = 0; i < cur_index.size(); ++i)
      {
        auto bdi = prim[cur_index[i]]->bound();
        tobesplity.push_back(split_point_t(cur_index[i], true, bdi.min.y - bd.min.y));
        tobesplity.push_back(split_point_t(cur_index[i], false, bdi.max.y - bd.min.y));
      }
      if (tobesplity.size() > 0)
        std::sort(tobesplity.begin(), tobesplity.end());
      f32 aa = 0;
      f32 ab = 0;
      int na = 0, nb = tobesplity.size()*0.5;
      for (int i = 0; i < tobesplity.size(); ++i)
      {
        if (tobesplity[i].low == false) nb--;
        aa = ((d.x*d.z) + tobesplity[i].point*(d.z + d.x)) * 2;
        ab = ((d.x*d.z) + (d.y - tobesplity[i].point)*(d.z + d.x)) * 2;
        f32 pa = aa * inv_ta;
        f32 pb = ab * inv_ta;
        f32 eb_ = (na == 0 || nb == 0) ? eb : 0.f;
        eb_ = 0;
        f32 cost = trav_t + isc_t*(1.f - eb_)*(pa*na + pb*nb);
        if (cost < old_cost)
        {
          old_cost = cost;
          best_aixs = 1;
          best_index = i;
        }
        if (tobesplity[i].low == true) na++;
      }
    }
    std::vector<split_point_t> tobesplitz;
    {
      //z
      for (int i = 0; i < cur_index.size(); ++i)
      {
        auto bdi = prim[cur_index[i]]->bound();
        tobesplitz.push_back(split_point_t(cur_index[i], true, bdi.min.z - bd.min.z));
        tobesplitz.push_back(split_point_t(cur_index[i], false, bdi.max.z - bd.min.z));
      }
      if (tobesplitz.size() > 0)
        std::sort(tobesplitz.begin(), tobesplitz.end());
      f32 aa = 0;
      f32 ab = 0;
      int na = 0, nb = tobesplitz.size()*0.5;
      for (int i = 0; i < tobesplitz.size(); ++i)
      {
        if (tobesplitz[i].low == false) nb--;
        aa = ((d.y*d.x) + tobesplitz[i].point*(d.x + d.y)) * 2;
        ab = ((d.y*d.x) + (d.z - tobesplitz[i].point)*(d.x + d.y)) * 2;
        f32 pa = aa * inv_ta;
        f32 pb = ab * inv_ta;
        f32 eb_ = (na == 0 || nb == 0) ? eb : 0.f;
        eb_ = 0;
        f32 cost = trav_t + isc_t*(1.f - eb_)*(pa*na + pb*nb);
        if (cost < old_cost)
        {
          old_cost = cost;
          best_aixs = 2;
          best_index = i;
        }
        if (tobesplitz[i].low == true) na++;
      }
    }

    f32 t;
    switch (best_aixs)
    {
    case 0:
      for (int i = 0; i < best_index; ++i)
        if (tobesplitx[i].low)
          oa.push_back(tobesplitx[i].prim_index);
      for (int i = best_index + 1; i < tobesplitx.size(); ++i)
        if (!tobesplitx[i].low)
          ob.push_back(tobesplitx[i].prim_index);
      t = tobesplitx[best_index].point / d.x;
      bd.split(0, t, bda, bdb);
      best_point = tobesplitx[best_index].point;
      break;
    case 1:
      for (int i = 0; i < best_index; ++i)
        if (tobesplity[i].low)
          oa.push_back(tobesplity[i].prim_index);
      for (int i = best_index + 1; i < tobesplity.size(); ++i)
        if (!tobesplity[i].low)
          ob.push_back(tobesplity[i].prim_index);

      t = tobesplity[best_index].point / d.y;
      bd.split(1, t, bda, bdb);
      best_point = tobesplity[best_index].point;
      break;
    case 2:
      for (int i = 0; i < best_index; ++i)
        if (tobesplitz[i].low)
          oa.push_back(tobesplitz[i].prim_index);
      for (int i = best_index + 1; i < tobesplitz.size(); ++i)
        if (!tobesplitz[i].low)
          ob.push_back(tobesplitz[i].prim_index);

      t = tobesplitz[best_index].point / d.z;
      bd.split(2, t, bda, bdb);
      best_point = tobesplitz[best_index].point;
      break;
    default:
      break;
    }
  }

  OLPBRKDTreeNode* a, *b;

  //处理扁平的bounding box
  if (bda.max[axis] - bda.min[axis] <= SMALL_F)
  {
    a = nullptr;
    b = alloc_node(true);
    b->init_as_leaf(cur_index);
    b->bound = bd;
    b->depth = 0;
    b->split_axis = -1;
    b->split_pos = -1.f;
  }
  else
  {
    build_kdtree(bda, prim, oa, depth - 1, a);
    build_kdtree(bdb, prim, ob, depth - 1, b);
  }

  toparent = alloc_node(false);
  toparent->bound = bd;
  toparent->left = a;
  toparent->right = b;
  toparent->depth = depth;
  toparent->split_axis = best_aixs;
  toparent->split_pos = best_point + bd.min[best_aixs];
}
```


# 参考文献

[1]C. Wächter and A. Keller. Instant Ray Tracing: The Bounding Interval Hierarchy.
In T. Akenine-Möller and W. Heidrich, editors, Rendering Techniques 2006 (Proc. of
17th Eurographics Symposium on Rendering), pages 139–149, 2006.

[2] S. Woop, G. Marmitt, and P. Slusallek. B-KD trees for hardware accelerated ray
tracing of dynamic scenes. In GH ’06: Proceedings of the 21st ACM SIGGRAPH/Eu-
rographics symposium on Graphics hardware, pages 67–77. ACM, 2006.