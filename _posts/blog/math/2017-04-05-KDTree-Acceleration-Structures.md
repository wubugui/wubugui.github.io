---
layout: post
title: "Acceleration Structures"
modified:
categories: blog
excerpt:
tags: [Ray Tracing,Acceleration Structures,KD-Tree]
image:
  feature:
date: 2017-04-05T08:13:52-03:10
---

# 概述

Acceleration Structures可以使得ray tracing复杂度从$O(n)$减少到$O(logn)$.加速比率是非常之高的，但是萦绕而来的有两大问题——高效的构建和高效的tracing，这两个问题必须平衡，以似的每一帧的渲染时间达到最小。

对于静态的场景，可以采用预先构建并且储存Acceleration Structures的方案来削减Acceleration Structures构建的时间，从而每一帧的渲染时间取决于Acceleration Structures的遍历。但是对于动态场景来说，几乎每一帧都会有object的移动或者变形，每次操作都将会导致Acceleration Structures无效化，以至于必须花时间去处理Acceleration Structures。


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
>
>假设ray均匀分布，那么$p_l$,$p_r$等于ray通过父节点，且通过本节点的概率，使用几何概率进行计算，也就是包围盒的面积比上父节点包围盒的面积。
>自顶向下构建近似地贪婪计算cost函数即可找到分割点。
>使用SAH还可作为一个终结条件：
>
>$$Terminate(N)=\left\{\begin{matrix}true & minC(N)>|N|*C_{intersect}\\ false & otherwise\end{matrix}\right.$$

# 静态场景

## 构建KD-Tree

自顶向下递归构建kdtree的算法复杂度为$O(nlogn)$($n+\frac{n}{2}+\frac{n}{2}+4* \frac{n}{4}+..+n*\frac{n}{n}$).n是primitive个数。

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
上面实现，大部分代码在使用SAH寻找分割平面，为了使得整个kdtree分布在一块连续的内存中，并以便于后处理排序以及序列化，使用了`alloc_node`自定义分配内存。

这种算法最大弊端在于每一次都要去扫描primitive来决定最终的分割面，有一种采样SAH cost的方法是**在轴上均匀分布的位置计算cost**，跨越分割点的primitive按照计入两边.对于给定的$(N_L,N_R,S_L,S_R)$，可以通过线性插值primitive数量和表面积来计算cost函数的二次逼近。对于变化很大的区间，亦可以通过采样固定数量的位置来提高近似质量。**当primitive的数量少于指定的采样数之后，再变回基本的遍历primitive的方法**来计算cost。
这种采样方法有几大优点：

-	两种算法都具有复杂度$O(nlogn)$，但采样方法可以使用simd加速来减小复杂度常数。
-	采样算法将排序等更多的工作推迟到了叶节点（也就是primitive少于指定采样数的时候），此时排序的个数变小。考虑在构造期间可能会忽略整个子树。
-	采样在数据集储存不连续（cache locality低）的时候也可以工作的比较高效，而排序在此时则相反。

## 缓存KD-Tree

对于复杂的场景来说，构建kdtree可能会耗费相当多的时间。然而，对于静态场景而言，一旦构造好一次kdtree，就可以把kdtree永久保存下来，在需要的时候再直接读入即可，如此则节约了大量的构造时间，对于大场景，暂存kdtree也是重要的。

如果kdtree被分配到了一块连续的内存之内，那序列化起来则会方便得多，而将节点集中分配并不是一件难事。序列化kdtree，主要要做两件事：
-	替换子节点偏移和重定位子节点地址
-	正确序列化和反序列化primitive索引

剩下的只需要把整块连续的内存直接二进制dump到本地文件就好了，读入的时候，将整块内存直接分配出来，然后重定位内存。

初步的实现如下：
```cpp
  class OLPBRKDTreeNodeWrite
  {
  public:
    struct
    {
      u64 size;
      //相对于vec_mem的偏移
      u64 offset;
      u64 pad;
#ifdef _DEBUG
      u64 pad1;
#endif
    } index;
    u64 left_offset;
    u64 right_offset;
    RBAABB bound;
    int split_axis;
    float split_pos;

    //debug
    int depth;
    int state;
    //序列化根节点标记
    bool root;
    bool leaf;

  };
  struct HeadWrite
  {
    //总尺寸
    u64 total_size;
    //单个节点大小
    u64 single_block_size;
    //vector偏移量
    u64 vector_offset;
    //节点个数
    u64 node_num;

  };
  
  //序列化kdtree
  static void serialize(const char* filename)
  {
    size_t block_size = linear_mem.frame.get_alloc_size(sizeof(OLPBRKDTreeNode));
    f32 tt = (f32)linear_mem.frame.get_allocated_memory() / block_size;
    int block_num = (int)tt;
    printf("serialize %f nodes!\n node size:%d\ntotal size:%d\n", tt, block_size, linear_mem.frame.get_allocated_memory());
    
    size_t node_size = linear_mem.frame.get_allocated_memory();
    size_t total_size = 0;
    for (int k = 0; k < block_num;++k)
    {
      OLPBRKDTreeNode* node = ((OLPBRKDTreeNode*)&linear_mem.mf.memory_ptr[k*block_size]);
      total_size += node->index.size()*sizeof(int);
    }

    size_t head_size = sizeof(HeadWrite);
    
    char* mem = new char[head_size+node_size+total_size];
    HeadWrite* head_mem = (HeadWrite*)mem;
    char* node_mem = &mem[head_size];
    char* vec_mem = &mem[node_size+head_size];

    head_mem->node_num = block_num;
    head_mem->single_block_size = block_size;
    head_mem->total_size = head_size + node_size + total_size;
    head_mem->vector_offset = node_size + head_size;

    if (sizeof(OLPBRKDTreeNodeWrite) != sizeof(OLPBRKDTreeNode))
      printf("write:%d~origin:%d\n", sizeof(OLPBRKDTreeNodeWrite) , sizeof(OLPBRKDTreeNode));
    CHECK(sizeof(OLPBRKDTreeNodeWrite) == sizeof(OLPBRKDTreeNode));
    CHECK(sizeof(OLPBRKDTreeNodeWrite) == block_size);

    OLPBRKDTreeNodeWrite* cur_save_node = (OLPBRKDTreeNodeWrite*)node_mem;
    size_t cur_vec_offset = 0;

    for (int k = 0; k < block_num;++k)
    {
      OLPBRKDTreeNode* node = ((OLPBRKDTreeNode*)&linear_mem.mf.memory_ptr[k*block_size]);
      memcpy(cur_save_node, node, block_size);
      size_t vsize = node->index.size()*sizeof(int);
      cur_save_node->index.size = vsize;
      cur_save_node->index.offset = cur_vec_offset;
      cur_save_node->index.pad = 0;
#ifdef _DEBUG
      cur_save_node->index.pad1 = 0;
#endif
      memcpy(vec_mem + cur_vec_offset, node->index.data(), vsize);
      cur_vec_offset += vsize;
      if (node->left)
      {
        cur_save_node->left_offset = (u64)((u8*)node->left - (u8*)linear_mem.mf.memory_ptr);
      }
      else
        cur_save_node->left_offset = 0xffffffffffffffff;
      if (node->right)
      { 
        cur_save_node->right_offset = (u64)((u8*)node->right - (u8*)linear_mem.mf.memory_ptr);
      }
      else
        cur_save_node->right_offset = 0xffffffffffffffff;

      cur_save_node++;

    }
    std::ofstream fout(filename, std::ios::binary);
    fout.write((char*)mem, head_size + node_size + total_size);
    fout.close();
    delete[] mem;
  }

  static void* deserialize(const char* filename)
  {
    release_frame();
    std::ifstream fin(filename, std::ios::binary);
    if (!fin)
    {
      printf("read %s failed!\n", filename);
      return nullptr;
    }
    fin.seekg(0, fin.end);
    int read_size = fin.tellg();
    fin.seekg(0, fin.beg);
    char* mem = new char[read_size];
    fin.read(mem, read_size);
    fin.close();

    HeadWrite* head = (HeadWrite*)mem;
    u64 node_num = head->node_num;
    u64 block_size = head->single_block_size;
    u64 total_szie = head->total_size;
    u64 vector_off = head->vector_offset;
    CHECK(total_szie==read_size);
    char* vec_mem = &mem[vector_off];
    OLPBRKDTreeNodeWrite* cur_read_node = (OLPBRKDTreeNodeWrite*)&mem[sizeof(HeadWrite)];

    void* p = linear_mem.frame.alloc(block_size*node_num,false);
    memcpy(p, cur_read_node, block_size*node_num);
    OLPBRKDTreeNode* node = (OLPBRKDTreeNode*)p;
    OLPBRKDTreeNode* ret = nullptr;

    for (int k = 0; k < node_num;++k)
    {
      u32 icount = cur_read_node->index.size / sizeof(int);
      u32 ioff = cur_read_node->index.offset;
      new(&node->index) std::vector<int>();
      node->index.resize(icount);
      memcpy((node->index.data()), vec_mem + ioff, icount*sizeof(int));
      ret = node->root ? node : ret;

      if (cur_read_node->left_offset != 0xffffffffffffffff)
        node->left = (OLPBRKDTreeNode*)((u64)cur_read_node->left_offset + (u64)p);
      else
        node->left = nullptr;
      if (cur_read_node->right_offset != 0xffffffffffffffff)
        node->right = (OLPBRKDTreeNode*)((u64)cur_read_node->right_offset + (u64)p);
      else
        node->right = nullptr;

      cur_read_node++;
      node++;
    }
    delete[] mem;
    return ret;
  }
```

## Ray遍历KD-Tree

最简单的kdtree遍历就是以任何一种顺序遍历二叉树，不过这种方法的复杂度为$O(n)$，没什么意义。

标准的遍历是沿着光线遍历，分为3种情况：
-	$t_{split}<t_{min}$ or $t_{split}>t_{max}$，仅相交一个child
-	$t_{min}<t_{split}<t_{max}$，相交两个child
-	不相交

值得注意的时候，沿ray遍历找到一个交点之后不能停止，因为一个primitive可能横跨很多个bounding box，如果遇到以下情形，停止遍历将会导致错误：

![](https://github.com/wubugui/FXXKTracer/raw/master/pic/kdtree_ray_insc.png)

可以简单地使用递归来直接实现遍历：
```cpp
static bool intersection_res(const OLPBRRay& ray, const OLPBRKDTreeNode* node, OLPBRItersc* isc, const std::vector<OLPBRGeometry*>& prims)
  {
    
    if (!node)
    {
      return false;
    }

    float tmin, tmax;
    if (!node->bound.intersection(ray.o, ray.d, tmin, tmax))
      return false;
    if (node->leaf)
    {
      bool ret = false;
      for (int i = 0; i < node->index.size();++i)
      {
        ret |= prims[node->index[i]]->intersect(ray, *isc);
      }
      return ret;
    }
    int axis = node->split_axis;
    RBVector3 inv_dir(1.f / ray.d.x, 1.f / ray.d.y, 1.f / ray.d.z);
    OLPBRKDTreeNode* first, *second;
    float tplane = (node->split_pos - ray.o[axis])*inv_dir[axis];
    if (inv_dir[axis] > 0)
    {
      if (tplane<=0)
      {
        first = node->right;
        second = nullptr;
      }
      else
      {
        first = node->left;
        second = node->right;
      }
    }
    else
    { 
      if (tplane <= 0)
      {
        first = node->left;
        second = nullptr;
      }
      else
      {
        first = node->right;
        second = node->left;
      }
    }


    //blow for debug
   // if (first || second)
    {
      bool a = intersection_res(ray,first,isc,prims);
      //如果包含了交点在aabb中才算，预防相交的交点不再本aabb中的情形
      if (a) 
      {
        RBVector3 sp = ray.o + ray.d*(isc->dist_ - isc->epsilon_);
		/*
		也可以看上图
		|    |   __B_     |
		| A  |   |~~|     |
		|____|___|__|___C_|
		光线首先穿过Bound A与C平面相交，此时返回true，然而这个交点可能不是最近的，甚至可能根本不在Bound A中！
		只要检测到的最近点一定满足：宿主object一定与A有交；一定是最近的点。
		*/
        if (first->bound.is_contain(sp))
          return a;
      }
      bool b = intersection_res(ray,second, isc, prims);
      return a||b;
    }

    //printf("kdtree intersc should not to be here!%d,%d\n",first,second);
    //return false;
  }
```
上述算法将 条件$t_{split}<t_{min}$ or $t_{split}>t_{max}$归纳为两个child都相交的情形，仅仅把$t_{split}<0$的情景记为仅相交一个child。这种归纳方案并不会有什么问题，只是相对低效一些。

另外，上述算法做了提前的遍历终结，条件为相交的最近点包含在本bounding box中：
```cpp
      if (a) 
      {
        RBVector3 sp = ray.o + ray.d*(isc->dist_ - isc->epsilon_);
        if (first->bound.is_contain(sp))
          return a;
      }
```

很多时候，递归由于函数自身的调用会产生很多问题，比如内存占用，函数call overhead，GPU不友好等等。

所以通常会使用迭代算法来进行遍历，对于要访问两个child的情形，使用一个栈来储存第二个child一遍第一个访问完成后继续访问。

```cpp
#define MAX_TODON 64
static bool intersection(const OLPBRRay& ray, const OLPBRKDTreeNode* tree, 
OLPBRItersc *isc, const std::vector<OLPBRGeometry*>& prims)
{
  float tmin, tmax;
  if (!tree) return false;
  if (!tree->bound.intersection(ray.o, ray.d, tmin, tmax)) return false;
  CHECK(tmin >= 0);
  RBVector3 inv_dir(1.f / ray.d.x, 1.f / ray.d.y, 1.f / ray.d.z);
  kd_todo_t todo[MAX_TODON];
  int todo_pos = 0;
  bool hit = false;
  const OLPBRKDTreeNode*node = tree;
  while (node != nullptr || todo_pos != 0)
  {
    if (ray.max_t < tmin) break;
    //有可能某个节点是空节点，但是此时栈中依然有节点需要处理
    if (!node)
    {
      --todo_pos;
      node = todo[todo_pos].node;
      tmin = todo[todo_pos].tmin;
      tmax = todo[todo_pos].tmax;
    }
    if (!node)
      continue;
    if (!node->leaf)
    {
      int axis = node->split_axis;
      float tplane = (node->split_pos - ray.o[axis])*inv_dir[axis];
      const OLPBRKDTreeNode* first_child, *second_child;
      int below_first = ((ray.o[axis] < node->split_pos) || (ray.o[axis] == node->split_pos&&ray.d[axis] <= 0));
      if (below_first)
      {
        first_child = node->left;
        second_child = node->right;
      }
      else
      {
        first_child = node->right;
        second_child = node->left;
      }
      if (tplane > tmax || tplane <= 0) node = first_child;
      else if (tplane < tmin) node = second_child;
      else
      {
        todo[todo_pos].node = second_child;
        todo[todo_pos].tmin = tplane;
        todo[todo_pos].tmax = tmax;
        ++todo_pos;
        node = first_child;
        tmax = tplane;
      }
    }
    else
      for (auto obj_i : node->index) hit |= prims[obj_i]->intersect(ray, *isc); 
    if (todo_pos > 0)
    {
      --todo_pos;
      node = todo[todo_pos].node;
      tmin = todo[todo_pos].tmin;
      tmax = todo[todo_pos].tmax;
    }
    else
      break;
  }
  return hit;
}
```
## 基于Rope的kdtree

有一种更加GPU友好的遍历方法，使用一个“Rope”结构，使叶节点bounding box每一个面都指向与其相邻的包含所有节点的最小子树。这样就不需要一个附加的栈来记录遍历轨迹了。

使用一个后期处理可以很简单地构建“Rope”，同时使用一个优化函数，将“Rope” push到最小的子树上去。

Rope等待实现...

## Packet ray遍历

当ray表现出一些公共属性时，某些计算仅需要做一次，就可以在所有ray之间共享。这通常会用在同原点的ray相交测试primitive上。这时候可以将这些点使用SIMD同时遍历kdtree，算法上基本和单个ray遍历是一样的，但是要使用mask来屏蔽那些不需要结算的ray，注意这种方案在靠近root的地方很有效，会随着越来越远离root节点越来越分散而丢失其优点，在这种方案下必须所有的ray都找到交点后才能退出。

# 动态场景

kdtree对于动态场景一般是重建。可以考虑使用采样SAH和binning of primitives[3] 来提升重建效率。

# 参考文献

[1] C. Wächter and A. Keller. Instant Ray Tracing: The Bounding Interval Hierarchy.
In T. Akenine-Möller and W. Heidrich, editors, Rendering Techniques 2006 (Proc. of
17th Eurographics Symposium on Rendering), pages 139–149, 2006.

[2] S. Woop, G. Marmitt, and P. Slusallek. B-KD trees for hardware accelerated ray
tracing of dynamic scenes. In GH ’06: Proceedings of the 21st ACM SIGGRAPH/Eu-
rographics symposium on Graphics hardware, pages 67–77. ACM, 2006.

[3] S. Popov, J. Günther, H.-P. Seidel, and P. Slusallek. Experiences with streaming con-
struction of SAH KD-trees. In Proceedings of the 2006 IEEE Symposium on Interac-
tive Ray Tracing, pages 89–94. IEEE, 2006.