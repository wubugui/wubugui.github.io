---
layout: post
title: "Blender基本操作"
modified:
categories: blog
excerpt:
tags: [blender]
image:
feature:
date: 2016-07-22T08:08:50-14:02
---

# 几句废话

因为最近想把自己的渲染器搞成blender插件这样创作测试起来更加方便，于是先学习了一下blender，然而Blender的操作对于用贯Max的我来说实在有些别扭，没有用过根本就搞不懂，所以写这篇博客绝对很有价值，哈哈哈。。。

# 窗口与面板

Ctrl/Shift+鼠标左右中按键可以操作视角。

`Ctrl + <-   ` 和` Ctrl+->` **切换不同的应用布局**。

*`Ctrl+Alt+U`弹出用户设置，如需回到默认设置，到根目录`\blender\config\`中删除`startup.blend`即可。*

面板右上角往左边拖动时**拉出一个新面板**（默认是和当前面板一样的），如果想要**收起这个面板**，就把这个右上角的三角形拖到另外一个面板上，那么被覆盖的面板就消失了。

在面板内`Ctrl+MMB`上下移动，**缩放面板内容大小**，鼠标放在上面按`Home`**恢复原来大小**。

面板内鼠标右键选择**面板布局**。

`Shift-Spacebar`,` Ctrl-Down `or` Ctrl-Up` 来决定一个面板的**最大化**。

`Ctrl`点击右上角三角拖动到一个面板，**两个面板交换内容**。

`Shift`点击右上角三角拖动一个面板，可以把这个面板拖出来变成一个新窗口。

`Ctrl-Shift-Z`重做

Alt+RMB 选择一圈边

在3DView按`N`可以显示一个很有用的辅助面板，包括显示法线。**改变大小位置**这些功能等
￼

![](https://github.com/wubugui/FXXKTracer/raw/master/blender/22-46-11.jpg)


`Shift+D`复制

选中物体按M可以把它挪到另外一层

`Ctrl+A`对3DView中一个对象应用一个效果

# Pie Menus

在`文件-用户设置-插件-社区版-用户界面`打开User Interface：Pie Menus Official。

然后在3DView中就可以使用Pie Menus了：

-   Tab：交互模式
-   Z：  Shade+solid vs smooth shading
-   Q：摄像机
-   Tab+Shift+Ctrl：Snapping
-   .：锚点
-   Ctrl+Space：操作，平移旋转缩放

# 蜡笔系统

蜡笔系统用于打草稿，可以在任何地方画东西。

3DView中的蜡笔选项卡
￼

![](https://github.com/wubugui/FXXKTracer/raw/master/blender/17-21-21.jpg)


中可以选中测量标尺，以及量角器，单击进入这个模式，`Ctrl+MLB`添加一把标尺。如果在标尺的中间拉动可以量角。ESC退出。Delete删除标尺。

**绘画 D-LMB**

画出新的笔画（多根短线，连接的线）。画的笔画会在你释放鼠标后停止。

**线 Ctrl-D-LMB**

以橡皮筋模式画一根线（定首末点位置）。释放鼠标按钮的时候线绘制完成。

**多边形 Ctrl-D-RMB**

点几下点就画出一些相互连接的线。这些线会自动添加在两个点之间。

**擦除 D-RMB**

添加displacement要在数据选项卡处理添加UV贴图的数据。

----

随便搞了一下，渲染了一支铅笔（以后或许会有更多渲染作品^_^）：

￼

![](https://github.com/wubugui/FXXKTracer/raw/master/blender/untitled1.png)


放张相同时间用传统着色方法着色调出来的图：


![](https://github.com/wubugui/FXXKTracer/raw/master/blender/untitled2.png)
￼
