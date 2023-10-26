---
title: "SpikeTrain可视化库搭建笔记(一)--需求调研"
date: 2023-10-25T11:03:42+08:00
draft: false
tags: ["Spike", "NWB", "Elephant"]
categories: ["SpikePyLab"]
twemoji: true
lightgallery: true
---

题外话:
需要整理一下笔记, 之前的一些不符合要求的需要修改, 例如图片过多、描述过少，缺少代码参考, 以及标题不清楚、标签/目录不够清晰等. 另外有一些内容也需要整理成高质量文档, 例如 QtDev 这个工程, 他主要是前端的开发, 也包括数据流的处理, 在整理文档的时候也要记录问题, 进行宏观的、细节的调整 ---- 尽管目前开发的中心已转向后端. 另一方面, 此前学习的关于 Java 多线程的知识, 看起来似乎很不规范(指无法一眼从标题、标签看出来大致的主题是什么)

### 系统定位
一个通用的侵入式数据可视化算法python库, 能够以插件的形式应用到其他场景, 例如用于spike分析的桌面应用.

### 接口调研
主要调研了 `elephant` 库下的 `statistic` 包, 该包下有一些统计分析的函数, 从分析功能上分三类:
一是发放率评估, 包括`mean_firing_rate`, `instantaneous_rate`, `time_histogram`, 它们接收的输入都包含 `neo.SpikeTrain`, 该类型存储单个神经元在一个时间区间里的发放时间戳.

二是锋间间隔分析, 包括 `isi` `cv` `lv`, `isi`是锋间间隔分布, 它接收 SpikeTrain 对象作为输入. `cv` 和 `lv` 则以锋间间隔序列作为输入(list, np.array, ...), 输出是一个指标性的值, 它反映着数据波动性大小.

三是统计学交叉分析, 包括 `fanofactor` `Complexity`, 它们接收 SpikeTrain 作为输入. 输出是一个衡量指标, 目前仅能理解如何去计算 `complexity_histogram`, 但还未能理解这一分析的语义, 也即结果中包含的信息.

可以看到这些统计分析中用于spike数据存储的数据结构是 `SpikeTrain`, 它表示一个神经元在一段时间内的发放时间戳, 具体的, 你可以使用一个浮点数数组表示发放时间戳, 规定数据单位(使用Quatities库)例如ms, 接着指明时间区间, 包括起始时间(若未规定则默认为0)和终止时间(必须指定), 也可以指定神经元的名称, 例如 `ch134#0`. 根据此前对 NeuroExplorer 各种功能的调研, 这一数据结构可以作为绘图分析的基础. 大多数单神经元分析都只需要这些信息(很多还需要一个bin_size参数). 另外, Elephant 库以 neo库为基础, 兼容了很多数据格式, 如果仅仅是读取发放数据并将它们存储为 SpikeTrain 格式

### 系统的需求
分三类, 每一类最好是具有统一的输入输出.
第一是单神经元分析, 这些函数的输入输出及其功能都十分明确且不复杂.
```
Rate Histogram(curve)
Interspike Interval Histograms
Autocorrelograms
Rasters
Cumulative Activity Displays
Instantaneous Firing Frequencies
Interspike Intervals vs Time
Poincare Maps
```
第二是交叉相关分析:
```
Peri-Event Histograms
Cross-correlograms
Joint PSTH
```

第三是事件相关分析:
```
Peri-Event Histograms
Cross-correlograms
Peri-Event Rasters
Joint PSTH
Synchrony versus Time
```

这三种分析的区别是输入的对象, 第一种需要单个神经元发放的时间序列, 
第二种需要多个神经元的发放时间序列, 第三种需要单个神经元的发放时间序列以及事件的刺激时间序列.

如果单从数据内容看, 对于某一个特定的事件刺激时间序列, 它也只需要时间戳序列和所在的时间区间, 而该事件的类型
(例如 start, stop, target, left, right, go等等)则可以认为是事件的名称(类比`ch134#0`).

实际上, 一部分的分析方法将神经元自身的发放序列或者另一个神经元的发放序列作为事件进行作图, 此时事件类型理解为某个神经元发放了.
例如 Autocorrelograms, 周围发放概率直方图, 假定时刻t处有发放, 该直方图描述了时刻t周围(t+Δt)有发放的概率(频率/频数)分布.统计对象为单神经元[t0, t1]内的发放序列s.对于序列s中的某个发放时刻t,将区间[t-B0, t+B1]按照bin均分, 统计每一个bin内发放的次数(形成了一个向量C_t), 遍历序列s, 将得到的向量求和即得到整个序列的周围发放频数分布向量, 可选择绘制为曲线或直方图.

进一步来说, 换一个思路, 我们需要的是一个对SpikeTrain分箱(bin)的操作, 和一个依据事件时间戳序列对SpikeTrain分段(epochs)的操作.

### 可视化算法(详细描述)











