---
title: 从 HiWA 到 ROCA-HiWA 学习总览：给自己和同学看的版本
date: 2026-07-10
tags:
  - HiWA
  - ROCA-HiWA
  - TACO
  - 学习文档
  - 项目总览
---

# 从 HiWA 到 ROCA-HiWA 学习总览：给自己和同学看的版本

这份文档不是实验流水账，而是给还不熟悉这个项目的人看的学习版说明。

它回答四个问题：

1. 我们一开始在做什么？
2. 为什么后来从 HiWA 走到 Soft-HiWA、ROCA-HiWA？
3. TACO 的东西我们到底用了哪些，哪些试了但没带来提升？
4. 现在真正值得继续推进的方向是什么？

---

## 1. 项目最开始：我们想复现 HiWA

我们一开始接手的是 HiWA 论文和代码。

HiWA 的全名大致可以理解为：

> Hierarchical Wasserstein Alignment，即“层级最优传输对齐”。

它要解决的问题是：

> 有两个数据域，比如神经活动 $X$ 和运动轨迹 $Y$，它们不是一一配对的，怎么在没有明确标签对应的情况下把它们对齐？

在我们的项目里：

- source domain：猕猴神经元活动；
- target domain：手臂运动 / 光标运动；
- 目标：让神经活动经过某种对齐后，能更好对应运动方向或运动轨迹。

HiWA 的核心不是普通的一个 OT，而是两层：

1. 先分组；
2. 组和组之间做 transport；
3. 每对组内部再做样本级 transport；
4. 同时估计正交变换 $R$。

你可以把它想成：

```text
神经点云 X
   ↓ 分组
神经组 X_1, X_2, ...
   ↓ 组间最优传输 P
运动组 Y_1, Y_2, ...
   ↓ 每对组内部样本 OT
更新局部旋转和全局旋转 R
```

所以 HiWA 本来就有：

- group-level OT；
- sample-level OT；
- rotation / orthogonal alignment。

这点很重要，因为后来我们讨论 TACO 时，不能误以为 HiWA 没有 sample OT。

---

## 2. HiWA 复现后发现了什么？

我们把 HiWA 的 Python 复现跑通后，发现一个现象：

> 连续运动几何的 $R^2$ 比较稳定，但方向 accuracy 对随机初始化很敏感。

这里有两个指标：

### 2.1 Direction accuracy 是什么？

方向 accuracy 可以理解为：

> 对齐之后，神经点对应的运动方向类别有没有猜对。

如果有四个方向，随机猜大约是 25%。

我们希望越高越好。

### 2.2 Movement $R^2$ 是什么？

$R^2$ 衡量的是连续运动轨迹预测得好不好。

粗略理解：

- $R^2=1$：完美解释；
- $R^2=0$：和用均值预测差不多；
- $R^2<0$：比均值预测还差。

我们的现象是：

> 有些 seed 的 $R^2$ 很高，但 direction accuracy 很低。

这说明一个重要问题：

> 连续几何对齐和离散方向语义不是同一件事。

也就是说，模型可能把整体轨迹形状对齐得不错，但方向标签语义翻了、反了、错了。

---

## 3. 为什么引入 TACO？

TACO 给我们的启发不是一句“软聚类”这么简单。

TACO 的核心思想可以拆成几块：

1. soft prototype；
2. soft assignment；
3. representative / prototype；
4. group-conditioned OT；
5. 用上层结构调控下层样本对齐。

简单说：

> TACO 不只是把样本硬分成几个组，而是让每个样本可以以不同概率属于多个 prototype，再用这些 prototype 组织跨域对齐。

所以我们想把 TACO 的思想迁移到 HiWA 上。

---

## 4. 第一版：Soft-Prototype HiWA

我们先把 HiWA 最开始的硬分组改成 soft grouping。

原来硬分组是：

```text
一个点只属于一个组
```

soft assignment 是：

```text
一个点 70% 属于组 1，20% 属于组 2，10% 属于组 3
```

数学上写成：

$$
a_{nk}^X
$$

表示 source 样本 $x_n$ 属于第 $k$ 个组的概率。

然后每个组的代表点是加权平均：

$$
r_k^X
=
\frac{\sum_n a_{nk}^Xx_n}{\sum_n a_{nk}^X}.
$$

target domain 同理：

$$
r_l^Y
=
\frac{\sum_m a_{ml}^Yy_m}{\sum_m a_{ml}^Y}.
$$

这一步得到的是：

> 每个 domain 中的 soft representatives。

---

## 5. Soft-HiWA 的结果：没有直接提高 accuracy

我们做了很多尝试：

- soft random init；
- warm start；
- temperature annealing；
- pure annealing；
- hard-start；
- rotation stabilization。

结果大致是：

| 尝试 | 结果 |
|---|---|
| soft random init | 不稳定 |
| warm start | 稳定一些，但 accuracy 没明显提升 |
| temperature annealing | 没解决 accuracy 问题 |
| pure annealing | 说明问题不是简单 prototype drift |
| hard-start | 稳定性有帮助 |

这说明：

> 单纯把硬分组改成软分组，不足以让 HiWA 变强。

但 soft representatives 后来变得非常关键，因为它帮助我们发现了真正的问题。

---

## 6. 真正关键发现：HiWA 搜索的是 O(3)，不是 SO(3)

这是整个项目最重要的发现之一。

正交矩阵集合：

$$
O(3)=\{R:R^\top R=I\}.
$$

其中包含两类：

### 6.1 det=+1

$$
SO(3)=\{R:R^\top R=I,\det(R)=+1\}.
$$

这是正常旋转，不改变手性。

### 6.2 det=-1

这是带反射的正交变换。

可以理解为镜像翻转。

我们发现：

> HiWA / Soft-HiWA 实际上可能跑到 det=+1 或 det=-1 两个分支，而这两个分支对应的 direction accuracy 差别很大。

也就是说，模型不是简单“没学好”，而是：

> 它在两个正交分支之间不稳定。

这就是 ROCA-HiWA 出现的原因。

---

## 7. ROCA-HiWA：目前最有价值的贡献

ROCA 全名：

> Representative-Oriented Component-Aware Soft-HiWA

中文可以理解成：

> 用 soft representatives 的有向几何结构，选择正确的正交分支。

核心想法：

1. 对 det=+1 跑一遍；
2. 对 det=-1 跑一遍；
3. 不看方向标签；
4. 只看 soft representatives 形成的有向四面体；
5. 用有向体积选择分支。

如果有四个代表点，在三维空间中它们可以组成一个四面体。

source 代表点的有向体积：

$$
\det([r_2^X-r_1^X,r_3^X-r_1^X,r_4^X-r_1^X]).
$$

target 代表点也有一个有向体积。

如果两个有向体积符号对应，就能判断应该选 det=+1 还是 det=-1。

选择规则大致是：

$$
\hat{s}
=
\operatorname{sign}
\left(
\det(X_{\mathrm{rep}})
\det(Y_{\mathrm{rep}})
\right).
$$

注意：

> 这个选择不使用 direction label。

---

## 8. ROCA 的真实数据结果

ROCA 在独立 seeds 50-69 上：

```text
20/20 次选中高 accuracy 分支
```

在坐标单轴符号翻转实验 seeds 70-79 上：

```text
30/30 次随坐标手性变化自动翻转选择
30/30 次选中高 accuracy 分支
```

这说明：

> ROCA 不是机械固定选择 det=-1，而是真的根据几何手性变化做选择。

这是目前最强、最干净、最能写成论文故事的结果。

---

## 9. 合成真值验证

我们还做了 synthetic determinant validation。

也就是人为构造：

- true det=+1；
- true det=-1；
- 不同 noise；
- oracle soft assignment；
- learned soft assignment；
- normal / near-degenerate representatives。

结果：

> 总体 determinant recovery 约 0.995，normal 条件基本全对，失败集中在 near-degenerate + learned soft 的弱代表点结构上。

这说明：

> ROCA 的逻辑不是只在真实数据上碰巧有效，它在合成真值中也能恢复 determinant。

---

## 10. 后来我们尝试把更多 TACO 机制加入优化

你一直提醒我：不能只用 soft clustering，TACO 后面还有东西。

所以我们后来继续试了三个模块。

---

## 11. 模块一：代表点控制 group transport

我们让代表点影响组间 transport cost：

$$
C_{kl}^{guided}
=
C_{kl}^{local}
+
\lambda_T C_{kl}^{rep}.
$$

其中：

$$
C_{kl}^{rep}
=
\|Rr_k^X-r_l^Y\|^2.
$$

这个模块的意思是：

> 如果两个组的代表点在当前旋转下更接近，那么组间 transport 更倾向于匹配它们。

结果：

- 小权重有时不破坏；
- 但 seed 50 出现过灾难性失败；
- $R^2$ 可以掉到负数。

结论：

> 代表点控制 group transport 很危险，不能作为主贡献。

---

## 12. 模块二：代表点直接参与旋转 R 更新

这是你特别指出的：

> TACO 里代表元不只是看一眼，还应该参与正交矩阵更新。

于是我们补了：

$$
\min_{R^\top R=I}
\mathcal L_{\mathrm{HiWA}}(R)
+
\lambda_R
\sum_{kl}P_{kl}\|Rr_k^X-r_l^Y\|^2.
$$

代码里新增：

```text
representative_rotation_weight
```

结果：

| $\lambda_R$ | 结果 |
|---:|---|
| 0.001 | 基本不破坏，但也不提升 |
| 0.003 | 轻微下降 |
| 0.01 | 下降明显 |

结论：

> 这个机制补齐了 TACO 思路，但在当前数据上不是提升来源。

---

## 13. 模块三：component-conditioned sample OT

这是最像 TACO 后半部分的东西。

HiWA 本来就有 sample OT。

我们不是新增 sample OT，而是把 sample OT 的 cost 从纯几何：

$$
C_{nm}^{geom}
$$

改成：

$$
C_{nm}^{total}
=
C_{nm}^{geom}
+
\beta C_{nm}^{comp}.
$$

其中：

$$
C_{nm}^{comp}
=
-\log(a_n^XP(a_m^Y)^\top+\epsilon).
$$

意思是：

> 如果两个样本的 soft group 通过 $P$ 更兼容，那么它们更容易被匹配。

结果：

| $\beta$ | Direction accuracy | Movement $R^2$ | 解释 |
|---:|---:|---:|---|
| 0 | 0.592295 | 0.601457 | baseline |
| 0.0001 | 0.592295 | 0.601405 | 基本不变 |
| 0.0003 | 0.592295 | 0.601310 | 基本不变 |
| 0.001 | 0.577849 | 0.605532 | $R^2$ 升，accuracy 降 |
| 0.003 | 0.564205 | 0.605396 | $R^2$ 升，accuracy 明显降 |
| 0.01 | 0.569021 | 0.602562 | 仍低于 baseline |

结论：

> component-conditioned sample OT 可以改变连续几何对齐，甚至略微提高 $R^2$，但会损害方向 accuracy。

这进一步说明：

> 这个数据里连续运动几何和方向语义之间存在 trade-off。

---

## 14. 所以现在我们到底做出了什么？

现在最清楚的结论是：

### 14.1 正结果

1. HiWA Python 复现跑通；
2. Soft-HiWA 跑通；
3. soft representatives 有结构；
4. 发现 O(3) determinant branch instability；
5. ROCA 可以无标签选择 determinant 分支；
6. ROCA 在真实数据和坐标翻转中表现稳定；
7. ROCA 在合成 determinant 真值中基本有效。

### 14.2 负结果

1. soft grouping 本身没有明显提高 accuracy；
2. temperature annealing 没解决问题；
3. representative group transport guidance 可能崩；
4. representative rotation guidance 没提升；
5. component-conditioned sample OT 提高 $R^2$ 时会降低 accuracy。

这些负结果不是浪费，而是告诉我们：

> 当前神经数据的关键问题不是“连续几何没对齐”，而是“O(3) 分支导致方向语义不稳定”。

---

## 15. 当前最合理的论文骨干

不要写成：

> 我们把 TACO 全部塞进 HiWA，然后全面提升。

因为实验不支持。

应该写成：

> HiWA 已经能进行连续几何对齐，但在无标签场景下存在 O(3) determinant branch instability。TACO 的 soft representatives 给了我们一种跨域结构锚点。我们提出 ROCA-HiWA，用 representatives 的有向 simplex 结构无标签选择 determinant branch，从而提升方向语义稳定性。进一步的 TACO-style optimization 模块被实现并消融，但在当前数据上不构成主要性能来源。

这才诚实，也更像真正科研。

---

## 16. 现在还不能声称什么？

不能声称：

1. ROCA 已经在所有数据集普遍有效；
2. TACO-style optimization 全面提升 HiWA；
3. component-conditioned sample OT 提高 direction accuracy；
4. 自动聚类已经完成；
5. 高维和 $K>4$ 已经解决；
6. 论文创新性已经完全确定。

---

## 17. 下一步应该做什么？

我建议下一步不要继续调 $\lambda_T,\lambda_R,\beta$。

因为我们已经知道：

> 这些优化项在当前数据上不是主增益。

下一步应该转向一般化：

### 17.1 自动 $K$

老师提到的自动聚类机器是有意义的。

可以研究：

- 如何选择 $K$；
- $K$ 变化时 ROCA 是否稳定；
- 如果 $K>d+1$，怎么选代表点 simplex。

### 17.2 simplex voting

当前 ROCA 用的是 $d=3,K=4$，刚好四个点构成四面体。

但如果 $K>4$，可以从 $K$ 个代表点里选很多个四面体，然后投票：

$$
\hat{s}
=
\operatorname{vote}
\left(
\operatorname{sign}
(\det(X_S)\det(Y_{\pi(S)}))
\right).
$$

这会让 ROCA 从特殊情况变成一般方法。

### 17.3 confidence / rejection

如果代表点接近共面，ROCA 不应该强行判断。

可以加入：

- oriented volume margin；
- condition number；
- matching margin；
- warning / reject 机制。

### 17.4 第二、第三数据集

老师说至少三个数据集。

那我们后面应该找：

1. 当前猕猴神经—运动数据；
2. 另一个神经数据，最好 session-to-session；
3. 一个非神经跨域数据，测试方法一般性。

但是注意：

> 换数据之前，主方法定义要先清楚。

---

## 18. 一句话总结

我们从复现 HiWA 开始，发现它的问题不是简单 accuracy 不够，而是无标签对齐中的正交分支不稳定。TACO 给了我们 soft representatives 的结构工具。真正成功的是 ROCA：用代表点有向几何无标签选择 determinant branch。后续把 TACO 的代表点优化、component-conditioned sample OT 都接进了 HiWA，但它们在当前神经数据上没有带来 direction accuracy 提升，因此应作为消融和机制分析，而不是主贡献。

当前最值得继续推进的是：

> 把 ROCA 从 $d=3,K=4$ 的特殊四面体选择器，推广成支持自动 $K$、simplex voting、confidence rejection、多个数据集验证的一般无标签结构稳定对齐方法。

