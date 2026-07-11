---
title: 正交 Procrustes 问题的求解
tags:
  - HiWA
  - optimal-transport
  - Procrustes
  - SVD
  - Stiefel-manifold
  - Obsidian
created: 2026-06-28
---

# 正交 Procrustes 问题的求解

## 0. 本节在 HiWA 论文主线中的位置

我们前面已经学过 HiWA 的核心框架：

```text
分布对齐
↓
Wasserstein 距离
↓
经验测度
↓
运输计划 Q_ij
↓
局部代价 C_ij(R,Q_ij)
```

本节继续解释论文方法中的下一步：

```text
固定 Q_ij
↓
如何更新正交矩阵 R
↓
正交 Procrustes 问题
↓
SVD 求解
```

在 HiWA 中，$Q_{ij}$ 负责第 $i$ 个源簇 $X_i$ 和第 $j$ 个目标簇 $Y_j$ 之间的点级软对应，而 $R$ 负责把源数据整体旋转或反射到目标数据所在的坐标系中。

本节要解决的问题是：

> 当 $Q_{ij}$ 已经固定时，如何求最优的正交变换 $R$？

---

## 1. HiWA 中为什么需要正交矩阵 $R$

源数据簇和目标数据簇通常不在同一个坐标系中。即使它们内部结构相似，直接比较

$$
X_i(k)
$$

和

$$
Y_j(l)
$$

也可能距离很远。

因此 HiWA 不直接比较：

$$
\|X_i(k)-Y_j(l)\|_2^2
$$

而是比较：

$$
\|RX_i(k)-Y_j(l)\|_2^2
$$

这里 $R$ 是一个正交矩阵，满足：

$$
R^\top R=I.
$$

它表示旋转或反射。

### 1.1 正交矩阵保持长度

对任意向量 $x$，有：

$$
\|Rx\|_2^2
=
(Rx)^\top(Rx)
=
x^\top R^\top R x
=
x^\top I x
=
\|x\|_2^2.
$$

所以正交矩阵不会拉长或压缩向量。

### 1.2 正交矩阵保持距离

对任意两个点 $x,y$，有：

$$
\|Rx-Ry\|_2
=
\|R(x-y)\|_2
=
\|x-y\|_2.
$$

所以 $R$ 保持点与点之间的距离。

### 1.3 正交矩阵在 HiWA 中的意义

HiWA 假设源数据和目标数据的内部几何结构大体保持，只是坐标系可能不同。因此，论文选择正交变换，而不是任意线性变换。

可以这样记：

```text
R 只允许数据整体旋转或反射；
R 不允许数据被随意拉伸、压缩、扭曲。
```

---

## 2. HiWA 的局部代价函数

论文中，给定簇对 $(i,j)$ 时，局部代价为：

$$
C_{ij}(R,Q_{ij})
=
\frac{1}{D}
\sum_{k,l}
Q_{ij}(k,l)
\|RX_i(k)-Y_j(l)\|_2^2.
$$

其中：

- $D$ 是数据维度；
- $Q_{ij}(k,l)$ 是从 $X_i(k)$ 到 $Y_j(l)$ 的运输质量；
- $R$ 是正交变换；
- $\frac{1}{D}$ 是维度归一化。

这一项的意思是：

> 在给定软对应 $Q_{ij}$ 和正交变换 $R$ 的情况下，第 $i$ 个源簇和第 $j$ 个目标簇之间的平均每维对齐代价。

本节固定 $Q_{ij}$，只考虑如何优化 $R$。

因此要解的问题是：

$$
\min_{R^\top R=I}
\sum_{k,l}
Q_{ij}(k,l)
\|RX_i(k)-Y_j(l)\|_2^2.
$$

为了推导方便，暂时省略常数因子 $\frac{1}{D}$。因为 $\frac{1}{D}>0$，它不会改变最优的 $R$。

---

## 3. 固定 $Q_{ij}$ 后得到正交 Procrustes 问题

令：

$$
x_k=X_i(k),
$$

$$
y_l=Y_j(l),
$$

$$
Q_{kl}=Q_{ij}(k,l).
$$

问题变为：

$$
\min_{R^\top R=I}
\sum_{k,l}
Q_{kl}
\|Rx_k-y_l\|_2^2.
$$

这就是一个加权的正交 Procrustes 问题。

---

## 4. 展开平方项

先看单个平方项：

$$
\|Rx_k-y_l\|_2^2
=
(Rx_k-y_l)^\top(Rx_k-y_l).
$$

展开得到：

$$
\|Rx_k-y_l\|_2^2
=
x_k^\top R^\top R x_k
-
2y_l^\top R x_k
+
y_l^\top y_l.
$$

因为 $R^\top R=I$，所以：

$$
x_k^\top R^\top R x_k
=
x_k^\top x_k
=
\|x_k\|_2^2.
$$

因此：

$$
\|Rx_k-y_l\|_2^2
=
\|x_k\|_2^2
+
\|y_l\|_2^2
-
2y_l^\top R x_k.
$$

代回目标函数：

$$
\sum_{k,l}
Q_{kl}
\|Rx_k-y_l\|_2^2
=
\sum_{k,l}
Q_{kl}
\|x_k\|_2^2
+
\sum_{k,l}
Q_{kl}
\|y_l\|_2^2
-
2
\sum_{k,l}
Q_{kl}
y_l^\top R x_k.
$$

前两项不依赖于 $R$，只有最后一项依赖于 $R$。

所以最小化原目标等价于最大化：

$$
\sum_{k,l}
Q_{kl}
y_l^\top R x_k.
$$

于是得到等价问题：

$$
\max_{R^\top R=I}
\sum_{k,l}
Q_{kl}
y_l^\top R x_k.
$$

---

## 5. 为什么可以写成迹的形式

现在解释：

$$
\sum_{k,l}
Q_{kl}
y_l^\top R x_k
=
\operatorname{tr}
\left(
R^\top YQ^\top X^\top
\right).
$$

这里为了简化记号，令：

$$
X=[x_1,x_2,\dots,x_m]\in\mathbb{R}^{D\times m},
$$

$$
Y=[y_1,y_2,\dots,y_n]\in\mathbb{R}^{D\times n},
$$

$$
Q\in\mathbb{R}^{m\times n}.
$$

### 5.1 第一步：把标量写成矩阵元素

因为标量转置不变：

$$
y_l^\top R x_k
=
x_k^\top R^\top y_l.
$$

观察矩阵：

$$
X^\top R^\top Y.
$$

它的维度是：

$$
m\times n.
$$

它的第 $(k,l)$ 个元素为：

$$
(X^\top R^\top Y)_{kl}
=
x_k^\top R^\top y_l
=
y_l^\top R x_k.
$$

所以：

$$
\sum_{k,l}
Q_{kl}
y_l^\top R x_k
=
\sum_{k,l}
Q_{kl}
(X^\top R^\top Y)_{kl}.
$$

### 5.2 第二步：识别为 Frobenius 内积

Frobenius 内积定义为：

$$
\langle A,B\rangle_F
=
\sum_{k,l}
A_{kl}B_{kl}.
$$

因此：

$$
\sum_{k,l}
Q_{kl}
(X^\top R^\top Y)_{kl}
=
\langle Q,X^\top R^\top Y\rangle_F.
$$

而 Frobenius 内积可以写成迹：

$$
\langle A,B\rangle_F
=
\operatorname{tr}(A^\top B).
$$

所以：

$$
\langle Q,X^\top R^\top Y\rangle_F
=
\operatorname{tr}
\left(
Q^\top X^\top R^\top Y
\right).
$$

### 5.3 第三步：使用迹的循环性质

迹满足循环性质：

$$
\operatorname{tr}(ABCD)
=
\operatorname{tr}(DABC).
$$

因此：

$$
\operatorname{tr}
\left(
Q^\top X^\top R^\top Y
\right)
=
\operatorname{tr}
\left(
R^\top YQ^\top X^\top
\right).
$$

所以：

$$
\sum_{k,l}
Q_{kl}
y_l^\top R x_k
=
\operatorname{tr}
\left(
R^\top YQ^\top X^\top
\right).
$$

回到论文记号，就是：

$$
\sum_{k,l}
Q_{ij}(k,l)
Y_j(l)^\top R X_i(k)
=
\operatorname{tr}
\left(
R^\top Y_jQ_{ij}^\top X_i^\top
\right).
$$

---

## 6. 得到标准正交 Procrustes 形式

定义：

$$
A_{ij}
=
Y_jQ_{ij}^\top X_i^\top.
$$

则优化问题变成：

$$
\max_{R^\top R=I}
\operatorname{tr}
\left(
R^\top A_{ij}
\right).
$$

这就是标准的正交 Procrustes 问题。

一般形式为：

$$
\max_{R^\top R=I}
\operatorname{tr}(R^\top A),
$$

其中 $A$ 是已知矩阵，$R$ 是待求的正交矩阵。

---

## 7. SVD 求解正交 Procrustes 问题

现在证明：

若

$$
A=U\Sigma V^\top
$$

是 $A$ 的奇异值分解，则

$$
R^*=UV^\top
$$

是

$$
\max_{R^\top R=I}
\operatorname{tr}(R^\top A)
$$

的一个最优解。

---

## 8. SVD 解法证明

### 8.1 把 SVD 代入目标函数

由 SVD：

$$
A=U\Sigma V^\top.
$$

目标函数为：

$$
\operatorname{tr}(R^\top A).
$$

代入 $A=U\Sigma V^\top$，得到：

$$
\operatorname{tr}(R^\top A)
=
\operatorname{tr}(R^\top U\Sigma V^\top).
$$

利用迹的循环性质：

$$
\operatorname{tr}(R^\top U\Sigma V^\top)
=
\operatorname{tr}(V^\top R^\top U\Sigma).
$$

注意：

$$
V^\top R^\top U
=
(U^\top R V)^\top.
$$

令：

$$
M=U^\top R V.
$$

由于 $U,R,V$ 都是正交矩阵，所以 $M$ 也是正交矩阵：

$$
M^\top M
=
(U^\top R V)^\top(U^\top R V)
=
V^\top R^\top U U^\top R V
=
V^\top R^\top R V
=
V^\top V
=
I.
$$

于是目标函数变成：

$$
\operatorname{tr}(R^\top A)
=
\operatorname{tr}(M^\top\Sigma).
$$

所以原问题等价于：

$$
\max_{M^\top M=I}
\operatorname{tr}(M^\top\Sigma).
$$

---

### 8.2 展开 $\operatorname{tr}(M^\top\Sigma)$

因为 $\Sigma$ 是对角矩阵：

$$
\Sigma=
\operatorname{diag}(\sigma_1,\sigma_2,\dots,\sigma_D),
$$

其中：

$$
\sigma_r\ge 0.
$$

于是：

$$
\operatorname{tr}(M^\top\Sigma)
=
\sum_{r=1}^D
\sigma_r M_{rr}.
$$

因为 $M$ 是正交矩阵，所以每个对角元素满足：

$$
M_{rr}\le 1.
$$

因此：

$$
\sum_{r=1}^D
\sigma_r M_{rr}
\le
\sum_{r=1}^D
\sigma_r.
$$

当 $M=I$ 时，所有对角元素都等于 $1$，于是取到上界：

$$
\operatorname{tr}(M^\top\Sigma)
=
\sum_{r=1}^D
\sigma_r.
$$

所以最大值为：

$$
\sum_{r=1}^D
\sigma_r.
$$

这也就是 $A$ 的核范数：

$$
\|A\|_*
=
\sum_{r=1}^D
\sigma_r.
$$

---

### 8.3 反推出最优 $R$

因为：

$$
M=U^\top R V.
$$

最大值在：

$$
M=I
$$

时取得。

所以：

$$
U^\top R V=I.
$$

两边左乘 $U$，右乘 $V^\top$，得到：

$$
R=UV^\top.
$$

因此：

$$
R^*=UV^\top.
$$

这就是 SVD 求解正交 Procrustes 问题的核心证明。

---

## 9. 放回 HiWA 中的结论

在 HiWA 中，固定 $Q_{ij}$ 后，要解：

$$
\min_{R^\top R=I}
\sum_{k,l}
Q_{ij}(k,l)
\|RX_i(k)-Y_j(l)\|_2^2.
$$

展开后等价于：

$$
\max_{R^\top R=I}
\operatorname{tr}
\left(
R^\top
Y_jQ_{ij}^\top X_i^\top
\right).
$$

令：

$$
A_{ij}
=
Y_jQ_{ij}^\top X_i^\top.
$$

对 $A_{ij}$ 做 SVD：

$$
A_{ij}=U\Sigma V^\top.
$$

则：

$$
R_{ij}=UV^\top.
$$

这就是论文中 `STIEFELALIGNMENT` 的数学来源。

---

## 10. 为什么论文算法中还会出现 $P_{ij}$ 和 ADMM 项

上面是最干净的 Procrustes 推导。但论文实际算法中，局部更新 $R_{ij}$ 时，输入 SVD 的矩阵通常不是单纯的：

$$
Y_jQ_{ij}^\top X_i^\top.
$$

而是类似：

$$
2P_{ij}Y_jQ_{ij}^\top X_i^\top
+
\mu(R-\Lambda_{ij}).
$$

这来自完整的 HiWA-ADMM 优化。
因为不同族的旋转是要求一致的：
- 这一对簇自己的局部数据对齐；
- 这一对簇在整体匹配中的重要性；
- 所有局部旋转最后必须和同一个全局旋转 $R$ 达成一致。

### 10.1 $P_{ij}$ 的作用

完整目标中有权重：

$$
\sum_{i,j}
P_{ij}C_{ij}(R,Q_{ij}).
$$

所以 $P_{ij}$ 越大，说明当前算法越认为：

> 源簇 $X_i$ 和目标簇 $Y_j$ 可能对应。

因此这对簇对更新 $R$ 的影响也越大。

### 10.2 $\mu(R-\Lambda_{ij})$ 的作用

HiWA 为了并行优化，引入局部变量 $R_{ij}$，但最终又要求所有局部 $R_{ij}$ 与全局 $R$ 一致。

也就是说：

$$
R_{ij}=R.
$$

ADMM 使用惩罚项和乘子项来推动局部变量与全局变量达成共识。

因此局部 $R_{ij}$ 的更新同时受到两股力量影响：

```text
数据证据：Y_j Q_ij^T X_i^T
簇级权重：P_ij
全局共识拉力：μ(R - Λ_ij)
```

所以算法中的 SVD 更新仍然是 Procrustes 型问题，只是矩阵 $A$ 被替换成了包含数据证据和共识项的矩阵。

---

## 11. 为什么不能直接死记 $R=UV^\top$

需要注意，不同教材中 Procrustes 问题的写法可能不同。

如果目标是：

$$
\max_{R^\top R=I}
\operatorname{tr}(R^\top A)
$$

且：

$$
A=U\Sigma V^\top,
$$

那么：

$$
R=UV^\top.
$$

但如果目标函数写成：

$$
\max_{R^\top R=I}
\operatorname{tr}(RA)
$$

或者 $A$ 的定义方向反过来，那么结果可能变成：

$$
R=VU^\top.
$$

所以不要死背，要看清楚目标函数中 $R$ 的位置，以及 $A$ 是如何定义的。

在本节 HiWA 推导中，我们得到的是：

$$
\max_{R^\top R=I}
\operatorname{tr}
\left(
R^\top A_{ij}
\right),
$$

所以解是：

$$
R_{ij}=UV^\top.
$$

---

## 12. 直观理解

矩阵

$$
A_{ij}=Y_jQ_{ij}^\top X_i^\top
$$

记录了：

> 在当前软匹配 $Q_{ij}$ 下，目标簇 $Y_j$ 和源簇 $X_i$ 的加权对应结构。

SVD：

$$
A_{ij}=U\Sigma V^\top
$$

可以理解为提取这两个簇之间最重要的方向对应。

然后：

$$
R_{ij}=UV^\top
$$

表示：

> 把源簇的主要方向旋转到目标簇的主要方向上。

因此，在 HiWA 的一轮迭代中：

```text
Q_ij 给出点级软对应
↓
A_ij = Y_j Q_ij^T X_i^T 汇总软对应关系
↓
SVD(A_ij) 提取主要方向
↓
R_ij = U V^T 得到最佳局部正交对齐
```

---

## 13. 本节核心公式链

这一节最重要的公式链如下：

$$
C_{ij}(R,Q_{ij})
=
\frac{1}{D}
\sum_{k,l}
Q_{ij}(k,l)
\|RX_i(k)-Y_j(l)\|_2^2
$$

固定 $Q_{ij}$，优化 $R$：

$$
\min_{R^\top R=I}
\sum_{k,l}
Q_{ij}(k,l)
\|RX_i(k)-Y_j(l)\|_2^2
$$

展开平方项：

$$
\|Rx_k-y_l\|_2^2
=
\|x_k\|_2^2
+
\|y_l\|_2^2
-
2y_l^\top R x_k
$$

去掉与 $R$ 无关的常数项：

$$
\max_{R^\top R=I}
\sum_{k,l}
Q_{ij}(k,l)
y_l^\top R x_k
$$

写成迹形式：

$$
\max_{R^\top R=I}
\operatorname{tr}
\left(
R^\top Y_jQ_{ij}^\top X_i^\top
\right)
$$

定义：

$$
A_{ij}
=
Y_jQ_{ij}^\top X_i^\top
$$

得到标准 Procrustes 问题：

$$
\max_{R^\top R=I}
\operatorname{tr}(R^\top A_{ij})
$$

SVD：

$$
A_{ij}=U\Sigma V^\top
$$

最优解：

$$
R_{ij}=UV^\top
$$

---

## 14. 本节一句话总结

> 在 HiWA 中，固定点级软运输计划 $Q_{ij}$ 后，更新正交矩阵 $R_{ij}$ 的问题可以从加权平方距离最小化转化为 $\max_{R^\top R=I}\operatorname{tr}(R^\top A_{ij})$ 的正交 Procrustes 问题，其中 $A_{ij}=Y_jQ_{ij}^\top X_i^\top$。对 $A_{ij}$ 做 SVD 得到 $A_{ij}=U\Sigma V^\top$ 后，最优正交变换为 $R_{ij}=UV^\top$。

---

## 15. 自检问题

你可以用下面几个问题检查自己是否掌握本节内容：

1. 为什么 $R^\top R=I$ 可以让 $\|Rx_k-y_l\|^2$ 中的 $\|Rx_k\|^2$ 变成 $\|x_k\|^2$？正交性
2. 为什么最小化平方距离等价于最大化 $\sum_{k,l}Q_{kl}y_l^\top R x_k$？公式推导
3. 为什么 $\sum_{k,l}Q_{kl}y_l^\top R x_k$ 可以写成 $\operatorname{tr}(R^\top YQ^\top X^\top)$？frobenius内积的矩阵表达式
4. 为什么 $\max_{R^\top R=I}\operatorname{tr}(R^\top A)$ 的解是 $R=UV^\top$？奇异值分解
5. 在 HiWA 中，$A_{ij}=Y_jQ_{ij}^\top X_i^\top$ 的直观意义是什么？对应于$Y_{j} X_{i}$对应的结构
6. 为什么论文实际算法中 SVD 的输入还会包含 $P_{ij}$ 和 ADMM 共识项？

---

## 16. 下一步学习方向

下一节应该学习：

# Sinkhorn 更新 $Q_{ij}$ 和 $P$

也就是：

```text
固定 R_ij
↓
如何更新 Q_ij
↓
固定所有 C_ij
↓
如何更新 P
↓
为什么两者都可以用 Sinkhorn
```

这会把 HiWA 的另一个核心算法模块补上。
