---
title: 第七章 Algorithm 1 理解
tags:
  - optimal-transport
  - HiWA
  - algorithm
  - ADMM
  - Sinkhorn
  - Procrustes
created: 2026-06-28
status: Obsidian-compatible
---

# 第七章 Algorithm 1 理解

## 1. 本章学习目标

本章的目标是把前面已经学过的知识全部串起来，真正看懂 HiWA 论文中的 Algorithm 1。

前面已经学习过：

1. 经验测度与 Wasserstein 距离；
2. 点级运输计划 $Q_{ij}$；
3. 簇级对应矩阵 $P$；
4. 局部代价 $C_{ij}(R,Q_{ij})$；
5. 正交 Procrustes 问题与 SVD；
6. 熵正则化最优传输与 Sinkhorn；
7. 拉格朗日乘子法、ADMM、乘子 $\Lambda_{ij}$ 和参数 $\mu$。

现在 Algorithm 1 要做的事情是：

> 在一个循环中交替更新 $R_{ij}$、$Q_{ij}$、$P$、全局旋转 $\widetilde R$ 和 ADMM 乘子 $\Lambda_{ij}$，直到整体对齐结果稳定。

---

## 2. 论文算法要解决的核心问题

HiWA 的核心显式优化问题可以理解为：

$$
\min_{P,R,\{Q_{ij}\}}
\sum_{i,j}P_{ij}C_{ij}(R,Q_{ij})
+
H_{\varepsilon_1}(P)
+
\sum_{i,j}H_{\varepsilon_2}(Q_{ij}).
$$

其中：

- $P$ 是簇级对应矩阵；
- $Q_{ij}$ 是第 $(i,j)$ 对簇内部的点级运输计划；
- $R$ 是全局正交变换；
- $C_{ij}(R,Q_{ij})$ 是源簇 $X_i$ 对齐目标簇 $Y_j$ 的局部代价；
- $H_{\varepsilon_1}(P)$ 是簇级熵正则；
- $H_{\varepsilon_2}(Q_{ij})$ 是点级熵正则。

局部代价是：

$$
C_{ij}(R,Q_{ij})
=
\frac{1}{D}
\sum_{k,l}
Q_{ij}(k,l)
\lVert RX_i(k)-Y_j(l)\rVert_2^2.
$$

这里 $D$ 是数据维度。除以 $D$ 的目的是把总平方误差变成平均每维平方误差，主要用于尺度稳定。

---

## 3. 为什么 Algorithm 1 中要引入局部旋转 $R_{ij}$？

原始目标中，所有簇对共享同一个全局旋转 $R$。也就是说，每一项都是：

$$
C_{ij}(R,Q_{ij}).
$$

这样所有簇对都纠缠在同一个 $R$ 上，不方便并行计算。

为了把问题拆开，论文引入每一对簇自己的局部旋转：

$$
R_{ij}.
$$

于是局部代价变成：

$$
C_{ij}(R_{ij},Q_{ij}).
$$

但为了不改变原始问题，需要加入一致性约束：

$$
R_{ij}=\widetilde R,
\quad \forall i,j.
$$

这里 $\widetilde R$ 是全局旋转。

因此，ADMM 形式下的问题可以理解为：

$$
\min_{P,\widetilde R,\{R_{ij},Q_{ij}\}}
\sum_{i,j}
\left[
P_{ij}C_{ij}(R_{ij},Q_{ij})
+
H_{\varepsilon_2}(Q_{ij})
\right]
+
H_{\varepsilon_1}(P)
$$

subject to

$$
R_{ij}=\widetilde R,
\quad \forall i,j.
$$

这一步叫做变量分裂。它的作用是：

> 先让每对簇可以有自己的局部旋转 $R_{ij}$，从而能够并行更新；再通过 ADMM 约束让这些局部旋转逐渐达成一个全局共识 $\widetilde R$。

---

## 4. Algorithm 1 中的变量表

| 符号 | 含义 | 更新方法 |
|---|---|---|
| $P$ | 簇级对应矩阵 | 外层 Sinkhorn |
| $Q_{ij}$ | 第 $(i,j)$ 对簇内部的点级运输计划 | 内层 Sinkhorn |
| $R_{ij}$ | 第 $(i,j)$ 对簇的局部正交变换 | Procrustes / SVD |
| $\widetilde R$ | 全局正交变换 | Procrustes / SVD |
| $\Lambda_{ij}$ | ADMM 缩放乘子，记录局部与全局的偏差 | 直接迭代更新 |
| $C_{ij}$ | 第 $(i,j)$ 对簇的当前对齐代价 | 由 $R_{ij}$ 和 $Q_{ij}$ 计算 |
| $\varepsilon_1$ | $P$ 的熵正则强度 | 超参数 |
| $\varepsilon_2$ | $Q_{ij}$ 的熵正则强度 | 超参数 |
| $\mu$ | ADMM 共识惩罚强度 | 超参数 |

---

## 5. Algorithm 1 的总流程

可以先把算法抽象成下面的伪代码：

```text
初始化 P, Q_ij, R_ij, 全局旋转 R~, Lambda_ij

重复直到收敛：

    对每一对簇 (i,j) 并行：

        1. 更新局部旋转 R_ij
           使用 Procrustes / SVD

        2. 更新点级运输计划 Q_ij
           使用 Sinkhorn

        3. 计算局部代价 C_ij

    4. 更新簇级对应矩阵 P
       使用 Sinkhorn

    5. 更新全局旋转 R~
       使用 Procrustes / SVD

    6. 更新 ADMM 乘子 Lambda_ij
```

这个算法本质上是交替优化：

> 固定其他变量，更新其中一个变量；再固定新的变量，更新下一个变量；如此循环。

---

## 6. 初始化步骤

### 6.1 初始化 $P$

如果一开始不知道哪个源簇对应哪个目标簇，可以把 $P$ 初始化成均匀矩阵：

$$
P_{ij}=\frac{1}{S^2}.
$$

这里 $S$ 是簇数量。

这个初始化的意思是：

> 一开始所有簇对都被认为同样可能对应。

如果 $P$ 被看成总质量为 $1$ 的簇级运输计划，则它的行和、列和通常满足：

$$
P\mathbf{1}=\frac{1}{S}\mathbf{1},
$$

$$
P^\top\mathbf{1}=\frac{1}{S}\mathbf{1}.
$$

---

### 6.2 初始化全局旋转 $\widetilde R$

全局旋转需要满足正交约束：

$$
\widetilde R^\top\widetilde R=I.
$$

实际算法中可以随机初始化一个矩阵，然后投影到正交矩阵集合上，也可以用一个初始的 SVD 方式得到。

---

### 6.3 初始化乘子 $\Lambda_{ij}$

通常设：

$$
\Lambda_{ij}=0.
$$

直观上，$\Lambda_{ij}$ 是偏差账本。刚开始还没有局部旋转和全局旋转之间的历史偏差，所以设为零。

---

## 7. 第一步：更新局部旋转 $R_{ij}$

在每轮迭代中，对每一对簇 $(i,j)$，先固定：

- 当前 $P_{ij}$；
- 当前 $Q_{ij}$；
- 当前全局旋转 $\widetilde R$；
- 当前乘子 $\Lambda_{ij}$。

然后更新局部旋转 $R_{ij}$。

论文中的更新形式是：

$$
R_{ij}
=
\operatorname{STIEFELALIGNMENT}
\left(
2P_{ij}Y_jQ_{ij}^\top X_i^\top
+
\mu(\widetilde R-\Lambda_{ij})
\right).
$$

---

### 7.1 这一项为什么长这样？

它由两部分组成：

$$
2P_{ij}Y_jQ_{ij}^\top X_i^\top
+
\mu(\widetilde R-\Lambda_{ij}).
$$

第一部分：

$$
2P_{ij}Y_jQ_{ij}^\top X_i^\top
$$

来自局部数据对齐项：

$$
P_{ij}C_{ij}(R_{ij},Q_{ij}).
$$

它表示：

> 在当前点级运输计划 $Q_{ij}$ 下，这一对簇本身希望 $R_{ij}$ 朝哪个方向旋转。

其中：

- $Y_jQ_{ij}^\top X_i^\top$ 是软匹配下的 Procrustes 结构矩阵；
- $P_{ij}$ 表示这对簇的重要性；
- 系数 $2$ 来自平方距离展开中的交叉项。

第二部分：

$$
\mu(\widetilde R-\Lambda_{ij})
$$

来自 ADMM 共识项。它表示：

> 局部旋转 $R_{ij}$ 不能只听这一对簇自己的意见，还要靠近全局旋转 $\widetilde R$，并考虑乘子 $\Lambda_{ij}$ 记录的历史偏差。

---

### 7.2 为什么可以用 SVD？

更新 $R_{ij}$ 的问题可以写成：

$$
\max_{R_{ij}^\top R_{ij}=I}
\operatorname{tr}
\left[
R_{ij}^\top
\left(
2P_{ij}Y_jQ_{ij}^\top X_i^\top
+
\mu(\widetilde R-\Lambda_{ij})
\right)
\right].
$$

令：

$$
A_{ij}^{\mathrm{ADMM}}
=
2P_{ij}Y_jQ_{ij}^\top X_i^\top
+
\mu(\widetilde R-\Lambda_{ij}).
$$

则问题变为标准 Procrustes 形式：

$$
\max_{R_{ij}^\top R_{ij}=I}
\operatorname{tr}(R_{ij}^\top A_{ij}^{\mathrm{ADMM}}).
$$

对 $A_{ij}^{\mathrm{ADMM}}$ 做 SVD：

$$
A_{ij}^{\mathrm{ADMM}}=U\Sigma V^\top.
$$

于是：

$$
R_{ij}=UV^\top.
$$

---

## 8. 第二步：更新点级运输计划 $Q_{ij}$

更新完 $R_{ij}$ 后，固定它，更新 $Q_{ij}$。

先定义点级代价矩阵：

$$
M_{ij}(k,l)
=
\frac{1}{D}
\lVert R_{ij}X_i(k)-Y_j(l)\rVert_2^2.
$$

于是 $Q_{ij}$ 的更新问题是：

$$
\min_{Q_{ij}\in U(n_{x,i},n_{y,j})}
\langle Q_{ij},M_{ij}\rangle
+
\frac{\varepsilon_2}{P_{ij}}
\sum_{k,l}Q_{ij}(k,l)
\bigl(\log Q_{ij}(k,l)-1\bigr).
$$

---

### 8.1 为什么有效正则强度是 $\varepsilon_2/P_{ij}$？

原来的子问题中有：

$$
P_{ij}\langle Q_{ij},M_{ij}\rangle
+
\varepsilon_2H(Q_{ij}).
$$

当 $P_{ij}>0$ 时，除以 $P_{ij}$，最优解不变：

$$
\langle Q_{ij},M_{ij}\rangle
+
\frac{\varepsilon_2}{P_{ij}}H(Q_{ij}).
$$

所以有效熵强度为：

$$
\eta_{ij}=\frac{\varepsilon_2}{P_{ij}}.
$$

直觉是：

- $P_{ij}$ 大，说明这对簇可信，$Q_{ij}$ 可以更精细；
- $P_{ij}$ 小，说明这对簇不可信，$Q_{ij}$ 应该更软，避免过拟合错误簇对。

---

### 8.2 Sinkhorn 结构

定义核矩阵：

$$
K_{ij}(k,l)
=
\exp\left(-\frac{M_{ij}(k,l)}{\eta_{ij}}\right).
$$

也就是：

$$
K_{ij}(k,l)
=
\exp\left(-\frac{P_{ij}M_{ij}(k,l)}{\varepsilon_2}\right).
$$

Sinkhorn 的解具有形式：

$$
Q_{ij}=\operatorname{diag}(u)K_{ij}\operatorname{diag}(v).
$$

其中 $u,v$ 通过交替归一化求得，使 $Q_{ij}$ 满足边缘约束。

---

## 9. 第三步：计算局部代价 $C_{ij}$

得到新的 $R_{ij}$ 和 $Q_{ij}$ 后，计算：

$$
C_{ij}
=
\frac{1}{D}
\sum_{k,l}
Q_{ij}(k,l)
\lVert R_{ij}X_i(k)-Y_j(l)\rVert_2^2.
$$

也可以写成：

$$
C_{ij}=\langle Q_{ij},M_{ij}\rangle.
$$

这个值表示：

> 在当前局部旋转和点级运输计划下，第 $i$ 个源簇对齐第 $j$ 个目标簇的代价。

所有 $C_{ij}$ 组成簇级代价矩阵：

$$
C=(C_{ij}).
$$

这个矩阵会用于更新 $P$。

---

## 10. 第四步：更新簇级对应矩阵 $P$

固定所有局部代价 $C_{ij}$ 后，更新 $P$。

问题是：

$$
\min_{P\in B_S}
\sum_{i,j}P_{ij}C_{ij}
+
\varepsilon_1
\sum_{i,j}P_{ij}\bigl(\log P_{ij}-1\bigr).
$$

这也是熵正则化最优传输问题。

---

### 10.1 $P$ 的作用

$P_{ij}$ 表示源簇 $X_i$ 和目标簇 $Y_j$ 的对应强度。

如果 $C_{ij}$ 小，说明 $X_i$ 和 $Y_j$ 对齐代价低，$P_{ij}$ 倾向于变大。

如果 $C_{ij}$ 大，说明 $X_i$ 和 $Y_j$ 对齐代价高，$P_{ij}$ 倾向于变小。

---

### 10.2 外层 Sinkhorn

定义簇级核矩阵：

$$
K^P_{ij}=\exp\left(-\frac{C_{ij}}{\varepsilon_1}\right).
$$

然后：

$$
P=\operatorname{diag}(u)K^P\operatorname{diag}(v).
$$

通过 Sinkhorn 迭代调整 $u,v$，使 $P$ 满足双随机约束：

$$
P\mathbf{1}=\frac{1}{S}\mathbf{1},
$$

$$
P^\top\mathbf{1}=\frac{1}{S}\mathbf{1}.
$$

---

## 11. 第五步：更新全局旋转 $\widetilde R$

现在每对簇都有自己的局部旋转 $R_{ij}$，但最终需要一个统一的全局旋转 $\widetilde R$。

ADMM 中全局旋转的更新本质上是：

$$
\min_{\widetilde R^\top\widetilde R=I}
\sum_{i,j}
\lVert R_{ij}-\widetilde R+\Lambda_{ij}\rVert_F^2.
$$

这可以写成：

$$
\min_{\widetilde R^\top\widetilde R=I}
\sum_{i,j}
\lVert \widetilde R-(R_{ij}+\Lambda_{ij})\rVert_F^2.
$$

意思是：

> 找一个正交矩阵 $\widetilde R$，让它尽量靠近所有修正后的局部旋转 $R_{ij}+\Lambda_{ij}$。

展开后等价于：

$$
\max_{\widetilde R^\top\widetilde R=I}
\operatorname{tr}
\left[
\widetilde R^\top
\sum_{i,j}(R_{ij}+\Lambda_{ij})
\right].
$$

令：

$$
A_{\mathrm{global}}
=
\sum_{i,j}(R_{ij}+\Lambda_{ij}).
$$

对它做 SVD：

$$
A_{\mathrm{global}}=U\Sigma V^\top.
$$

得到：

$$
\widetilde R=UV^\top.
$$

这一步表示：

> 全局旋转是所有局部旋转意见的正交投影式汇总。

---

## 12. 第六步：更新乘子 $\Lambda_{ij}$

最后更新 ADMM 乘子：

$$
\Lambda_{ij}
\leftarrow
\Lambda_{ij}+R_{ij}-\widetilde R.
$$

它的意义是：

> 如果当前局部旋转 $R_{ij}$ 和全局旋转 $\widetilde R$ 不一致，就把这个偏差记录下来，下一轮继续修正。

如果某个局部旋转一直偏离全局旋转，那么 $\Lambda_{ij}$ 会逐渐积累这种偏差，使下一轮 $R_{ij}$ 的更新受到更强的纠偏作用。

所以 $\Lambda_{ij}$ 可以理解为：

```text
局部旋转与全局旋转不一致的历史账本。
```

---

## 13. Algorithm 1 的变量相互作用闭环

Algorithm 1 的核心闭环可以写成：

```text
Q_ij 决定当前点级软对应
↓
R_ij 根据 Q_ij 更新当前簇对的局部旋转
↓
R_ij 和 Q_ij 共同决定 C_ij
↓
所有 C_ij 共同决定 P
↓
P 反过来影响每个簇对的重要性
↓
所有 R_ij 汇总成全局旋转 R~
↓
Lambda_ij 记录 R_ij 和 R~ 的偏差
↓
进入下一轮迭代
```

这就是 HiWA 算法的主循环。

---

## 14. 几个容易误解的点

### 14.1 Algorithm 1 不是只求一次 $R$

它不是：

```text
先求一次 R，然后固定 R 求 Q 和 P。
```

而是：

```text
不断交替更新 R_ij、Q_ij、P、R~、Lambda_ij。
```

每一轮中，变量都会互相影响。

---

### 14.2 ADMM 不是用来替代 Procrustes 的

Procrustes / SVD 解决的是：

> 在正交约束下如何更新旋转矩阵。

ADMM 解决的是：

> 多个局部旋转 $R_{ij}$ 如何与全局旋转 $\widetilde R$ 达成一致。

所以它们不是同一层面的东西。

---

### 14.3 Sinkhorn 不是只用一次

Sinkhorn 在 Algorithm 1 中至少有两处核心作用：

1. 更新点级运输计划 $Q_{ij}$；
2. 更新簇级对应矩阵 $P$。

二者都是熵正则最优传输问题。

---

### 14.4 $P$ 不只是最后的输出

$P$ 不只是算法最后告诉我们“哪个簇对应哪个簇”。它在迭代中也参与影响 $R_{ij}$ 和 $Q_{ij}$ 的更新。

例如，更新 $R_{ij}$ 时出现：

$$
2P_{ij}Y_jQ_{ij}^\top X_i^\top.
$$

更新 $Q_{ij}$ 时出现有效正则强度：

$$
\frac{\varepsilon_2}{P_{ij}}.
$$

所以 $P$ 是整个算法中的主动变量，而不是被动结果。

---

## 15. 本章最重要的总结

Algorithm 1 的本质是：

> 通过交替优化，把簇级匹配、点级匹配和全局几何对齐一起学出来。

更具体地说：

- $Q_{ij}$ 负责点级软匹配；
- $P$ 负责簇级软匹配；
- $R_{ij}$ 负责每对簇的局部旋转；
- $\widetilde R$ 负责全局统一旋转；
- $\Lambda_{ij}$ 负责纠正局部旋转和全局旋转之间的不一致；
- Procrustes / SVD 用来更新旋转；
- Sinkhorn 用来更新运输计划；
- ADMM 用来协调局部和全局。

一句话版本：

> HiWA 的 Algorithm 1 是一个由 Procrustes、Sinkhorn 和 ADMM 共同组成的交替优化算法：Procrustes 负责旋转对齐，Sinkhorn 负责运输匹配，ADMM 负责局部旋转和全局旋转的一致性。

---

## 16. 下一步学习内容

Algorithm 1 理解之后，后面可以进入论文的理论部分：

1. Theorem 4.1：什么时候 HiWA 可以恢复正确簇匹配；
2. Theorem 4.2：几何扰动下 Wasserstein alignment 的误差控制；
3. Lemma 4.3：为什么利用层级结构会比普通 WA 更稳定。

这些理论结果会回答：

> HiWA 为什么不只是一个算法技巧，而是有一定数学保证的方法。
