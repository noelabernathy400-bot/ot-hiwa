---
title: 论文中的ADMM项解释
tags:
  - optimal-transport
  - HiWA
  - ADMM
  - Procrustes
  - paper-reading
created: 2026-06-28
status: Obsidian-compatible
---

# 论文中的 ADMM 项解释

## 0. 本节在论文主线中的位置

前面我们已经学过：

```text
经验测度
↓
Wasserstein 距离
↓
运输计划 Q_ij
↓
局部代价 C_ij(R,Q_ij)
↓
固定 Q_ij 后，R 的更新变成正交 Procrustes 问题
```

标准的局部 Procrustes 问题是：

$$
\max_{R^\top R=I}
\operatorname{tr}
\left(
R^\top Y_jQ_{ij}^\top X_i^\top
\right).
$$

如果记：

$$
A_{ij}=Y_jQ_{ij}^\top X_i^\top,
$$

那么就是：

$$
\max_{R^\top R=I}\operatorname{tr}(R^\top A_{ij}).
$$

对 $A_{ij}$ 做 SVD：

$$
A_{ij}=U\Sigma V^\top,
$$

最优解为：

$$
R=UV^\top.
$$

但是论文 Algorithm 1 中并不是直接把

$$
Y_jQ_{ij}^\top X_i^\top
$$

送进 `STIEFELALIGNMENT`，而是把下面这个矩阵送进去：

$$
2P_{ij}Y_jQ_{ij}^\top X_i^\top
+
\mu(R-\Lambda_{ij}).
$$

本节要解释：

1. 这个式子从哪里来；
2. 为什么它和标准 Procrustes 不矛盾；
3. $2P_{ij}$、$\mu$、$\Lambda_{ij}$ 各自是什么意思；
4. 为什么论文需要用 ADMM 做这种处理。

---

## 1. 为什么标准 Procrustes 不够？

如果只看一对簇 $(X_i,Y_j)$，并且固定运输计划 $Q_{ij}$，我们可以解：

$$
\min_{R^\top R=I}
\sum_{k,l}
Q_{ij}(k,l)
\|RX_i(k)-Y_j(l)\|_2^2.
$$

这个问题只关心：

> 这一对簇自己应该如何对齐。

但是 HiWA 的目标不是给每一对簇都学一个完全独立的旋转。论文真正想要的是：

> 整个源数据集 $X$ 通过同一个全局正交变换 $R$ 对齐到目标数据集 $Y$。

也就是说，不同簇对不能各转各的。所有局部对齐结果最后要服从同一个全局 $R$。

如果每个簇对都独立学一个 $R_{ij}$，就会变成：

```text
X_1 对 Y_1 用一个旋转
X_2 对 Y_2 用另一个旋转
X_3 对 Y_3 又用另一个旋转
```

这不是全局分布对齐，而是很多局部配准问题的拼凑。

所以论文要同时做到两件事：

1. 每一对簇可以根据自己的局部数据给出一个旋转意见；
2. 所有局部旋转最终必须和同一个全局旋转达成一致。

这就是 ADMM 项出现的根本原因。

---

## 2. 原始 HiWA 优化问题

论文的核心显式目标可以理解为：

$$
\min_{P,R,\{Q_{ij}\}}
\sum_{i,j}
P_{ij}C_{ij}(R,Q_{ij}),
$$

其中：

$$
C_{ij}(R,Q_{ij})
=
\frac{1}{D}
\sum_{k,l}
Q_{ij}(k,l)
\|RX_i(k)-Y_j(l)\|_2^2.
$$

这里：

- $P_{ij}$ 是簇级对应强度；
- $Q_{ij}$ 是点级运输计划；
- $R$ 是全局正交变换；
- $D$ 是数据维度。

这个目标中，所有簇对共享同一个 $R$。这在数学上很自然，但在算法上不方便并行求解，因为每一个 $(i,j)$ 都通过同一个 $R$ 耦合在一起。

---

## 3. ADMM 的变量拆分思想

为了让每个簇对可以并行更新，论文引入局部变量：

$$
R_{ij}.
$$

然后要求每个局部旋转都和全局旋转一致：

$$
R_{ij}=R.
$$

于是问题可以改写成：

$$
\min_{P,R,\{R_{ij},Q_{ij}\}}
\sum_{i,j}
P_{ij}C_{ij}(R_{ij},Q_{ij})
$$

并加上约束：

$$
R_{ij}=R,\quad \forall i,j.
$$

这一步的直觉是：

```text
原来：
所有簇对共用一个 R，因此更新困难。

现在：
每对簇先更新自己的 R_ij，
然后通过约束 R_ij = R 保持全局一致。
```

这就是 ADMM 适合使用的结构：局部变量可以分开更新，但它们受到共识约束。

---

## 4. 增广拉格朗日项从哪里来？

对于约束：

$$
R_{ij}=R,
$$

ADMM 不会简单地强制它立刻成立，而是把它放进一个惩罚项中。

采用 scaled dual variable 的写法，可以得到类似下面的局部目标：

$$
\min_{R_{ij}^\top R_{ij}=I}
P_{ij}C_{ij}(R_{ij},Q_{ij})
+
\frac{\mu}{2D}
\|R_{ij}-R+\Lambda_{ij}\|_F^2.
$$

这里：

- $R_{ij}$ 是当前要更新的局部旋转；
- $R$ 是当前全局旋转；
- $\Lambda_{ij}$ 是 scaled dual variable，也可以理解为记录过去偏差的乘子变量；
- $\mu>0$ 是 ADMM penalty parameter，控制局部变量向全局一致靠拢的强度；
- $\|\cdot\|_F$ 是 Frobenius 范数。

这个式子由两部分组成：

$$
P_{ij}C_{ij}(R_{ij},Q_{ij})
$$

是数据对齐项；

$$
\frac{\mu}{2D}
\|R_{ij}-R+\Lambda_{ij}\|_F^2
$$

是全局共识惩罚项。

---

## 5. 第一部分：数据项如何产生 $2P_{ij}Y_jQ_{ij}^\top X_i^\top$？

先看数据项：

$$
P_{ij}C_{ij}(R_{ij},Q_{ij})
=
\frac{P_{ij}}{D}
\sum_{k,l}
Q_{ij}(k,l)
\|R_{ij}X_i(k)-Y_j(l)\|_2^2.
$$

为了简化符号，记：

$$
x_k=X_i(k),\quad y_l=Y_j(l),\quad Q_{kl}=Q_{ij}(k,l).
$$

平方项展开：

$$
\|R_{ij}x_k-y_l\|_2^2
=
(R_{ij}x_k-y_l)^\top(R_{ij}x_k-y_l).
$$

继续展开：

$$
\|R_{ij}x_k-y_l\|_2^2
=
x_k^\top R_{ij}^\top R_{ij}x_k
-
2y_l^\top R_{ij}x_k
+
y_l^\top y_l.
$$

由于 $R_{ij}$ 是正交矩阵：

$$
R_{ij}^\top R_{ij}=I,
$$

所以：

$$
x_k^\top R_{ij}^\top R_{ij}x_k
=
x_k^\top x_k
=
\|x_k\|_2^2.
$$

因此：

$$
\|R_{ij}x_k-y_l\|_2^2
=
\|x_k\|_2^2
+
\|y_l\|_2^2
-
2y_l^\top R_{ij}x_k.
$$

前两项不依赖 $R_{ij}$，只有第三项依赖 $R_{ij}$。

所以数据项中真正影响 $R_{ij}$ 优化的是：

$$
-\frac{2P_{ij}}{D}
\sum_{k,l}
Q_{kl}y_l^\top R_{ij}x_k.
$$

把双重求和写成迹形式：

$$
\sum_{k,l}
Q_{kl}y_l^\top R_{ij}x_k
=
\operatorname{tr}
\left(
R_{ij}^\top Y_jQ_{ij}^\top X_i^\top
\right).
$$

因此数据项等价于：

$$
-\frac{2P_{ij}}{D}
\operatorname{tr}
\left(
R_{ij}^\top Y_jQ_{ij}^\top X_i^\top
\right)
+
\text{const}.
$$

最小化这个式子，等价于最大化：

$$
\operatorname{tr}
\left[
R_{ij}^\top
\left(
2P_{ij}Y_jQ_{ij}^\top X_i^\top
\right)
\right].
$$

所以数据项贡献出的 Procrustes 输入矩阵是：

$$
2P_{ij}Y_jQ_{ij}^\top X_i^\top.
$$

这解释了公式中的第一部分。

---

## 6. 为什么有系数 $2$？

系数 $2$ 来自平方距离展开中的交叉项：

$$
-2y_l^\top R_{ij}x_k.
$$

也就是说：

$$
\|R_{ij}x_k-y_l\|^2
=
\text{常数项}
-
2y_l^\top R_{ij}x_k.
$$

因此在转化为最大化迹问题时，会出现：

$$
2Y_jQ_{ij}^\top X_i^\top.
$$

如果只看标准 Procrustes 问题，有时候会把这个 $2$ 省略，因为对目标矩阵乘一个正数不会改变 SVD 得到的最优正交方向。

但是在 ADMM 中还要和另一项

$$
\mu(R-\Lambda_{ij})
$$

相加，所以这个 $2$ 会影响数据项和共识项之间的相对权重。因此论文保留了这个系数。

---

## 7. 为什么有 $P_{ij}$？

$P_{ij}$ 是簇级对应强度。

如果 $P_{ij}$ 很大，说明当前认为：

> 源簇 $X_i$ 和目标簇 $Y_j$ 很可能对应。

那么这对簇提供的局部旋转证据就应该更重要。

如果 $P_{ij}$ 很小，说明当前认为：

> $X_i$ 和 $Y_j$ 不太可能对应。

那么这对簇就不应该强烈影响局部旋转更新。

所以数据项被乘上：

$$
P_{ij}.
$$

直观地说：

```text
P_ij 大：
这对簇更可信，数据项影响更强。

P_ij 小：
这对簇不太可信，数据项影响更弱。
```

极端情况下，如果：

$$
P_{ij}=0,
$$

则数据项变成：

$$
2P_{ij}Y_jQ_{ij}^\top X_i^\top=0.
$$

这时局部 $R_{ij}$ 主要由全局共识项决定。

---

## 8. 第二部分：ADMM 共识项如何产生 $\mu(R-\Lambda_{ij})$？

现在看共识惩罚项：

$$
\frac{\mu}{2D}
\|R_{ij}-R+\Lambda_{ij}\|_F^2.
$$

展开 Frobenius 范数：

$$
\|R_{ij}-R+\Lambda_{ij}\|_F^2
=
\operatorname{tr}
\left[
(R_{ij}-R+\Lambda_{ij})^\top
(R_{ij}-R+\Lambda_{ij})
\right].
$$

只保留与 $R_{ij}$ 有关的项。

令：

$$
B=R-\Lambda_{ij}.
$$

则：

$$
R_{ij}-R+\Lambda_{ij}
=
R_{ij}-B.
$$

所以：

$$
\|R_{ij}-B\|_F^2
=
\operatorname{tr}
\left[
(R_{ij}-B)^\top(R_{ij}-B)
\right].
$$

展开：

$$
\|R_{ij}-B\|_F^2
=
\operatorname{tr}(R_{ij}^\top R_{ij})
-
2\operatorname{tr}(R_{ij}^\top B)
+
\operatorname{tr}(B^\top B).
$$

由于 $R_{ij}^\top R_{ij}=I$，所以：

$$
\operatorname{tr}(R_{ij}^\top R_{ij})
=
\operatorname{tr}(I)
=
D.
$$

这是常数。

同时：

$$
\operatorname{tr}(B^\top B)
$$

也不依赖 $R_{ij}$，也是常数。

所以真正影响 $R_{ij}$ 的是：

$$
-2\operatorname{tr}(R_{ij}^\top B).
$$

代回 $B=R-\Lambda_{ij}$，得到：

$$
-2\operatorname{tr}
\left[
R_{ij}^\top(R-\Lambda_{ij})
\right].
$$

乘上系数：

$$
\frac{\mu}{2D},
$$

得到：

$$
-\frac{\mu}{D}
\operatorname{tr}
\left[
R_{ij}^\top(R-\Lambda_{ij})
\right].
$$

因此，最小化共识惩罚项等价于最大化：

$$
\operatorname{tr}
\left[
R_{ij}^\top \mu(R-\Lambda_{ij})
\right].
$$

所以共识项贡献出的 Procrustes 输入矩阵是：

$$
\mu(R-\Lambda_{ij}).
$$

---

## 9. 两部分合并

局部 $R_{ij}$ 更新目标是：

$$
\min_{R_{ij}^\top R_{ij}=I}
P_{ij}C_{ij}(R_{ij},Q_{ij})
+
\frac{\mu}{2D}
\|R_{ij}-R+\Lambda_{ij}\|_F^2.
$$

根据上面的推导，它等价于：

$$
\max_{R_{ij}^\top R_{ij}=I}
\operatorname{tr}
\left[
R_{ij}^\top
\left(
2P_{ij}Y_jQ_{ij}^\top X_i^\top
+
\mu(R-\Lambda_{ij})
\right)
\right].
$$

令：

$$
A_{ij}^{\mathrm{ADMM}}
=
2P_{ij}Y_jQ_{ij}^\top X_i^\top
+
\mu(R-\Lambda_{ij}).
$$

于是：

$$
R_{ij}
=
\arg\max_{R_{ij}^\top R_{ij}=I}
\operatorname{tr}
\left[
R_{ij}^\top A_{ij}^{\mathrm{ADMM}}
\right].
$$

这又回到了标准正交 Procrustes 形式。

所以可以做 SVD：

$$
A_{ij}^{\mathrm{ADMM}}
=
U\Sigma V^\top.
$$

然后更新：

$$
R_{ij}=UV^\top.
$$

---

## 10. 它和标准 Procrustes 的关系

标准 Procrustes 是：

$$
\max_{R^\top R=I}
\operatorname{tr}
\left(
R^\top Y_jQ_{ij}^\top X_i^\top
\right).
$$

论文中的 ADMM 局部 Procrustes 是：

$$
\max_{R_{ij}^\top R_{ij}=I}
\operatorname{tr}
\left[
R_{ij}^\top
\left(
2P_{ij}Y_jQ_{ij}^\top X_i^\top
+
\mu(R-\Lambda_{ij})
\right)
\right].
$$

两者的结构完全一样，都是：

$$
\max_{R^\top R=I}\operatorname{tr}(R^\top A).
$$

区别只是输入矩阵 $A$ 不同。

标准问题中：

$$
A=Y_jQ_{ij}^\top X_i^\top.
$$

论文 ADMM 中：

$$
A=
2P_{ij}Y_jQ_{ij}^\top X_i^\top
+
\mu(R-\Lambda_{ij}).
$$

所以论文并没有放弃 Procrustes，而是把 Procrustes 嵌入到了 ADMM 的局部更新中。

---

## 11. 每一项的直观意义

这个公式可以分成两部分：

$$
\underbrace{
2P_{ij}Y_jQ_{ij}^\top X_i^\top
}_{\text{局部数据证据}}
+
\underbrace{
\mu(R-\Lambda_{ij})
}_{\text{全局共识拉力}}.
$$

### 11.1 局部数据证据

$$
2P_{ij}Y_jQ_{ij}^\top X_i^\top
$$

表示：

> 当前这对簇 $(X_i,Y_j)$ 根据运输计划 $Q_{ij}$，认为局部旋转 $R_{ij}$ 应该朝哪个方向取值。

其中：

- $Y_jQ_{ij}^\top X_i^\top$ 是由软对应 $Q_{ij}$ 汇总出来的交叉结构；
- $P_{ij}$ 是簇级权重；
- $2$ 来自平方距离展开。

---

### 11.2 全局共识拉力

$$
\mu(R-\Lambda_{ij})
$$

表示：

> 当前全局旋转 $R$ 对局部旋转 $R_{ij}$ 的约束，同时用 $\Lambda_{ij}$ 修正过去的局部-全局偏差。

其中：

- $\mu$ 越大，$R_{ij}$ 越被强迫靠近全局 $R$；
- $\mu$ 越小，$R_{ij}$ 越能听从自己的局部数据；
- $\Lambda_{ij}$ 记录过去迭代中 $R_{ij}$ 与 $R$ 的不一致。

---

## 12. $\Lambda_{ij}$ 的作用到底是什么？

如果只有惩罚项：

$$
\frac{\mu}{2D}
\|R_{ij}-R\|_F^2,
$$

那么算法只是简单地让局部变量靠近全局变量。

但 ADMM 更精细。它引入 $\Lambda_{ij}$，使惩罚项变成：

$$
\frac{\mu}{2D}
\|R_{ij}-R+\Lambda_{ij}\|_F^2.
$$

$\Lambda_{ij}$ 的作用可以理解为：

> 记录这一对簇的局部旋转和全局旋转之间过去的偏差，并在后续更新中进行修正。

如果某个局部 $R_{ij}$ 总是偏离全局 $R$，那么 $\Lambda_{ij}$ 会积累这种偏差，使后面的更新不只是简单靠近 $R$，而是带有历史纠偏。

ADMM 的乘子更新通常具有下面的形式：

$$
\Lambda_{ij}
\leftarrow
\Lambda_{ij}+R_{ij}-R.
$$

这个更新表达的是：

```text
如果 R_ij 比全局 R 偏得多，
Lambda_ij 就会记录这个偏差；
下一次更新 R_ij 时，
这个偏差会进入 R - Lambda_ij，
从而修正局部变量。
```

注意，不同论文或代码可能对乘子符号采用不同约定，所以有时你会看到 $R+\Lambda_{ij}$ 或 $R-\Lambda_{ij}$。关键不在符号本身，而在于：

> $\Lambda_{ij}$ 是用来积累和修正局部-全局不一致的对偶变量。

---

## 13. $\mu$ 的作用是什么？

$\mu$ 是 ADMM penalty parameter。

它控制两个目标之间的平衡：

```text
局部数据对齐
vs
全局共识一致
```

如果 $\mu$ 很小：

$$
\mu(R-\Lambda_{ij})
$$

这一项影响弱，局部 $R_{ij}$ 更听从数据项：

$$
2P_{ij}Y_jQ_{ij}^\top X_i^\top.
$$

如果 $\mu$ 很大：

$$
\mu(R-\Lambda_{ij})
$$

这一项影响强，局部 $R_{ij}$ 会更接近全局 $R$。

所以：

- $\mu$ 太小：局部变量可能各自为政，全局一致性差；
- $\mu$ 太大：局部变量太早被强行拉到全局，可能忽视局部数据结构；
- 合适的 $\mu$：兼顾局部证据和全局一致。

---

## 14. 极端情况帮助理解

### 14.1 如果没有 ADMM 共识项

令：

$$
\mu=0.
$$

则输入矩阵变成：

$$
2P_{ij}Y_jQ_{ij}^\top X_i^\top.
$$

这时每对簇只根据自己的数据更新 $R_{ij}$。

如果 $P_{ij}>0$，这就接近标准 Procrustes。

缺点是：不同簇对可能得到不同旋转，无法形成统一全局对齐。

---

### 14.2 如果这对簇不重要

令：

$$
P_{ij}=0.
$$

则数据项消失：

$$
2P_{ij}Y_jQ_{ij}^\top X_i^\top=0.
$$

输入矩阵变成：

$$
\mu(R-\Lambda_{ij}).
$$

这表示：

> 这对簇不提供有效数据证据，局部 $R_{ij}$ 主要服从全局共识。

---

### 14.3 如果 $\mu$ 很大

当：

$$
\mu\gg P_{ij},
$$

局部更新主要由：

$$
\mu(R-\Lambda_{ij})
$$

决定。

这表示：

> 局部旋转被强烈拉向全局旋转。

算法更稳定，但可能牺牲局部数据适配性。

---

### 14.4 如果 $P_{ij}$ 很大且 $\mu$ 适中

此时数据项和共识项都有效：

$$
2P_{ij}Y_jQ_{ij}^\top X_i^\top
+
\mu(R-\Lambda_{ij}).
$$

这表示：

> 这对簇既是可信匹配，又不能脱离全局旋转。

这是 HiWA 想要的状态。

---

## 15. 为什么这样处理有利于并行？

引入 $R_{ij}$ 后，每一对簇的局部更新只依赖于：

- 当前 $P_{ij}$；
- 当前 $Q_{ij}$；
- 当前全局 $R$；
- 当前 $\Lambda_{ij}$；
- 自己的 $X_i,Y_j$。

所以不同 $(i,j)$ 的 $R_{ij}$ 可以并行更新。

也就是说：

```text
for each pair (i,j):
    update R_ij independently
```

这正是论文说的 distributed ADMM 思路。

如果不拆成 $R_{ij}$，所有簇对共享一个 $R$，更新就会互相耦合，不容易做成分布式或并行块更新。

---

## 16. 全局 $R$ 如何更新？

局部变量更新后，还要更新全局 $R$。

根据共识项：

$$
\sum_{i,j}
\|R_{ij}-R+\Lambda_{ij}\|_F^2,
$$

固定 $R_{ij}$ 和 $\Lambda_{ij}$，优化全局 $R$ 等价于：

$$
\min_{R^\top R=I}
\sum_{i,j}
\|R-(R_{ij}+\Lambda_{ij})\|_F^2.
$$

展开后等价于：

$$
\max_{R^\top R=I}
\operatorname{tr}
\left[
R^\top
\sum_{i,j}
(R_{ij}+\Lambda_{ij})
\right].
$$

因此全局 $R$ 也可以通过 `STIEFELALIGNMENT` 更新：

$$
R
=
\operatorname{STIEFELALIGNMENT}
\left(
\sum_{i,j}(R_{ij}+\Lambda_{ij})
\right).
$$

直观意思是：

> 全局 $R$ 是所有局部旋转意见加上对偶修正后的共识结果。

在代码实现中，求和有时会写成平均：

$$
\frac{1}{S^2}
\sum_{i,j}(R_{ij}+\Lambda_{ij}).
$$

由于乘以正数不会改变 Procrustes 的最优方向，所以求和和平均在方向上等价。

---

## 17. 局部更新、全局更新、乘子更新之间的循环

ADMM 的核心循环可以理解为：

```text
第一步：局部更新
每一对簇根据自己的数据项和全局共识项更新 R_ij。

第二步：全局更新
把所有 R_ij 的意见汇总成新的全局 R。

第三步：乘子更新
更新 Lambda_ij，记录 R_ij 和 R 之间的不一致。
```

用公式表示：

### 局部更新

$$
R_{ij}
=
\operatorname{STIEFELALIGNMENT}
\left(
2P_{ij}Y_jQ_{ij}^\top X_i^\top
+
\mu(R-\Lambda_{ij})
\right).
$$

### 全局更新

$$
R
=
\operatorname{STIEFELALIGNMENT}
\left(
\sum_{i,j}(R_{ij}+\Lambda_{ij})
\right).
$$

### 乘子更新

$$
\Lambda_{ij}
\leftarrow
\Lambda_{ij}+R_{ij}-R.
$$

这三个步骤的共同目标是：

> 既让每个局部旋转符合自己的簇对数据，又让所有局部旋转逐渐达成同一个全局旋转。

---

## 18. 这个公式在 HiWA 中的整体意义

公式

$$
2P_{ij}Y_jQ_{ij}^\top X_i^\top
+
\mu(R-\Lambda_{ij})
$$

不是新的距离，也不是新的最优传输定义。

它是：

> 在 ADMM 框架下，把局部数据对齐项和全局共识惩罚项合并后得到的 Procrustes 输入矩阵。

它保留了标准 Procrustes 的形式：

$$
\max_{R^\top R=I}\operatorname{tr}(R^\top A),
$$

只是把 $A$ 从单纯的局部数据矩阵变成了：

$$
A_{ij}^{\mathrm{ADMM}}
=
\text{局部数据证据}
+
\text{全局共识拉力}.
$$

所以标准 Procrustes 和论文公式的关系是：

```text
标准 Procrustes：
只根据一对点云更新 R。

HiWA ADMM：
根据一对簇的数据证据更新 R_ij，
同时考虑这对簇的重要性 P_ij，
并通过 ADMM 项让 R_ij 与全局 R 保持一致。
```

---

## 19. 一句话总结

论文中的

$$
2P_{ij}Y_jQ_{ij}^\top X_i^\top
+
\mu(R-\Lambda_{ij})
$$

是更新局部旋转 $R_{ij}$ 时送入 SVD 的矩阵。

其中：

- $Y_jQ_{ij}^\top X_i^\top$ 来自标准正交 Procrustes；
- $2$ 来自平方距离展开的交叉项；
- $P_{ij}$ 表示这对簇在外层匹配中的可信度和重要性；
- $\mu(R-\Lambda_{ij})$ 来自 ADMM 的全局共识惩罚；
- $\mu$ 控制共识强度；
- $\Lambda_{ij}$ 记录并修正局部旋转与全局旋转之间的历史偏差。

因此，这个公式和标准 Procrustes 不矛盾。它是标准 Procrustes 在 HiWA 的分布式 ADMM 求解框架中的增强版本。

---

## 20. 当前阶段你应该掌握的主线

学完本节后，你应该能把这一段论文算法讲成下面这样：

> HiWA 原本需要学习一个全局正交变换 $R$，但所有簇对共享同一个 $R$ 会导致优化耦合。论文于是用 ADMM 给每一对簇引入局部旋转 $R_{ij}$，并加上约束 $R_{ij}=R$。更新 $R_{ij}$ 时，目标由两部分组成：一部分是加权局部 Wasserstein 对齐代价 $P_{ij}C_{ij}$，它展开后给出 $2P_{ij}Y_jQ_{ij}^\top X_i^\top$；另一部分是 ADMM 共识惩罚项，它展开后给出 $\mu(R-\Lambda_{ij})$。两者相加后仍然是标准 Procrustes 形式，所以可以对这个合成矩阵做 SVD，得到 $R_{ij}=UV^\top$。

---

## 21. 自检问题

1. 为什么标准 Procrustes 中只有 $Y_jQ_{ij}^\top X_i^\top$，而论文中多了 $P_{ij}$？
2. 为什么平方距离展开会带来系数 $2$？
3. $\mu$ 太大和太小分别会带来什么影响？
4. $\Lambda_{ij}$ 为什么不是数据项，而是对偶修正项？
5. 为什么引入 $R_{ij}$ 后可以并行更新？
6. 为什么加了 ADMM 项以后，问题仍然可以用 SVD 求？

---

## 22. 下一步学习方向

下一节应该进入论文 Algorithm 1 中另一条重要更新：

$$
Q_{ij}
=
\operatorname{SINKHORN}(\cdots).
$$

也就是：

> 固定 $R_{ij}$ 后，如何更新点级运输计划 $Q_{ij}$？

这会把我们带入熵正则最优传输和 Sinkhorn 迭代。
