---
title: 涉及到的对应方法
aliases:
  - 熵正则、Sinkhorn 与 ADMM 在 HiWA 中的作用
tags:
  - HiWA
  - entropy-regularization
  - Sinkhorn
  - ADMM
  - optimal-transport
created: 2026-06-28
status: Obsidian-compatible
---

# 涉及到的对应方法

## 0. 为什么叫“涉及到的对应方法”？

我们读 HiWA 论文时，遇到了很多看起来突然出现的方法：

- 熵正则；
- Sinkhorn；
- ADMM；
- 拉格朗日乘子；
- 二次罚项；
- 缩放乘子 $\Lambda_{ij}$；
- 惩罚系数 $\mu$。

这些方法都不是论文主题本身，但它们是论文算法能够运行的关键工具。

本文档把这些方法统一解释为：

> HiWA 为了求解层级最优传输对齐问题而使用的优化工具。

主线如下：

```text
HiWA 原始目标
↓
P 和 Q 是运输计划，需要最优传输方法
↓
给 P 和 Q 加熵正则，使其可以用 Sinkhorn 求
↓
R 是全局旋转，和所有簇对耦合
↓
用 ADMM 把全局 R 拆成局部 R_ij
↓
用拉格朗日乘子和二次罚项保证 R_ij 与 R 一致
```

---

## 1. HiWA 中到底有哪些变量？

| 变量 | 类型 | 作用 |
|---|---|---|
| $P$ | 簇级运输计划 | 判断哪个源簇对应哪个目标簇 |
| $Q_{ij}$ | 点级运输计划 | 判断第 $i$ 个源簇和第 $j$ 个目标簇内部点如何对应 |
| $R$ | 全局正交变换 | 把整个源数据旋转或反射到目标数据 |
| $R_{ij}$ | 局部正交变换 | ADMM 中第 $(i,j)$ 个簇对的局部旋转 |
| $\Lambda_{ij}$ | 缩放拉格朗日乘子 | 记录 $R_{ij}$ 与 $R$ 的历史不一致 |
| $\mu$ | ADMM 惩罚系数 | 控制局部旋转服从全局旋转的强度 |
| $\varepsilon_1$ | 熵正则参数 | 控制 $P$ 的软硬程度 |
| $\varepsilon_2$ | 熵正则参数 | 控制 $Q_{ij}$ 的软硬程度 |

---

## 2. HiWA 的裸目标

忽略算法细节时，HiWA 的核心目标可以理解为：

$$
\min_{P,R,\{Q_{ij}\}}
\sum_{i,j}
P_{ij}C_{ij}(R,Q_{ij}).
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

这一项的意思是：如果第 $i$ 个源簇和第 $j$ 个目标簇通过 $Q_{ij}$ 进行点级软匹配，并用全局旋转 $R$ 对齐，那么这一对簇的对齐代价是多少？

但是这个裸问题不好直接求，因为：

1. $P$ 和 $Q_{ij}$ 是带边缘约束的运输计划；
2. $R$ 是正交矩阵，约束 $R^\top R=I$ 非凸；
3. 所有簇对共享同一个 $R$，耦合很强；
4. 如果没有正则，$P$ 和 $Q_{ij}$ 容易过早变成硬匹配，优化不稳定。

所以论文引入了熵正则和 ADMM。

---

# 第一部分：熵正则与 Sinkhorn

## 3. 裸最优传输问题

给定代价矩阵：

$$
C=(C_{kl}),
$$

我们要求运输计划：

$$
Q=(Q_{kl}),
$$

满足：

$$
Q\mathbf 1=a,
$$

$$
Q^\top\mathbf 1=b,
$$

$$
Q_{kl}\ge 0.
$$

裸 OT 问题是：

$$
\min_Q
\sum_{k,l}Q_{kl}C_{kl}.
$$

这可以写成矩阵内积：

$$
\min_Q
\langle Q,C\rangle.
$$

含义是：在所有满足质量守恒的运输计划中，找总代价最小的那个。

---

## 4. 为什么裸 OT 不够好算？

裸 OT 是线性规划。它理论清楚，但在算法中可能有几个问题：

1. 解可能很稀疏，容易形成硬匹配；
2. 对噪声和初始化比较敏感；
3. 大规模计算时成本高；
4. 不方便嵌入反复交替更新的模型中。

HiWA 中需要不断更新大量的 $Q_{ij}$ 和 $P$，所以需要快速、稳定的子程序。

这就是熵正则和 Sinkhorn 的作用。

---

## 5. 熵正则是什么？

熵正则 OT 写成：

$$
\min_Q
\langle Q,C\rangle
+
\varepsilon
\sum_{k,l}
Q_{kl}(\log Q_{kl}-1).
$$

这里：

$$
H_{\varepsilon}(Q)
=
\varepsilon
\sum_{k,l}
Q_{kl}(\log Q_{kl}-1).
$$

注意，这里的符号有时叫“负熵正则”，因为 Shannon entropy 是：

$$
\operatorname{Ent}(Q)
=
-
\sum_{k,l}Q_{kl}\log Q_{kl}.
$$

而优化中常写：

$$
\sum_{k,l}Q_{kl}(\log Q_{kl}-1).
$$

它和负熵只差一些常数和符号习惯。

---

## 6. 加熵的公式是经验公式吗？

不是单纯的经验公式。

它是一种标准的凸正则化方法，来源可以从三个角度理解。

### 6.1 角度一：负熵正则化

原问题是：

$$
\min_Q \langle Q,C\rangle.
$$

为了避免解过硬，加入一个鼓励扩散的正则项：

$$
\varepsilon\sum_{k,l}Q_{kl}\log Q_{kl}.
$$

由于 $q\log q$ 是凸函数，所以这个正则项会让问题更平滑、更容易优化。

写成：

$$
\varepsilon\sum_{k,l}Q_{kl}(\log Q_{kl}-1)
$$

是因为它的导数更干净：

$$
\frac{d}{dq}\left[q(\log q-1)\right]
=
\log q.
$$

如果写成 $q\log q$，导数是：

$$
\log q+1.
$$

所以 $q(\log q-1)$ 这种形式不是随便写的，而是为了让一阶条件更简洁。

### 6.2 角度二：最大熵思想

在所有运输计划中，如果多个方案代价差不多，我们希望不要过早选一个极端方案，而是保留更高不确定性。

熵越大，运输计划越分散；熵越小，运输计划越尖锐。

加入负熵正则相当于：在追求低运输代价的同时，也鼓励运输计划不要太尖锐。

参数 $\varepsilon$ 控制这个权衡。

### 6.3 角度三：KL 正则化

熵正则也可以理解成相对于某个参考分布的 KL 正则化。

如果参考分布是均匀的，那么：

$$
\operatorname{KL}(Q\mid \text{uniform})
$$

会包含类似：

$$
\sum_{k,l}Q_{kl}\log Q_{kl}
$$

的项。

所以熵正则不是凭经验乱加，而是有明确的凸优化和信息论来源。

---

## 7. 为什么熵正则会产生 Sinkhorn 形式？

考虑问题：

$$
\min_Q
\sum_{k,l}Q_{kl}C_{kl}
+
\varepsilon\sum_{k,l}Q_{kl}(\log Q_{kl}-1)
$$

subject to

$$
Q\mathbf 1=a,
$$

$$
Q^\top\mathbf 1=b.
$$

引入拉格朗日乘子 $\alpha_k$ 和 $\beta_l$，构造：

$$
\mathcal L(Q,\alpha,\beta)
=
\sum_{k,l}Q_{kl}C_{kl}
+
\varepsilon\sum_{k,l}Q_{kl}(\log Q_{kl}-1)
+
\sum_k\alpha_k\left(a_k-\sum_l Q_{kl}\right)
+
\sum_l\beta_l\left(b_l-\sum_k Q_{kl}\right).
$$

对 $Q_{kl}$ 求导：

$$
\frac{\partial \mathcal L}{\partial Q_{kl}}
=
C_{kl}
+
\varepsilon\log Q_{kl}
-
\alpha_k
-
\beta_l.
$$

令其为 $0$：

$$
C_{kl}
+
\varepsilon\log Q_{kl}
-
\alpha_k
-
\beta_l
=
0.
$$

所以：

$$
\log Q_{kl}
=
\frac{\alpha_k+\beta_l-C_{kl}}{\varepsilon}.
$$

指数化：

$$
Q_{kl}
=
\exp\left(\frac{\alpha_k}{\varepsilon}\right)
\exp\left(-\frac{C_{kl}}{\varepsilon}\right)
\exp\left(\frac{\beta_l}{\varepsilon}\right).
$$

令：

$$
u_k=
\exp\left(\frac{\alpha_k}{\varepsilon}\right),
$$

$$
v_l=
\exp\left(\frac{\beta_l}{\varepsilon}\right),
$$

$$
K_{kl}=
\exp\left(-\frac{C_{kl}}{\varepsilon}\right).
$$

为避免和测度符号 $\nu$ 混淆，实际记号常用 $u_k$。于是：

$$
Q_{kl}=u_kK_{kl}v_l.
$$

矩阵形式是：

$$
Q=\operatorname{diag}(u)K\operatorname{diag}(v).
$$

这就是 Sinkhorn 形式。

---

## 8. Sinkhorn 算法怎么更新？

因为 $Q$ 要满足：

$$
Q\mathbf 1=a,
$$

$$
Q^\top\mathbf 1=b.
$$

代入：

$$
Q=\operatorname{diag}(u)K\operatorname{diag}(v),
$$

得到：

$$
u\odot(Kv)=a.
$$

这里为了避免混淆，把上式改写成标准记号：

$$
u_u\odot(Kv)=a.
$$

这里符号$\odot{}$为逐元素乘法,展开就是： 
$$
			s_{r}​(Kt)_{r}=a_{r}.
$$


如果：
$$
s= \begin{bmatrix} s_1\\ s_2\\ s_3 \end{bmatrix}, \qquad Kt= \begin{bmatrix} u_1\\ u_2\\ u_3 \end{bmatrix}​​​
$$
那么：
$$
s\odot(Kt) = \begin{bmatrix} s_1u_1\\ s_2u_2\\ s_3u_3 \end{bmatrix}​​​
$$
就是对应位置相乘。
若记缩放向量为 $u$，则：

$$
u_u=u.
$$

所以：

$$
u_u=\frac{a}{Kv}.
$$

也就是常见写法：

$$
u_u^{t+1}=\frac{a}{Kv^t}.
$$

再令 $\nu_u$ 直接记作 $u$，得到：

$$
u_u^{t+1}=\frac{a}{Kv^t},
$$

$$
v^{t+1}=\frac{b}{K^\top \nu_u^{t+1}}.
$$

为了避免符号冲突，在笔记中可以把 Sinkhorn 的两个缩放向量记作 $s,t$：

$$
s^{t+1}=\frac{a}{Kt^t},
$$

$$
t^{t+1}=\frac{b}{K^\top s^{t+1}}.
$$

除法均为逐元素除法。也就是反复调整，先是行和然后是列和，一直反复下去让两个边界条件都满足！

---

## 9. HiWA 中的两个 Sinkhorn 问题

### 9.1 点级运输计划 $Q_{ij}$

固定 $R_{ij}$ 后，更新 $Q_{ij}$：

$$
Q_{ij}
=
\operatorname{Sinkhorn}(\text{点级代价矩阵}).
$$

点级代价是：

$$
C_{ij}^{\text{point}}(k,l)
=
\frac{1}{D}
\|R_{ij}X_i(k)-Y_j(l)\|_2^2.
$$

它解决的是：这一对簇内部，点和点之间怎么软匹配。

### 9.2 簇级运输计划 $P$

当所有簇对代价都算出来后，更新 $P$：

$$
P=
\operatorname{Sinkhorn}(\text{簇级代价矩阵}).
$$

它解决的是：哪个源簇应该对应哪个目标簇。

---

## 10. $\varepsilon_1$ 和 $\varepsilon_2$ 的区别

| 参数 | 作用对象 | 作用 |
|---|---|---|
| $\varepsilon_1$ | $P$ | 控制簇级对应的软硬程度 |
| $\varepsilon_2$ | $Q_{ij}$ | 控制点级对应的软硬程度 |

直观上：

```text
epsilon 越小：越接近裸 OT，匹配更硬
epsilon 越大：正则越强，匹配更软
```

---

# 第二部分：ADMM 与共识约束

## 11. 为什么还需要 ADMM？

熵正则解决的是 $P$ 和 $Q_{ij}$ 的运输计划问题。

但 $R$ 还有另一个困难：所有簇对共享同一个全局旋转 $R$。

如果直接优化，全局 $R$ 和所有 $Q_{ij}$、$P$ 耦合在一起。

论文想利用分块结构，让每个簇对可以独立更新自己的局部旋转。

所以引入局部副本：

$$
R_{ij}.
$$

并要求：

$$
R_{ij}=R.
$$

这就是 ADMM 的入口。

---

## 12. ADMM 不是模型假设，而是求解方法

这一点很重要。

熵正则可以看作对原始 OT 目标做平滑正则化。

ADMM 则主要是求解带约束问题的方法。

它不是说真实世界里存在一个额外物理能量：

$$
\|R_{ij}-R+\Lambda_{ij}\|_F^2.
$$

而是：为了求解约束 $R_{ij}=R$，算法构造了增广拉格朗日函数。

---

## 13. HiWA 中的变量分裂

原始目标：

$$
\sum_{i,j}P_{ij}C_{ij}(R,Q_{ij}).
$$

分裂后：

$$
\sum_{i,j}P_{ij}C_{ij}(R_{ij},Q_{ij})
$$

subject to

$$
R_{ij}=R.
$$

如果约束成立，两个问题等价。

分裂的好处是：

```text
每个 C_ij 可以用自己的 R_ij 处理
↓
每个簇对可以并行更新
↓
再通过 ADMM 拉回统一的 R
```

---

## 14. 增广拉格朗日项

对约束：

$$
R_{ij}-R=0,
$$

构造：

$$
\langle \Gamma_{ij},R_{ij}-R\rangle
+
\frac{\mu}{2D}\|R_{ij}-R\|_F^2.
$$

其中：

- $\Gamma_{ij}$ 是未缩放拉格朗日乘子；
- $\mu$ 是惩罚系数；
- $D$ 是维度，用于尺度归一化。

缩放后得到：

$$
\frac{\mu}{2D}
\|R_{ij}-R+\Lambda_{ij}\|_F^2
$$

忽略与当前变量无关的常数。

---

## 15. $\Lambda_{ij}$ 的配方过程

令：

$$
A=R_{ij}-R.
$$

从：

$$
\langle \Gamma_{ij},A\rangle
+
\frac{\mu}{2D}\|A\|_F^2
$$

出发。

定义缩放乘子：

$$
\Lambda_{ij}
=
\frac{D}{\mu}\Gamma_{ij}.
$$

于是：

$$
\Gamma_{ij}
=
\frac{\mu}{D}\Lambda_{ij}.
$$

代入：

$$
\frac{\mu}{D}\langle \Lambda_{ij},A\rangle
+
\frac{\mu}{2D}\|A\|_F^2.
$$

配方：

$$
\frac{\mu}{2D}
\left(
\|A\|_F^2
+
2\langle \Lambda_{ij},A\rangle
\right)
=
\frac{\mu}{2D}
\left(
\|A+\Lambda_{ij}\|_F^2
-
\|\Lambda_{ij}\|_F^2
\right).
$$

因此：

$$
\frac{\mu}{D}\langle \Lambda_{ij},A\rangle
+
\frac{\mu}{2D}\|A\|_F^2
=
\frac{\mu}{2D}
\|A+\Lambda_{ij}\|_F^2
-
\frac{\mu}{2D}
\|\Lambda_{ij}\|_F^2.
$$

在优化 $R_{ij}$ 时，第二项是常数，因此可忽略。

所以得到：

$$
\frac{\mu}{2D}
\|R_{ij}-R+\Lambda_{ij}\|_F^2.
$$

---

## 16. $\Lambda_{ij}$ 到底如何计算？

ADMM 每轮都会更新 $\Lambda_{ij}$。

缩放形式中：

$$
\Lambda_{ij}^{t+1}
=
\Lambda_{ij}^{t}
+
R_{ij}^{t+1}
-
R^{t+1}.
$$

其中：

- $R_{ij}^{t+1}$ 是本轮得到的局部旋转；
- $R^{t+1}$ 是本轮得到的全局旋转；
- $R_{ij}^{t+1}-R^{t+1}$ 是本轮局部与全局的不一致。

如果局部和全局一致：

$$
R_{ij}^{t+1}-R^{t+1}=0,
$$

则：

$$
\Lambda_{ij}^{t+1}=\Lambda_{ij}^{t}.
$$

如果不一致，则把偏差累加进去。

因此可以理解为：

> $\Lambda_{ij}$ 是一个历史偏差账本。

---

## 17. $\mu$ 的作用

$\mu$ 控制共识约束的强度。

如果 $\mu$ 很大，项：

$$
\frac{\mu}{2D}
\|R_{ij}-R+\Lambda_{ij}\|_F^2
$$

权重很大。

那么局部 $R_{ij}$ 会更强地靠近全局 $R$。

如果 $\mu$ 很小，局部 $R_{ij}$ 更自由，会更听当前簇对自己的数据项。

所以：

```text
mu 大：共识更强，局部自由度更小
mu 小：共识更弱，局部自由度更大
```

---

## 18. 更新 $R_{ij}$ 时为什么仍然是 Procrustes？

更新 $R_{ij}$ 的子问题是：

$$
\min_{R_{ij}^{\top}R_{ij}=I}
P_{ij}C_{ij}(R_{ij},Q_{ij})
+
\frac{\mu}{2D}
\|R_{ij}-R+\Lambda_{ij}\|_F^2.
$$

第一项给出局部数据证据：

$$
2P_{ij}Y_jQ_{ij}^{\top}X_i^{\top}.
$$

第二项给出全局共识拉力：

$$
\mu(R-\Lambda_{ij}).
$$

合并为：

$$
A_{ij}^{\mathrm{ADMM}}
=
2P_{ij}Y_jQ_{ij}^{\top}X_i^{\top}
+
\mu(R-\Lambda_{ij}).
$$

于是问题变成：

$$
\max_{R_{ij}^{\top}R_{ij}=I}
\operatorname{tr}\left[
R_{ij}^{\top}A_{ij}^{\mathrm{ADMM}}
\right].
$$

这仍然是标准 Orthogonal Procrustes 问题。

对：

$$
A_{ij}^{\mathrm{ADMM}}=U\Sigma V^\top
$$

做 SVD，得到：

$$
R_{ij}=UV^\top.
$$

---

# 第三部分：这些方法之间的关系

## 19. 熵正则和 ADMM 分别解决什么？

| 方法 | 解决的对象 | 解决的问题 |
|---|---|---|
| 熵正则 | $P,Q_{ij}$ | 运输计划过硬、不稳定、不好快速求 |
| Sinkhorn | $P,Q_{ij}$ | 快速求解熵正则 OT |
| ADMM | $R,R_{ij}$ | 全局旋转和局部簇对耦合太强 |
| 拉格朗日乘子 | 约束 $R_{ij}=R$ | 把一致性约束纳入优化 |
| 二次罚项 | 约束违反程度 | 稳定算法，推动局部和全局一致 |
| $\Lambda_{ij}$ | 历史偏差 | 记录并修正局部和全局的不一致 |
| $\mu$ | 共识强度 | 控制局部旋转服从全局旋转的力度 |

---

## 20. 为什么说它们都是“对应方法”？

HiWA 有两层对应：

1. 簇级对应：

$$
P_{ij}
$$

表示源簇 $X_i$ 对应目标簇 $Y_j$ 的强度。

2. 点级对应：

$$
Q_{ij}(k,l)
$$

表示源点 $X_i(k)$ 对应目标点 $Y_j(l)$ 的运输质量。

熵正则和 Sinkhorn 主要用于求这两类对应。

而 ADMM 不是直接求对应矩阵，但它保证：所有局部对应产生的局部旋转意见，最终合并成一个统一的全局变换。

所以本文档可以理解为：HiWA 中用于计算和协调对应关系的优化方法总结。

---

## 21. 最后总图

```text
HiWA 目标
min Σ_ij P_ij C_ij(R,Q_ij)
↓
P 和 Q_ij 是运输计划
↓
加熵正则
↓
使用 Sinkhorn 更新 P 和 Q_ij
↓
R 被所有簇对共享，难以并行
↓
引入局部 R_ij
↓
加约束 R_ij = R
↓
使用 ADMM 处理约束
↓
出现 Lambda_ij 和 mu
↓
更新 R_ij 时仍是 Procrustes
↓
SVD 得到 R_ij
```

---

## 22. 最容易混淆的点

### 22.1 熵正则不是 ADMM

熵正则用于 $P$ 和 $Q_{ij}$。

ADMM 用于 $R_{ij}=R$ 的一致性约束。

### 22.2 $\varepsilon$ 不是 $\mu$

$\varepsilon$ 控制运输计划的软硬程度。

$\mu$ 控制局部旋转和全局旋转的一致程度。

### 22.3 $\Lambda_{ij}$ 不是数据参数

$\Lambda_{ij}$ 不是从数据中观测来的，也不是模型假设。

它是 ADMM 算法中的乘子变量，用来记录约束违反的历史。

### 22.4 加熵不是经验乱加

熵正则是标准凸正则化，能带来：

1. 更平滑的目标；
2. 更稳定的软匹配；
3. Sinkhorn 形式的快速求解；
4. 更适合交替优化。

---

## 23. 一句话总结

> 在 HiWA 中，熵正则和 Sinkhorn 负责稳定、快速地计算簇级与点级运输对应；ADMM 负责把每个簇对的局部旋转意见协调成一个全局旋转。$\varepsilon$ 控制对应的软硬，$\mu$ 控制局部和全局的一致强度，$\Lambda_{ij}$ 则记录并修正这种一致性约束的历史偏差。
