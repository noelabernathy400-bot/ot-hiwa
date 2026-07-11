# TACO 数学基础：完整推导版

## 0. 阅读说明

这份笔记解释的是我们当前讨论中的 **TACO-style soft-prototype hierarchical OT** 数学框架。它不是只列公式，而是按“从问题自然推出公式”的方式，把以下部分连起来：

```text
问题背景
→ 数学对象
→ soft assignment
→ prototype
→ group-level OT
→ Sinkhorn 求解
→ instance-level OT
→ 与 HiWA 的本质区别
→ 迁移到猕猴神经元实验
```

为了避免混淆，本文会区分三类内容：

1. **论文原方法层面的结构**：TACO-style 方法的核心思想是学习 soft groups / prototypes，做 group-level OT，再用 group 对齐结果指导 instance-level alignment。
2. **根据方法自然推出的数学解释**：例如 softmax assignment 可以从熵正则化的最优分配问题推出；prototype 可以从加权最小二乘中心推出；Sinkhorn 可以从熵正则 OT 的拉格朗日条件推出。
3. **面向猕猴神经元项目的迁移建议**：例如把 TACO 的 soft-prototype 思想迁移到 HiWA 猕猴神经元实验，构造 Soft-Prototype HiWA。这是我们的研究改造方向，不等于原论文已经直接完成。

---

## 1. 问题背景：TACO 想解决什么问题

假设我们有两个空间：source space 和 target space。

在 TACO 的典型语境里，它们可能是：

```text
文本表征空间
几何表征空间
```

在你的猕猴神经元项目里，它们可以类比成：

```text
神经活动表征空间
运动行为表征空间
```

TACO-style 方法想解决的问题不是简单地说：

> 第 $i$ 个 source 样本对应第 $j$ 个 target 样本。

而是更有层次地问：

> source 中有哪些潜在 group？target 中有哪些潜在 group？哪些 source group 应该对应哪些 target group？在 group 对应关系的指导下，哪些样本应该对应哪些样本？

因此它的核心链条是：

```text
sample
→ soft group
→ group prototype
→ group-level OT
→ instance-level OT
```

这和 HiWA 有相似之处。HiWA 也是层级对齐：

```text
cluster-level alignment
→ within-cluster alignment
```

但是二者的关键差别在于：

```text
HiWA 中的 cluster 通常是一整个经验分布。
TACO-style 方法中的 group 通常被压缩成 prototype 点。
```

这个差别非常重要。后面会专门解释。

---

## 2. 数学对象：我们到底有什么数据

设 source 侧有 $n$ 个样本：

$$
x_1^s,x_2^s,\dots,x_n^s
$$

target 侧有 $m$ 个样本：

$$
x_1^t,x_2^t,\dots,x_m^t
$$

其中：

- 上标 $s$ 表示 source；
- 上标 $t$ 表示 target；
- $x_i^s$ 是 source 第 $i$ 个原始样本；
- $x_j^t$ 是 target 第 $j$ 个原始样本。

这些原始样本 $x$ 可以是文本、图结构、几何结构、神经放电向量或运动状态变量。

但是 TACO-style 方法通常不直接在原始 $x$ 上做对齐，而是先用 encoder 得到表征向量。定义 source encoder 和 target encoder：

$$
f_s,\\ f_t
$$

于是：

$$
z_i^s=f_s(x_i^s)
$$

$$
z_j^t=f_t(x_j^t)
$$

其中：

$$
z_i^s\in\mathbb{R}^d
$$

$$
z_j^t\in\mathbb{R}^d
$$

这里 $d$ 是表征空间维度。

从现在开始，真正被对齐的是两组向量：

$$
Z^s=\{z_1^s,z_2^s,\dots,z_n^s\}\subset\mathbb{R}^d
$$

$$
Z^t=\{z_1^t,z_2^t,\dots,z_m^t\}\subset\mathbb{R}^d
$$

---

## 3. 为什么不直接做样本级 OT

如果直接做样本级 OT，就要构造一个矩阵：

$$
T\in\mathbb{R}^{n\times m}
$$

其中 $T_{ij}$ 表示：

> source 第 $i$ 个样本有多少质量传输到 target 第 $j$ 个样本。

如果 $n$ 和 $m$ 都很大，那么 $T$ 会非常大。例如：

$$
n=1000,\qquad m=1000
$$

那么：

$$
T\in\mathbb{R}^{1000\times1000}
$$

这个矩阵有一百万个元素。

直接样本级 OT 有几个问题：

1. 计算量大；
2. 容易受噪声影响；
3. 缺少语义层级；
4. 没有利用样本背后可能存在 group / prototype / latent mode 的事实；
5. 如果直接对齐所有样本，可能出现局部错配。

所以 TACO-style 方法先引入 group。它不是直接问“哪个样本对应哪个样本”，而是先问“哪个 group 对应哪个 group”，然后再问“在 group 对应关系的指导下，哪个样本对应哪个样本”。

---

## 4. Soft assignment：为什么要让样本软分配到多个 group

假设 source 侧有 $K_s$ 个 group：

$$
G_1^s,G_2^s,\dots,G_{K_s}^s
$$

target 侧有 $K_t$ 个 group：

$$
G_1^t,G_2^t,\dots,G_{K_t}^t
$$

如果是 hard clustering，那么每个样本只能属于一个 group。例如：

$$
z_i^s\in G_3^s
$$

这表示 $z_i^s$ 完全属于第 3 个 group，不属于其他 group。

但是在复杂数据中，一个样本常常不是只属于一个纯粹模式。例如在神经元数据中，一个神经活动样本可能同时包含方向信息、速度信息、时间阶段信息、trial-to-trial noise 和神经元状态漂移。

所以更合理的做法是让一个样本以不同程度属于多个 group。

对 source 样本 $z_i^s$，定义 soft assignment 向量：

$$
a_i^s=(a_{i1}^s,a_{i2}^s,\dots,a_{iK_s}^s)
$$

其中：

$$
a_{ik}^s\geq0
$$

$$
\sum_{k=1}^{K_s}a_{ik}^s=1
$$

这里 $a_{ik}^s$ 表示 source 第 $i$ 个样本属于 source 第 $k$ 个 group 的程度。

例如：

$$
a_i^s=(0.70,0.20,0.10)
$$

意思是这个样本 70% 属于第 1 个 group，20% 属于第 2 个 group，10% 属于第 3 个 group。

target 侧同理：

$$
a_j^t=(a_{j1}^t,a_{j2}^t,\dots,a_{jK_t}^t)
$$

其中：

$$
a_{jl}^t\geq0
$$

$$
\sum_{l=1}^{K_t}a_{jl}^t=1
$$

---

## 5. Soft assignment 的公式为什么是 softmax

### 5.1 引入 group query

对 source 侧，假设每个 group 有一个可学习的 query 或 center：

$$
q_1^s,q_2^s,\dots,q_{K_s}^s
$$

其中：

$$
q_k^s\in\mathbb{R}^d
$$

$q_k^s$ 可以理解为第 $k$ 个 source group 的查询向量或方向向量。

对于样本 $z_i^s$，先计算它和第 $k$ 个 group query 的相似度：

$$
r_{ik}^s=\frac{\langle z_i^s,q_k^s\rangle}{\tau}
$$

其中：

- $\langle z_i^s,q_k^s\rangle$ 是内积相似度；
- $\tau>0$ 是 temperature；
- $\tau$ 越小，assignment 越尖锐；
- $\tau$ 越大，assignment 越平滑。

### 5.2 从优化问题推出 softmax

现在问题变成：已知每个 group 的得分 $r_{ik}^s$，如何得到一个非负且和为 1 的权重向量 $a_i^s$？

我们希望：

1. 权重偏向得分高的 group；
2. 权重不要过于极端；
3. 权重满足概率约束。

因此，对固定样本 $z_i^s$，考虑优化问题：

$$
\max_{a_i^s}
\sum_{k=1}^{K_s}a_{ik}^s r_{ik}^s
-
\sum_{k=1}^{K_s}a_{ik}^s\log a_{ik}^s
$$

约束为：

$$
a_{ik}^s\geq0
$$

$$
\sum_{k=1}^{K_s}a_{ik}^s=1
$$

目标函数中第一项：

$$
\sum_{k=1}^{K_s}a_{ik}^s r_{ik}^s
$$

表示加权相似度收益。它鼓励把权重分给高分 group。

第二项：

$$
-\sum_{k=1}^{K_s}a_{ik}^s\log a_{ik}^s
$$

是熵。它鼓励分配不要太硬。

所以这个优化问题的含义是：

> 在偏向高相似度 group 的同时，保持分配的平滑性。

### 5.3 拉格朗日推导

因为有约束：

$$
\sum_{k=1}^{K_s}a_{ik}^s=1
$$

引入拉格朗日乘子 $\lambda$，构造：

$$
\mathcal{J}(a_i^s,\lambda)
=
\sum_{k=1}^{K_s}a_{ik}^s r_{ik}^s
-
\sum_{k=1}^{K_s}a_{ik}^s\log a_{ik}^s
+
\lambda
\left(
\sum_{k=1}^{K_s}a_{ik}^s-1
\right)
$$

对 $a_{ik}^s$ 求偏导：

$$
\frac{\partial \mathcal{J}}{\partial a_{ik}^s}
=
r_{ik}^s-(\log a_{ik}^s+1)+\lambda
$$

令偏导为 0：

$$
r_{ik}^s-\log a_{ik}^s-1+\lambda=0
$$

移项得到：

$$
\log a_{ik}^s=r_{ik}^s+\lambda-1
$$

两边取指数：

$$
a_{ik}^s=\exp(r_{ik}^s+\lambda-1)
$$

把与 $k$ 无关的部分合并成常数 $C$：

$$
C=\exp(\lambda-1)
$$

于是：

$$
a_{ik}^s=C\exp(r_{ik}^s)
$$

使用归一化条件：

$$
\sum_{k=1}^{K_s}a_{ik}^s=1
$$

代入：

$$
\sum_{k=1}^{K_s}C\exp(r_{ik}^s)=1
$$

所以：

$$
C=
\frac{1}{
\sum_{r=1}^{K_s}\exp(r_{ir}^s)
}
$$

最终得到：

$$
a_{ik}^s=
\frac{\exp(r_{ik}^s)}
{\sum_{r=1}^{K_s}\exp(r_{ir}^s)}
$$

再代入：

$$
r_{ik}^s=\frac{\langle z_i^s,q_k^s\rangle}{\tau}
$$

得到：

$$
a_{ik}^s=
\frac{
\exp(\langle z_i^s,q_k^s\rangle/\tau)
}{
\sum_{r=1}^{K_s}
\exp(\langle z_i^s,q_r^s\rangle/\tau)
}
$$

这就是 softmax assignment。

因此 softmax 不是凭空来的，它可以理解成：

> 在 simplex 约束下，最大化“相似度收益 + 熵平滑”的最优解。

target 侧同理：

$$
a_{jl}^t=
\frac{
\exp(\langle z_j^t,q_l^t\rangle/\tau)
}{
\sum_{r=1}^{K_t}
\exp(\langle z_j^t,q_r^t\rangle/\tau)
}
$$

---

## 6. Assignment matrix

把所有 source 样本的 assignment 放到一个矩阵里：

$$
A^s\in\mathbb{R}^{n\times K_s}
$$

其中：

$$
A_{ik}^s=a_{ik}^s
$$

第 $i$ 行就是 source 第 $i$ 个样本对所有 source groups 的分配权重。

同理，target 侧：

$$
A^t\in\mathbb{R}^{m\times K_t}
$$

其中：

$$
A_{jl}^t=a_{jl}^t
$$

因为每个样本的 assignment 权重和为 1，所以：

$$
\sum_{k=1}^{K_s}A_{ik}^s=1
$$

$$
\sum_{l=1}^{K_t}A_{jl}^t=1
$$

矩阵 $A^s$ 和 $A^t$ 是连接样本层和 group 层的桥梁。

---

## 7. Prototype 是怎么来的

现在已经知道每个样本属于每个 group 的程度。

下一步是：如何用一个点表示一个 group？

source 第 $k$ 个 group 的 prototype 记为：

$$
c_k^s\in\mathbb{R}^d
$$

我们希望 $c_k^s$ 是第 $k$ 个 group 的代表中心。

既然样本 $z_i^s$ 对第 $k$ 个 group 的贡献是 $a_{ik}^s$，那么自然可以让 $c_k^s$ 最小化加权平方距离：

$$
c_k^s
=
\arg\min_{c\in\mathbb{R}^d}
\sum_{i=1}^{n}
a_{ik}^s\|z_i^s-c\|^2
$$

这个式子表示：

- 如果 $a_{ik}^s$ 大，样本 $z_i^s$ 对 prototype 影响大；
- 如果 $a_{ik}^s$ 小，样本 $z_i^s$ 对 prototype 影响小；
- 如果 $a_{ik}^s=0$，样本 $z_i^s$ 不影响第 $k$ 个 prototype。

### 7.1 推导 prototype 的闭式解

定义：

$$
F(c)=
\sum_{i=1}^{n}
a_{ik}^s\|z_i^s-c\|^2
$$

平方范数展开为：

$$
\|z_i^s-c\|^2=(z_i^s-c)^\top(z_i^s-c)
$$

对 $c$ 求梯度：

$$
\nabla_c\|z_i^s-c\|^2=2(c-z_i^s)
$$

所以：

$$
\nabla_cF(c)
=
\sum_{i=1}^{n}
a_{ik}^s2(c-z_i^s)
$$

整理：

$$
\nabla_cF(c)
=
2
\left(
\sum_{i=1}^{n}a_{ik}^s c
-
\sum_{i=1}^{n}a_{ik}^s z_i^s
\right)
$$

因为 $c$ 与 $i$ 无关：

$$
\sum_{i=1}^{n}a_{ik}^s c
=
\left(
\sum_{i=1}^{n}a_{ik}^s
\right)c
$$

令梯度为 0：

$$
2
\left(
\left(
\sum_{i=1}^{n}a_{ik}^s
\right)c
-
\sum_{i=1}^{n}a_{ik}^s z_i^s
\right)=0
$$

去掉 2：

$$
\left(
\sum_{i=1}^{n}a_{ik}^s
\right)c
=
\sum_{i=1}^{n}a_{ik}^s z_i^s
$$

如果：

$$
\sum_{i=1}^{n}a_{ik}^s>0
$$

则：

$$
c=
\frac{
\sum_{i=1}^{n}a_{ik}^s z_i^s
}{
\sum_{i=1}^{n}a_{ik}^s
}
$$

所以 source 第 $k$ 个 prototype 是：

$$
c_k^s=
\frac{
\sum_{i=1}^{n}a_{ik}^s z_i^s
}{
\sum_{i=1}^{n}a_{ik}^s
}
$$

target 第 $l$ 个 prototype 同理：

$$
c_l^t=
\frac{
\sum_{j=1}^{m}a_{jl}^t z_j^t
}{
\sum_{j=1}^{m}a_{jl}^t
}
$$

因此，prototype 不是随便定义出来的，而是：

> 加权平方误差意义下最优的 group 代表点。

---

## 8. Prototype 和 K-means 中心的关系

K-means 的中心也是平均点，但它使用 hard assignment。

如果样本 $z_i$ 属于第 $k$ 类，则：

$$
a_{ik}=1
$$

否则：

$$
a_{ik}=0
$$

所以 K-means 中心是：

$$
c_k=
\frac{
\sum_i a_{ik}z_i
}{
\sum_i a_{ik}
}
$$

只不过这里 $a_{ik}$ 只有 0 或 1。

TACO-style prototype 使用 soft assignment：

$$
a_{ik}\in[0,1]
$$

所以一个样本可以同时影响多个 prototype。

这就是：

```text
hard cluster center
→ soft prototype
```

的关系。

---

## 9. Prototype 和 HiWA cluster 的本质区别

TACO-style 方法中，一个 group 被表示成一个 prototype 点：

$$
c_k^s
$$

从测度角度看，这相当于一个 Dirac 测度：

$$
\delta_{c_k^s}
$$

也就是说：

$$
\text{一个 group}\approx\text{一个点}
$$

而 HiWA 中，一个 cluster 通常是一整个经验分布：

$$
\mu_k^s
=
\sum_{i\in G_k^s}
w_i^s\delta_{z_i^s}
$$

其中：

- $G_k^s$ 是第 $k$ 个 source cluster 的样本集合；
- $z_i^s$ 是 cluster 中的样本点；
- $w_i^s$ 是样本质量；
- $\delta_{z_i^s}$ 是位于 $z_i^s$ 的 Dirac 测度。

所以：

$$
\mu_k^{TACO}=\delta_{c_k}
$$

而：

$$
\mu_k^{HiWA}=\sum_i w_i\delta_{z_i}
$$

结论是：

```text
TACO-style group 是单点分布。
HiWA cluster 是多点经验分布。
```

所以 TACO-style 的 group 表示更轻、更粗；HiWA 的 cluster 表示更细、更保留组内结构。

---

## 10. Group masses：$\alpha$ 和 $\beta$ 怎么来

做 group-level OT 时，需要把 prototypes 看成概率分布。

source 侧 group distribution 写成：

$$
\mu_G^s=
\sum_{k=1}^{K_s}
\alpha_k\delta_{c_k^s}
$$

target 侧 group distribution 写成：

$$
\mu_G^t=
\sum_{l=1}^{K_t}
\beta_l\delta_{c_l^t}
$$

其中：

- $\alpha_k$ 是 source 第 $k$ 个 group 的质量；
- $\beta_l$ 是 target 第 $l$ 个 group 的质量；
- $\delta_{c_k^s}$ 是集中在 prototype $c_k^s$ 上的 Dirac 测度。

质量必须满足：

$$
\alpha_k\geq0
$$

$$
\beta_l\geq0
$$

$$
\sum_{k=1}^{K_s}\alpha_k=1
$$

$$
\sum_{l=1}^{K_t}\beta_l=1
$$

最简单的选择是均匀质量：

$$
\alpha_k=\frac{1}{K_s}
$$

$$
\beta_l=\frac{1}{K_t}
$$

另一种更自然的选择是根据 soft assignment 的总权重定义 group mass。

source 第 $k$ 个 group 的总 assignment 权重为：

$$
\sum_{i=1}^{n}a_{ik}^s
$$

如果每个 source 样本质量是 $1/n$，那么：

$$
\alpha_k=
\frac{1}{n}
\sum_{i=1}^{n}a_{ik}^s
$$

验证它是概率分布：

$$
\sum_{k=1}^{K_s}\alpha_k
=
\sum_{k=1}^{K_s}
\frac{1}{n}
\sum_{i=1}^{n}a_{ik}^s
$$

交换求和顺序：

$$
=
\frac{1}{n}
\sum_{i=1}^{n}
\sum_{k=1}^{K_s}a_{ik}^s
$$

因为：

$$
\sum_{k=1}^{K_s}a_{ik}^s=1
$$

所以：

$$
=
\frac{1}{n}
\sum_{i=1}^{n}1
=
1
$$

因此 $\alpha$ 是合法概率分布。

target 侧同理：

$$
\beta_l=
\frac{1}{m}
\sum_{j=1}^{m}a_{jl}^t
$$

并且：

$$
\sum_{l=1}^{K_t}\beta_l=1
$$

---

## 11. Group-level cost matrix 怎么来

现在 source groups 被表示为：

$$
c_1^s,c_2^s,\dots,c_{K_s}^s
$$

target groups 被表示为：

$$
c_1^t,c_2^t,\dots,c_{K_t}^t
$$

我们需要知道 source 第 $k$ 个 group 匹配 target 第 $l$ 个 group 的代价是多少。

于是定义 group-level cost matrix：

$$
D^{group}\in\mathbb{R}^{K_s\times K_t}
$$

其中：

$$
D_{kl}^{group}=d(c_k^s,c_l^t)
$$

如果用平方欧氏距离：

$$
D_{kl}^{group}=\|c_k^s-c_l^t\|^2
$$

如果用余弦距离：

$$
D_{kl}^{group}
=
1-
\frac{
\langle c_k^s,c_l^t\rangle
}{
\|c_k^s\|\|c_l^t\|
}
$$

在 TACO-style 方法中，$D_{kl}^{group}$ 通常来自 prototype 和 prototype 的距离。

在 HiWA 中，$D_{kl}^{group}$ 通常来自两个 cluster empirical distributions 之间的 Wasserstein-type cost 或局部对齐代价。

---

## 12. Group-level OT 的合法传输集合

现在有两个 group 分布：

$$
\mu_G^s=
\sum_{k=1}^{K_s}\alpha_k\delta_{c_k^s}
$$

$$
\mu_G^t=
\sum_{l=1}^{K_t}\beta_l\delta_{c_l^t}
$$

我们要找 group-level transport matrix：

$$
P\in\mathbb{R}^{K_s\times K_t}
$$

其中 $P_{kl}$ 表示 source 第 $k$ 个 group 有多少质量传到 target 第 $l$ 个 group。

因为 $P_{kl}$ 是质量，所以：

$$
P_{kl}\geq0
$$

source 第 $k$ 个 group 总质量是 $\alpha_k$，它的质量必须全部分配出去：

$$
\sum_{l=1}^{K_t}P_{kl}=\alpha_k
$$

target 第 $l$ 个 group 需要接收 $\beta_l$ 的质量：

$$
\sum_{k=1}^{K_s}P_{kl}=\beta_l
$$

把这些条件合起来，得到合法集合：

$$
U(\alpha,\beta)
=
\left\{
P\in\mathbb{R}_+^{K_s\times K_t}
:
P\mathbf{1}_{K_t}=\alpha,
P^\top\mathbf{1}_{K_s}=\beta
\right\}
$$

其中：

- $\mathbf{1}_{K_t}$ 是长度为 $K_t$ 的全 1 向量；
- $P\mathbf{1}_{K_t}=\alpha$ 是行和约束；
- $P^\top\mathbf{1}_{K_s}=\beta$ 是列和约束。

---

## 13. Group-level OT 的目标函数为什么是矩阵内积

如果 source 第 $k$ 个 group 向 target 第 $l$ 个 group 运输了 $P_{kl}$ 的质量，而单位质量运输代价是 $D_{kl}^{group}$，那么这部分成本是：

$$
D_{kl}^{group}P_{kl}
$$

所有 group pair 的总代价是：

$$
\sum_{k=1}^{K_s}
\sum_{l=1}^{K_t}
D_{kl}^{group}P_{kl}
$$

这个双重求和可以写成矩阵内积：

$$
\langle D^{group},P\rangle
=
\sum_{k=1}^{K_s}
\sum_{l=1}^{K_t}
D_{kl}^{group}P_{kl}
$$

于是，不加熵正则时，group-level OT 是：

$$
P^*
=
\arg\min_{P\in U(\alpha,\beta)}
\langle D^{group},P\rangle
$$

意思是：

> 在满足 group mass 约束的前提下，找总代价最小的 group-to-group 传输方案。

---

## 14. 为什么要加熵正则

标准 OT 可能给出很硬的匹配，比如很多 $P_{kl}=0$。

在 TACO-style 方法中，我们通常希望 group 对齐是平滑的，因为：

1. soft assignment 本来就是软的；
2. group 对应关系可能不是一一对应；
3. 熵正则可以改善数值稳定性；
4. 熵正则方便使用 Sinkhorn 算法；
5. 如果放进深度学习训练，平滑解更适合反向传播。

定义负熵形式：

$$
H(P)=
\sum_{k,l}
P_{kl}(\log P_{kl}-1)
$$

于是熵正则 group-level OT 为：

$$
P^*
=
\arg\min_{P\in U(\alpha,\beta)}
\langle D^{group},P\rangle
+
\varepsilon H(P)
$$

其中：

$$
\varepsilon>0
$$

是熵正则强度。

如果 $\varepsilon$ 很小，解更接近 hard OT。如果 $\varepsilon$ 很大，解更分散、更平滑。

---

## 15. Sinkhorn 算法的完整推导

现在推导 group-level OT 的 Sinkhorn 形式。

目标函数是：

$$
\min_{P\in U(\alpha,\beta)}
\sum_{k,l}D_{kl}^{group}P_{kl}
+
\varepsilon
\sum_{k,l}P_{kl}(\log P_{kl}-1)
$$

约束是：

$$
\sum_lP_{kl}=\alpha_k
$$

$$
\sum_kP_{kl}=\beta_l
$$

构造拉格朗日函数。引入行约束乘子 $u_k$ 和列约束乘子 $v_l$：

$$
\mathcal{L}(P,u,v)
=
\sum_{k,l}D_{kl}^{group}P_{kl}
+
\varepsilon
\sum_{k,l}P_{kl}(\log P_{kl}-1)
+
\sum_k u_k
\left(
\sum_lP_{kl}-\alpha_k
\right)
+
\sum_l v_l
\left(
\sum_kP_{kl}-\beta_l
\right)
$$

对 $P_{kl}$ 求偏导。

第一项给出：

$$
\frac{\partial}{\partial P_{kl}}
D_{kl}^{group}P_{kl}
=
D_{kl}^{group}
$$

第二项给出：

$$
\frac{\partial}{\partial P_{kl}}
\varepsilon P_{kl}(\log P_{kl}-1)
=
\varepsilon\log P_{kl}
$$

因为：

$$
\frac{d}{dx}x(\log x-1)=\log x
$$

行约束项给出 $u_k$，列约束项给出 $v_l$。所以一阶条件是：

$$
D_{kl}^{group}
+
\varepsilon\log P_{kl}
+
u_k
+
v_l
=
0
$$

这里为了避免和后面的缩放向量混淆，令行乘子写作 $\nu_k$，列乘子写作 $v_l$。

移项：

$$
\varepsilon\log P_{kl}
=
-D_{kl}^{group}-\nu_k-v_l
$$

两边除以 $\varepsilon$：

$$
\log P_{kl}
=
-\frac{D_{kl}^{group}}{\varepsilon}
-\frac{\nu_k}{\varepsilon}
-\frac{v_l}{\varepsilon}
$$

两边取指数：

$$
P_{kl}
=
\exp\left(-\frac{\nu_k}{\varepsilon}\right)
\exp\left(-\frac{D_{kl}^{group}}{\varepsilon}\right)
\exp\left(-\frac{v_l}{\varepsilon}\right)
$$

定义：

$$
a_k=
\exp\left(-\frac{\nu_k}{\varepsilon}\right)
$$

$$
K_{kl}=
\exp\left(-\frac{D_{kl}^{group}}{\varepsilon}\right)
$$

$$
b_l=
\exp\left(-\frac{v_l}{\varepsilon}\right)
$$

于是：

$$
P_{kl}=a_kK_{kl}b_l
$$

写成矩阵形式：

$$
P=\mathrm{diag}(a)K\mathrm{diag}(b)
$$

其中：

$$
K=\exp(-D^{group}/\varepsilon)
$$

这里的指数是逐元素指数。

接下来让 $P$ 满足边缘约束。

行和约束：

$$
P\mathbf{1}_{K_t}=\alpha
$$

代入：

$$
\mathrm{diag}(a)K\mathrm{diag}(b)\mathbf{1}_{K_t}=\alpha
$$

由于：

$$
\mathrm{diag}(b)\mathbf{1}_{K_t}=b
$$

所以：

$$
\mathrm{diag}(a)Kb=\alpha
$$

逐元素写为：

$$
a\odot(Kb)=\alpha
$$

因此：

$$
a=\frac{\alpha}{Kb}
$$

列和约束：

$$
P^\top\mathbf{1}_{K_s}=\beta
$$

同理得到：

$$
b=\frac{\beta}{K^\top a}
$$

所以 Sinkhorn 迭代为：

$$
a\leftarrow\frac{\alpha}{Kb}
$$

$$
b\leftarrow\frac{\beta}{K^\top a}
$$

收敛后得到：

$$
P^*=\mathrm{diag}(a)K\mathrm{diag}(b)
$$

这就是熵正则 OT 的 Sinkhorn 求解。

---

## 16. Group-level OT 的输出含义

$P^*$ 是 group correspondence matrix。

如果：

$$
P_{kl}^*
$$

很大，说明 source 第 $k$ 个 group 和 target 第 $l$ 个 group 有较强对应关系。

如果 $P^*$ 接近对角矩阵，说明 group 之间接近一一对应。

如果 $P^*$ 很分散，说明 group 对应关系不确定，或者存在多对多关系。

所以 $P^*$ 不是最终的样本匹配矩阵，而是 group 层面的软匹配矩阵。

---

## 17. 为什么还需要 instance-level OT

到目前为止，我们只知道 group 如何对应。但是最终仍然需要知道 source 样本 $i$ 应该和 target 样本 $j$ 有多强关系。

所以需要 instance-level alignment。

直接定义样本距离：

$$
D_{ij}^{inst}=d(z_i^s,z_j^t)
$$

例如：

$$
D_{ij}^{inst}=\|z_i^s-z_j^t\|^2
$$

如果只用这个距离做 OT，那么就是普通样本级 OT：

$$
T^*
=
\arg\min_{T\in U(p,q)}
\langle D^{inst},T\rangle
+
\varepsilon_{inst}H(T)
$$

其中：

- $T\in\mathbb{R}^{n\times m}$ 是样本级 transport matrix；
- $T_{ij}$ 表示 source 样本 $i$ 传到 target 样本 $j$ 的质量；
- $p\in\mathbb{R}^n$ 是 source 样本质量；
- $q\in\mathbb{R}^m$ 是 target 样本质量；
- $U(p,q)$ 是满足边缘约束的 transport set。

但是 TACO-style 方法的核心是：

> 样本级对齐应该受到 group-level 对齐结果的指导。

所以要把 $P^*$ 传递回样本层。

---

## 18. 从 group-level 到 instance-level：$W=A^sP^*(A^t)^\top$

我们已经有三个东西。

第一，source 样本 $i$ 属于 source group $k$ 的程度：

$$
a_{ik}^s
$$

第二，source group $k$ 对应 target group $l$ 的程度：

$$
P_{kl}^*
$$

第三，target 样本 $j$ 属于 target group $l$ 的程度：

$$
a_{jl}^t
$$

那么，source 样本 $i$ 和 target 样本 $j$ 通过 group pair $(k,l)$ 建立联系的强度自然是：

$$
a_{ik}^sP_{kl}^*a_{jl}^t
$$

但是 $i$ 和 $j$ 可能通过很多 group pair 建立联系，所以对所有 $k,l$ 求和：

$$
W_{ij}
=
\sum_{k=1}^{K_s}
\sum_{l=1}^{K_t}
a_{ik}^sP_{kl}^*a_{jl}^t
$$

这就是样本对的 group support。

它表示：

> 从 group 层面看，source 样本 $i$ 和 target 样本 $j$ 有多应该匹配。

矩阵形式为：

$$
W=A^sP^*(A^t)^\top
$$

检查维度：

$$
A^s\in\mathbb{R}^{n\times K_s}
$$

$$
P^*\in\mathbb{R}^{K_s\times K_t}
$$

所以：

$$
A^sP^*\in\mathbb{R}^{n\times K_t}
$$

又因为：

$$
(A^t)^\top\in\mathbb{R}^{K_t\times m}
$$

所以：

$$
A^sP^*(A^t)^\top\in\mathbb{R}^{n\times m}
$$

这正好是样本对样本的矩阵。

---

## 19. $W$ 如何指导 instance-level OT

这里要特别说明：不同实现可以采用不同方式。下面给出一种数学上自然的推导方式，它可以解释为什么 $W$ 能进入样本级 OT。

我们希望最终样本级 transport $T$ 不仅考虑样本距离 $D^{inst}$，还考虑 group support $W$。

如果 $W_{ij}$ 大，说明 group 层面支持 $i$ 和 $j$ 匹配。  
如果 $W_{ij}$ 小，说明 group 层面不支持 $i$ 和 $j$ 匹配。

一种自然方式是把 $W$ 当成 prior，让 $T$ 不要偏离 $W$ 太远。

考虑优化问题：

$$
\min_{T\in U(p,q)}
\langle D^{inst},T\rangle
+
\lambda\mathrm{KL}(T\|W+\delta)
$$

其中：

- $\lambda>0$ 控制 group prior 的强度；
- $\delta>0$ 防止 $W_{ij}=0$ 时出现 $\log0$；
- $\mathrm{KL}(T\|W+\delta)$ 表示 $T$ 相对于 $W+\delta$ 的 KL 偏离。

KL 项展开：

$$
\mathrm{KL}(T\|W+\delta)
=
\sum_{i,j}
T_{ij}
\log
\frac{T_{ij}}{W_{ij}+\delta}
$$

继续展开：

$$
=
\sum_{i,j}T_{ij}\log T_{ij}
-
\sum_{i,j}T_{ij}\log(W_{ij}+\delta)
$$

因此目标变成：

$$
\langle D^{inst},T\rangle
+
\lambda
\sum_{i,j}T_{ij}\log T_{ij}
-
\lambda
\sum_{i,j}T_{ij}\log(W_{ij}+\delta)
$$

把第一项和第三项合并：

$$
=
\sum_{i,j}
T_{ij}
\left[
D_{ij}^{inst}
-
\lambda\log(W_{ij}+\delta)
\right]
+
\lambda
\sum_{i,j}T_{ij}\log T_{ij}
$$

于是定义被 group support 修正后的样本级代价：

$$
\widetilde{D}_{ij}^{inst}
=
D_{ij}^{inst}
-
\lambda\log(W_{ij}+\delta)
$$

这样目标变成类似熵正则 OT：

$$
\min_{T\in U(p,q)}
\langle\widetilde{D}^{inst},T\rangle
+
\lambda
\sum_{i,j}T_{ij}\log T_{ij}
$$

如果用负熵形式，也可以写成：

$$
T^*
=
\arg\min_{T\in U(p,q)}
\langle\widetilde{D}^{inst},T\rangle
+
\varepsilon_{inst}H(T)
$$

核心是：

$$
\widetilde{D}_{ij}^{inst}
=
D_{ij}^{inst}
-
\lambda\log(W_{ij}+\delta)
$$

如果 $W_{ij}$ 大，则 $\log(W_{ij}+\delta)$ 较大，于是 $-\lambda\log(W_{ij}+\delta)$ 会降低代价。

这表示：

> group 层面支持的样本对，在 instance-level OT 中更容易被匹配。

---

## 20. Instance-level OT 的完整形式

定义 source 样本质量：

$$
p=(p_1,p_2,\dots,p_n)
$$

target 样本质量：

$$
q=(q_1,q_2,\dots,q_m)
$$

通常可以取均匀质量：

$$
p_i=\frac{1}{n}
$$

$$
q_j=\frac{1}{m}
$$

样本级合法传输集合为：

$$
U(p,q)
=
\left\{
T\in\mathbb{R}_+^{n\times m}
:
T\mathbf{1}_m=p,
T^\top\mathbf{1}_n=q
\right\}
$$

修正后的样本级代价是：

$$
\widetilde{D}_{ij}^{inst}
=
D_{ij}^{inst}
-
\lambda\log(W_{ij}+\delta)
$$

于是 instance-level OT 为：

$$
T^*
=
\arg\min_{T\in U(p,q)}
\langle\widetilde{D}^{inst},T\rangle
+
\varepsilon_{inst}H(T)
$$

其中：

$$
\langle\widetilde{D}^{inst},T\rangle
=
\sum_{i=1}^{n}
\sum_{j=1}^{m}
\widetilde{D}_{ij}^{inst}T_{ij}
$$

$T^*$ 是样本级对应矩阵。

如果 $T_{ij}^*$ 大，说明 source 样本 $i$ 和 target 样本 $j$ 在最终对齐中有较强联系。

---

## 21. 整体损失函数

TACO-style 方法最终通常不是只为了得到 $P^*$ 和 $T^*$，而是要训练 encoder，使 source 和 target 表征空间更一致。

group-level loss 可以写成：

$$
\mathcal{L}_{group}
=
\langle D^{group},P^*\rangle
+
\varepsilon_{group}H(P^*)
$$

instance-level loss 可以写成：

$$
\mathcal{L}_{inst}
=
\langle\widetilde{D}^{inst},T^*\rangle
+
\varepsilon_{inst}H(T^*)
$$

总的 alignment loss 可以写成：

$$
\mathcal{L}_{align}
=
\mathcal{L}_{group}
+
\gamma\mathcal{L}_{inst}
$$

其中 $\gamma$ 控制样本级损失的重要性。

如果还有下游任务，例如分类、回归、能量预测、神经解码，则可以加任务损失：

$$
\mathcal{L}_{total}
=
\mathcal{L}_{task}
+
\eta\mathcal{L}_{group}
+
\gamma\mathcal{L}_{inst}
$$

其中：

- $\mathcal{L}_{task}$ 是任务损失；
- $\eta$ 控制 group-level alignment 的权重；
- $\gamma$ 控制 instance-level alignment 的权重。

如果迁移到猕猴神经元实验，$\mathcal{L}_{task}$ 可以是 movement direction decoding loss。

但这是项目改造建议，不是 HiWA 原论文自带内容。

---

## 22. 具体小例子：2 个 source group 和 2 个 target group

现在用一个极小例子理解 group-level OT。

假设 source 有 2 个 group：

$$
G_1^s,G_2^s
$$

target 有 2 个 group：

$$
G_1^t,G_2^t
$$

它们的质量都是均匀的：

$$
\alpha=
\left(
\frac{1}{2},\frac{1}{2}
\right)
$$

$$
\beta=
\left(
\frac{1}{2},\frac{1}{2}
\right)
$$

假设 prototype 距离算出来的 group cost matrix 是：

$$
D^{group}
=
\begin{pmatrix}
0.1 & 2.0 \\
1.5 & 0.2
\end{pmatrix}
$$

这表示：

- $G_1^s$ 匹配 $G_1^t$ 的代价是 $0.1$，很小；
- $G_1^s$ 匹配 $G_2^t$ 的代价是 $2.0$，很大；
- $G_2^s$ 匹配 $G_1^t$ 的代价是 $1.5$，较大；
- $G_2^s$ 匹配 $G_2^t$ 的代价是 $0.2$，很小。

group-level transport matrix 是：

$$
P=
\begin{pmatrix}
P_{11} & P_{12} \\
P_{21} & P_{22}
\end{pmatrix}
$$

行和约束为：

$$
P_{11}+P_{12}=\frac{1}{2}
$$

$$
P_{21}+P_{22}=\frac{1}{2}
$$

列和约束为：

$$
P_{11}+P_{21}=\frac{1}{2}
$$

$$
P_{12}+P_{22}=\frac{1}{2}
$$

不加熵时，目标函数是：

$$
\min_P
0.1P_{11}
+
2.0P_{12}
+
1.5P_{21}
+
0.2P_{22}
$$

由于 $P_{11}$ 和 $P_{22}$ 的代价很小，而 $P_{12}$ 和 $P_{21}$ 的代价较大，所以最优传输会倾向于：

$$
P^*
\approx
\begin{pmatrix}
0.5 & 0 \\
0 & 0.5
\end{pmatrix}
$$

这表示：

- source group 1 对应 target group 1；
- source group 2 对应 target group 2。

如果加入熵正则，结果不会完全是 0，而会更平滑，例如：

$$
P^*
\approx
\begin{pmatrix}
0.47 & 0.03 \\
0.03 & 0.47
\end{pmatrix}
$$

这表示主要还是对角匹配，但允许少量跨 group 的软质量。

这就是熵正则 OT 的直观效果。

---

## 23. 小例子继续：group support 如何影响 instance-level

假设 source 有 3 个样本，target 有 3 个样本。

source assignment matrix 是：

$$
A^s=
\begin{pmatrix}
0.9 & 0.1 \\
0.8 & 0.2 \\
0.1 & 0.9
\end{pmatrix}
$$

这表示：

- source 样本 1 主要属于 group 1；
- source 样本 2 主要属于 group 1；
- source 样本 3 主要属于 group 2。

target assignment matrix 是：

$$
A^t=
\begin{pmatrix}
0.85 & 0.15 \\
0.2 & 0.8 \\
0.1 & 0.9
\end{pmatrix}
$$

假设 group-level OT 得到：

$$
P^*=
\begin{pmatrix}
0.47 & 0.03 \\
0.03 & 0.47
\end{pmatrix}
$$

那么样本级 group support 是：

$$
W=A^sP^*(A^t)^\top
$$

例如 source 样本 1 和 target 样本 1 的 support 是：

$$
W_{11}
=
\sum_{k=1}^{2}
\sum_{l=1}^{2}
a_{1k}^sP_{kl}^*a_{1l}^t
$$

展开：

$$
W_{11}
=
a_{11}^sP_{11}^*a_{11}^t
+
a_{11}^sP_{12}^*a_{12}^t
+
a_{12}^sP_{21}^*a_{11}^t
+
a_{12}^sP_{22}^*a_{12}^t
$$

代入数值：

$$
W_{11}
=
0.9\cdot0.47\cdot0.85
+
0.9\cdot0.03\cdot0.15
+
0.1\cdot0.03\cdot0.85
+
0.1\cdot0.47\cdot0.15
$$

因为 source 样本 1 和 target 样本 1 都主要属于 group 1，而 group 1 与 group 1 的 $P_{11}^*$ 很大，所以 $W_{11}$ 会比较大。

这意味着：

> group 层面支持 source 样本 1 和 target 样本 1 匹配。

相反，如果 source 样本 1 主要属于 group 1，而 target 样本 3 主要属于 group 2，那么它们的 support 会小一些，因为 $P_{12}^*$ 较小。

最后通过：

$$
\widetilde{D}_{ij}^{inst}
=
D_{ij}^{inst}
-
\lambda\log(W_{ij}+\delta)
$$

使得 group support 大的样本对代价更低，更容易在 instance-level OT 中被匹配。

---

## 24. TACO-style 方法和 HiWA 的本质区别

现在回答最核心的问题：这个公式到底在解决什么问题？和 HiWA 的本质区别在哪里？

### 24.1 共同点

TACO-style 和 HiWA 都有层级思想。

都可以抽象成：

```text
先做 group / cluster 层面的对齐
再做样本 / cluster 内部的细粒度对齐
```

因此二者都会出现类似的外层 OT：

$$
P^*
=
\arg\min_{P\in U(\alpha,\beta)}
\langle D^{group},P\rangle+\varepsilon H(P)
$$

这个公式解决的是：

> 哪个 group / cluster 对应哪个 group / cluster？

### 24.2 本质区别一：group 的表示不同

TACO-style 中：

$$
\text{group}\approx\delta_{c_k}
$$

也就是一个 prototype 点。

HiWA 中：

$$
\text{cluster}\approx\sum_iw_i\delta_{z_i}
$$

也就是一整个经验分布。

因此：

```text
TACO-style 是点代表组。
HiWA 是分布代表簇。
```

### 24.3 本质区别二：group cost 的来源不同

TACO-style 中：

$$
D_{kl}^{group}=d(c_k^s,c_l^t)
$$

也就是两个 prototype 点之间的距离。

HiWA 中：

$$
D_{kl}^{group}
\approx
W_2^2(\mu_k^s,\nu_l^t)
$$

或者更具体地：

$$
D_{kl}^{group}
=
\min_{R_{kl},Q_{kl}}
\sum_{i,j}
Q_{kl,ij}
\|R_{kl}x_{ki}^s-x_{lj}^t\|^2
$$

也就是说，HiWA 的 group cost 来自簇内经验分布之间的 OT 或局部对齐问题。

所以：

```text
TACO-style 的 group cost 更轻。
HiWA 的 group cost 更细。
```

### 24.4 本质区别三：后续目标不同

HiWA 更强调：

```text
对齐已有点云或已有分布结构
```

TACO-style 更强调：

```text
学习表征空间，并让 soft groups / prototypes 在表征空间中对齐
```

所以 TACO-style 方法更容易和 encoder、contrastive learning、下游任务 loss 结合。

---

## 25. 为什么这个公式会自然出现

公式：

$$
P^*
=
\arg\min_{P\in U(\alpha,\beta)}
\langle D^{group},P\rangle+\varepsilon H(P)
$$

自然出现的原因是：

1. 我们已经把 source 样本组织成 $K_s$ 个 groups；
2. 也把 target 样本组织成 $K_t$ 个 groups；
3. 每个 group 有一个质量；
4. 每一对 group 有一个匹配代价；
5. 因此就需要找一个矩阵 $P$，表示 group 之间如何运输质量；
6. 为了让总代价最小，就最小化 $\langle D^{group},P\rangle$；
7. 为了保证质量守恒，就加入 $P\in U(\alpha,\beta)$；
8. 为了平滑可微和方便求解，就加入熵正则 $\varepsilon H(P)$。

所以这个公式不是凭空来的，它是：

```text
group 表示
+ group 质量
+ group 匹配代价
+ 质量守恒约束
+ 熵正则
= group-level entropic OT
```

---

## 26. 它在 TACO / HiWA 框架中处于哪一步

在 TACO-style 框架中，它处于：

```text
soft assignment 之后
prototype 之后
instance-level OT 之前
```

完整流程是：

```text
原始样本
→ encoder 得到 representation
→ soft assignment 得到 A
→ 加权平均得到 prototypes
→ 计算 group cost D_group
→ group-level OT 得到 P*
→ 由 W = A_s P* A_t^T 得到样本级 group support
→ 修正 instance cost
→ instance-level OT 得到 T*
→ 形成 alignment loss
```

在 HiWA 中，类似位置是：

```text
cluster 内部代价计算之后
cluster-level OT 之中
全局旋转 / 局部旋转协调之前或过程中
```

但 HiWA 的 $D^{group}$ 来自 cluster empirical distributions，而不是 prototype 点距离。

---

## 27. 如何迁移到猕猴神经元实验

原始 HiWA 猕猴神经元实验大致是：

```text
神经放电数据
→ Factor Analysis 降到 3D
→ 根据 movement direction 得到 hard clusters
→ HiWA 做 cluster-level OT 和 cluster 内部 OT
→ 学全局旋转
→ 评估 direction decoding accuracy
```

如果引入 TACO-style soft-prototype 思想，可以改成：

```text
神经放电数据
→ neural encoder / 降维方法
→ neural representation
→ soft assignment
→ neural prototypes
→ movement prototypes
→ prototype-level group OT
→ group-guided instance OT
→ movement direction decoding
```

数学上可以写为：

$$
z_i^{neural}=f_{neural}(y_i)
$$

其中 $y_i$ 是原始神经放电样本，$z_i^{neural}$ 是神经表征。

然后：

$$
a_{ik}^{neural}
=
\frac{
\exp(\langle z_i^{neural},q_k^{neural}\rangle/\tau)
}{
\sum_r
\exp(\langle z_i^{neural},q_r^{neural}\rangle/\tau)
}
$$

得到 neural soft assignment。

neural prototype 是：

$$
c_k^{neural}
=
\frac{
\sum_i a_{ik}^{neural}z_i^{neural}
}{
\sum_i a_{ik}^{neural}
}
$$

movement 侧同理得到：

$$
c_l^{move}
$$

然后构造：

$$
D_{kl}^{group}
=
d(c_k^{neural},c_l^{move})
$$

再做：

$$
P^*
=
\arg\min_{P\in U(\alpha,\beta)}
\langle D^{group},P\rangle+\varepsilon H(P)
$$

这就是 Soft-Prototype HiWA 的第一步。

如果继续做 instance-level OT，则：

$$
W=A^{neural}P^*(A^{move})^\top
$$

然后：

$$
\widetilde{D}_{ij}^{inst}
=
D_{ij}^{inst}
-
\lambda\log(W_{ij}+\delta)
$$

最后做样本级 OT：

$$
T^*
=
\arg\min_{T\in U(p,q)}
\langle\widetilde{D}^{inst},T\rangle
+
\varepsilon_{inst}H(T)
$$

这样，你就把 HiWA 的 hard-cluster distribution alignment 改成了 TACO-style soft-prototype hierarchical alignment。

---

## 28. 对猕猴神经元项目的直观解释

在原始 HiWA 中，一个 reach direction 往往被看成一个 hard cluster。

例如：

```text
0 度方向
45 度方向
90 度方向
...
```

每个神经样本只属于一个方向簇。

但是真实神经活动可能不是这么干净。一个神经活动样本可能同时包含方向编码、速度编码、运动准备阶段、运动执行阶段、噪声和漂移。

所以可以引入 soft group：

$$
a_i^{neural}=(0.6,0.3,0.1)
$$

表示这个神经样本不是绝对属于某一个簇，而是由多个潜在神经模式混合而成。

然后 prototype 表示这些潜在神经模式的中心：

$$
c_k^{neural}
$$

movement 侧也可以有 movement prototypes：

$$
c_l^{move}
$$

group-level OT 学的是：

> 哪些神经响应模式对应哪些运动行为模式？

instance-level OT 学的是：

> 在这些模式对应关系的指导下，具体哪些神经样本对应哪些运动样本？

这就是 TACO-style 方法对猕猴神经元实验最有价值的启发。

---

## 29. 最终总结

公式：

$$
P^*
=
\arg\min_{P\in U(\alpha,\beta)}
\langle D^{group},P\rangle+\varepsilon H(P)
$$

解决的是：

> group 和 group 之间怎么软对应。

它自然出现，是因为我们已经有了：

```text
source groups
target groups
group masses
group-to-group costs
quality conservation constraints
```

所以需要一个 OT 矩阵 $P$ 来表达 group correspondences。

它和 HiWA 的本质区别是：

```text
TACO-style 方法通常用 prototype 点表示 group。
HiWA 通常用经验分布表示 cluster。
```

因此二者外层 OT 公式相似，但 $D^{group}$ 的来源不同：

$$
D_{kl}^{group,TACO}=d(c_k^s,c_l^t)
$$

而：

$$
D_{kl}^{group,HiWA}
\approx
W_2^2(\mu_k^s,\nu_l^t)
$$

如果迁移到猕猴神经元实验，可以这样理解：

> 原始 HiWA 是 hard cluster 的层级 OT；TACO-style 改造是 soft prototype 的层级 OT。它允许神经样本以不同程度属于多个潜在神经模式，再用这些模式和运动模式做 group-level 对齐，最后指导样本级对齐和运动方向解码。
