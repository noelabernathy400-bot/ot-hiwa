# sinkhorn处理更新Qij和Pij

## 1. 本节在整篇论文中的位置

在前面的学习中，我们已经理解了 HiWA 的几个核心部分：

1. 将簇写成经验测度；
2. 用 Wasserstein 距离衡量簇与簇之间的对齐代价；
3. 引入全局正交变换 $R$；
4. 固定 $Q_{ij}$ 时，把更新 $R$ 的问题化成正交 Procrustes 问题；
5. 为了让局部旋转 $R_{ij}$ 和全局旋转 $R$ 最终一致，引入 ADMM。

这一节要解决的问题是：

- 固定 $R_{ij}$ 之后，如何更新点级运输计划 $Q_{ij}$；
- 固定所有局部代价之后，如何更新簇级对应矩阵 $P$。

论文中的答案是：

> 这两步本质上都是 **熵正则化最优传输问题**，因此都可以用 **Sinkhorn 算法** 来求解。

---

## 2. 先纠正一个容易混淆的理解

一个比较自然但不完全准确的说法是：

> “先求 $R$，再求 $Q$，再用完全一致的方法求 $P$，然后循环。”

这个说法的主线方向是对的，但需要更准确地表述。

### 更准确的说法

HiWA 的每轮迭代大致是：

1. **固定当前的 $Q_{ij}$、$P$、$R$、$\Lambda_{ij}$，更新局部旋转 $R_{ij}$**；
2. **固定当前的 $R_{ij}$，更新每一对簇的点级运输计划 $Q_{ij}$**；
3. **根据更新后的 $R_{ij}$ 和 $Q_{ij}$，计算局部代价 $C_{ij}$**；
4. **固定这些 $C_{ij}$，更新簇级对应矩阵 $P$**；
5. **根据所有局部 $R_{ij}$，更新全局旋转 $R$**；
6. **更新乘子 $\Lambda_{ij}$，让局部旋转和全局旋转逐步一致。**

所以：

- 更新 $R_{ij}$ 用的是 **Procrustes + SVD**；
- 更新 $Q_{ij}$ 用的是 **Sinkhorn**；
- 更新 $P$ 也用的是 **Sinkhorn**；
- 更新 $R$ 和更新 $R_{ij}$ 的思想相近，也是正交 Procrustes 型更新；
- ADMM 不是“用来求 $R$ 的唯一方法”，而是用来处理“局部旋转和全局旋转一致性约束”的框架。

---

## 3. 更新 $Q_{ij}$：点级熵正则最优传输

### 3.1 局部代价函数

对于一对簇 $(i,j)$，局部代价是：

$$
C_{ij}(R_{ij},Q_{ij})
=
\frac{1}{D}
\sum_{k,l}
Q_{ij}(k,l)
\lVert R_{ij}X_i(k)-Y_j(l)\rVert_2^2.
$$

这里：

- $X_i(k)$ 是第 $i$ 个源簇中的第 $k$ 个点；
- $Y_j(l)$ 是第 $j$ 个目标簇中的第 $l$ 个点；
- $Q_{ij}(k,l)$ 是从 $X_i(k)$ 向 $Y_j(l)$ 运输的质量；
- $R_{ij}$ 是当前这一对簇的局部旋转；
- $D$ 是数据维度。

### 3.2 固定 $R_{ij}$，把它写成代价矩阵形式

固定 $R_{ij}$ 后，定义点对点代价矩阵：

$$
M_{ij}(k,l)
=
\frac{1}{D}
\lVert R_{ij}X_i(k)-Y_j(l)\rVert_2^2.
$$

那么局部代价可以写成：

$$
C_{ij}(R_{ij},Q_{ij})
=
\sum_{k,l}Q_{ij}(k,l)M_{ij}(k,l).
$$

也可以写成 Frobenius 内积形式：

$$
C_{ij}(R_{ij},Q_{ij})
=
\langle Q_{ij},M_{ij}\rangle.
$$

其中：

$$
\langle A,B\rangle
=
\sum_{k,l}A_{kl}B_{kl}.
$$

这一步的意义是：

> 一旦 $R_{ij}$ 固定，更新 $Q_{ij}$ 就变成了一个“在固定代价矩阵上求最优运输计划”的问题。

---

## 4. 为什么还要加熵正则？

如果没有熵正则，更新 $Q_{ij}$ 的问题就是：

$$
\min_{Q_{ij}\in U(n_{x,i},n_{y,j})}
P_{ij}\langle Q_{ij},M_{ij}\rangle.
$$

其中 $U(n_{x,i},n_{y,j})$ 表示合法运输计划集合，即 $Q_{ij}$ 满足：

- 非负；
- 行和约束；
- 列和约束。

但是论文没有直接用这个“裸 OT”，而是加上熵正则：

$$
H_{\varepsilon_2}(Q_{ij})
=
\varepsilon_2
\sum_{k,l}
Q_{ij}(k,l)
\bigl(\log Q_{ij}(k,l)-1\bigr).
$$

于是子问题变成：

$$
\min_{Q_{ij}\in U(n_{x,i},n_{y,j})}
P_{ij}\langle Q_{ij},M_{ij}\rangle
+
\varepsilon_2
\sum_{k,l}
Q_{ij}(k,l)
\bigl(\log Q_{ij}(k,l)-1\bigr).
$$

### 4.1 熵正则的作用

熵正则不是经验凑出来的，它是最优传输中的标准做法。它有几个作用：

1. **让运输计划更平滑**：避免过早得到过于尖锐、过于硬的匹配；
2. **让问题更稳定**：在数值优化时更容易处理；
3. **让问题可以用 Sinkhorn 算法快速求解**；
4. **从建模角度说，它鼓励“高熵”的柔性匹配。**

直觉上说：

- 不加熵时，$Q_{ij}$ 容易非常稀疏；
- 加熵后，$Q_{ij}$ 会更“软”，也更适合在早期迭代中避免错误的过硬对齐。

---

## 5. 为什么会出现 $\varepsilon_2 / P_{ij}$？

这是论文里一个很重要的细节。

我们有：

$$
\min_{Q_{ij}\in U(n_{x,i},n_{y,j})}
P_{ij}\langle Q_{ij},M_{ij}\rangle
+
\varepsilon_2
\sum_{k,l}
Q_{ij}(k,l)
\bigl(\log Q_{ij}(k,l)-1\bigr).
$$

由于对固定的 $(i,j)$ 来说，$P_{ij}$ 是一个常数，只要 $P_{ij}>0$，我们可以把整个目标除以 $P_{ij}$，最优解不变：

$$
\min_{Q_{ij}\in U(n_{x,i},n_{y,j})}
\langle Q_{ij},M_{ij}\rangle
+
\frac{\varepsilon_2}{P_{ij}}
\sum_{k,l}
Q_{ij}(k,l)
\bigl(\log Q_{ij}(k,l)-1\bigr).
$$

所以这对簇对应的“有效熵强度”变成了：

$$
\eta_{ij}=\frac{\varepsilon_2}{P_{ij}}.
$$

### 5.1 这背后的意义

这个设计非常有直觉：

- 如果 $P_{ij}$ 大，说明这对簇更可信，那么 $\eta_{ij}$ 小，$Q_{ij}$ 会更接近精细匹配；
- 如果 $P_{ij}$ 小，说明这对簇不太可信，那么 $\eta_{ij}$ 大，$Q_{ij}$ 会更软，不会对这个不可信簇对过拟合。

也就是说：

> 越可信的簇对，内部点匹配越精细；越不可信的簇对，内部点匹配越保守。

---

## 6. Sinkhorn 的标准形式

现在把问题写成标准熵正则 OT 形式。

给定代价矩阵 $M$，考虑：

$$
\min_{Q\in U(a,b)}
\sum_{k,l}Q_{kl}M_{kl}
+
\eta
\sum_{k,l}Q_{kl}(\log Q_{kl}-1).
$$

其中：

- $a$ 是源边缘分布；
- $b$ 是目标边缘分布；
- $U(a,b)$ 表示满足边缘约束的运输计划集合。

### 6.1 拉格朗日函数

为处理边缘约束，引入乘子 $\alpha,\beta$：

$$
\mathcal{L}(Q,\alpha,\beta)
=
\sum_{k,l}Q_{kl}M_{kl}
+
\eta\sum_{k,l}Q_{kl}(\log Q_{kl}-1)
+
\sum_k \alpha_k\left(\sum_l Q_{kl}-a_k\right)
+
\sum_l \beta_l\left(\sum_k Q_{kl}-b_l\right).
$$

对 $Q_{kl}$ 求偏导：

$$
\frac{\partial\mathcal{L}}{\partial Q_{kl}}
=
M_{kl}+
\eta\log Q_{kl}+
\alpha_k+
\beta_l.
$$

令偏导为 $0$：

$$
M_{kl}+
\eta\log Q_{kl}+
\alpha_k+
\beta_l=0.
$$

所以：

$$
\eta\log Q_{kl}
=
-M_{kl}-\alpha_k-\beta_l,
$$

从而：

$$
Q_{kl}
=
\exp\left(-\frac{M_{kl}}{\eta}\right)
\exp\left(-\frac{\alpha_k}{\eta}\right)
\exp\left(-\frac{\beta_l}{\eta}\right).
$$

令：

$$
K_{kl}=\exp\left(-\frac{M_{kl}}{\eta}\right),
$$

$$
u_k=\exp\left(-\frac{\alpha_k}{\eta}\right),
$$

$$
v_l=\exp\left(-\frac{\beta_l}{\eta}\right).
$$

则：

$$
Q_{kl}=u_kK_{kl}v_l.
$$

写成矩阵形式：

$$
Q=\operatorname{diag}(u)K\operatorname{diag}(v).
$$

这就是 Sinkhorn 的核心结构。

---

## 7. Sinkhorn 迭代是怎么来的？

我们还要让 $Q$ 满足边缘约束：

$$
Q\mathbf{1}=a,
$$

$$
Q^\top\mathbf{1}=b.
$$

因为：

$$
Q=\operatorname{diag}(u)K\operatorname{diag}(v),
$$

所以：

$$
\operatorname{diag}(u)Kv=a.
$$

于是：

$$
u=\frac{a}{Kv}.
$$

同理：

$$
v=\frac{b}{K^\top u}.
$$

这就得到 Sinkhorn 迭代：

$$
u^{(t+1)}=\frac{a}{Kv^{(t)}},
$$

$$
v^{(t+1)}=\frac{b}{K^\top u^{(t+1)}}.
$$

这两个式子交替迭代，直到边缘约束满足得足够好。

---

## 8. 在 HiWA 中如何更新 $Q_{ij}$？

对每一对簇 $(i,j)$：

### 第一步：计算代价矩阵

$$
M_{ij}(k,l)
=
\frac{1}{D}
\lVert R_{ij}X_i(k)-Y_j(l)\rVert_2^2.
$$

### 第二步：确定正则强度

$$
\eta_{ij}=\frac{\varepsilon_2}{P_{ij}}.
$$

### 第三步：构造核矩阵

$$
K_{ij}(k,l)
=
\exp\left(-\frac{M_{ij}(k,l)}{\eta_{ij}}\right).
$$

也可以写成：

$$
K_{ij}(k,l)
=
\exp\left(-\frac{P_{ij}M_{ij}(k,l)}{\varepsilon_2}\right).
$$

### 第四步：用 Sinkhorn 迭代求 $Q_{ij}$

$$
Q_{ij}=\operatorname{diag}(u)K_{ij}\operatorname{diag}(v).
$$

这里的 $u,v$ 通过交替归一化求得，使 $Q_{ij}$ 满足边缘分布约束。

---

## 9. 更新 $P$：外层 Sinkhorn

现在来看簇级对应矩阵 $P$。

当所有局部代价 $C_{ij}$ 都已计算出来后，我们得到一个簇级代价矩阵：

$$
C=(C_{ij})\in\mathbb{R}^{S\times S}.
$$

更新 $P$ 的问题是：

$$
\min_{P\in B_S}
\sum_{i,j}P_{ij}C_{ij}
+
\varepsilon_1
\sum_{i,j}P_{ij}(\log P_{ij}-1).
$$

这里：

- $B_S$ 是 Birkhoff polytope（Birkhoff 多面体）；
- 它表示双随机矩阵集合；
- 这意味着 $P$ 要满足簇级的质量平衡约束。

### 9.1 这和更新 $Q_{ij}$ 的形式完全一致

它仍然是一个熵正则 OT 问题，只不过：

- 运输对象从点变成了簇；
- 代价矩阵从 $M_{ij}(k,l)$ 变成了 $C_{ij}$；
- 正则强度从 $\eta_{ij}$ 变成了统一的 $\varepsilon_1$。

因此同样可以用 Sinkhorn。

定义：

$$
K^P_{ij}=\exp\left(-\frac{C_{ij}}{\varepsilon_1}\right).
$$

那么：

$$
P=\operatorname{diag}(u)K^P\operatorname{diag}(v),
$$

再通过 Sinkhorn 迭代，使其满足双随机约束。

如果采用均匀簇质量，则约束可写为：

$$
P\mathbf{1}=\frac{1}{S}\mathbf{1},
$$

$$
P^\top\mathbf{1}=\frac{1}{S}\mathbf{1}.
$$

---

## 10. 更新 $Q_{ij}$ 和更新 $P$ 的异同

### 相同点

它们都是：

1. 熵正则化最优传输问题；
2. 都有一个代价矩阵；
3. 都有边缘约束；
4. 都可以写成：

$$
\min_\Gamma \langle \Gamma,C\rangle + \text{熵项};
$$

5. 都可以用 Sinkhorn 算法求解。

### 不同点

| 对象 | 作用层级 | 代价矩阵 | 正则强度 |
|---|---|---|---|
| $Q_{ij}$ | 点级 | $M_{ij}(k,l)=\frac{1}{D}\lVert R_{ij}X_i(k)-Y_j(l)\rVert^2$ | $\varepsilon_2/P_{ij}$ |
| $P$ | 簇级 | $C_{ij}$ | $\varepsilon_1$ |

所以可以这样记：

```text
Q_ij：内层 Sinkhorn，解决簇内点级匹配。
P：外层 Sinkhorn，解决簇间对应。
```

---

## 11. 这一节在 Algorithm 1 中对应什么？

把本节内容放回 Algorithm 1，可以理解为：

1. 更新局部旋转 $R_{ij}$：用 Procrustes / SVD；
2. **更新点级运输计划 $Q_{ij}$：用内层 Sinkhorn；**
3. 计算局部代价 $C_{ij}$；
4. **更新簇级对应矩阵 $P$：用外层 Sinkhorn；**
5. 更新全局旋转 $R$；
6. 更新乘子 $\Lambda_{ij}$。

所以 Sinkhorn 在论文里出现了两次：

- 一次是求点级匹配 $Q_{ij}$；
- 一次是求簇级匹配 $P$。

---

## 12. 本节核心理解总结

### 12.1 你目前可以这样概括 HiWA 的迭代思想

> HiWA 不是“只在初始条件下求一次 $R$ 然后就结束”，而是在迭代中交替更新多个变量。固定当前运输计划时，用 Procrustes / SVD 更新局部旋转 $R_{ij}$；固定当前旋转时，用熵正则 OT 和 Sinkhorn 更新点级运输计划 $Q_{ij}$；再根据所有局部代价，用同样的熵正则 OT 思想和 Sinkhorn 更新簇级对应矩阵 $P$；最后再更新全局旋转 $R$ 和 ADMM 乘子 $\Lambda_{ij}$。整个算法就是不断在这些变量之间交替优化，直到收敛。

### 12.2 本节最重要的一句话

> 更新 $Q_{ij}$ 和更新 $P$ 本质上都是熵正则化最优传输问题，只是一个在点级层面上做，一个在簇级层面上做；因此它们都可以用 Sinkhorn 算法求解。

---

## 13. 你之后复习时最值得记住的公式

### 点级代价矩阵

$$
M_{ij}(k,l)=\frac{1}{D}\lVert R_{ij}X_i(k)-Y_j(l)\rVert_2^2.
$$

### 点级更新问题

$$
\min_{Q_{ij}\in U(n_{x,i},n_{y,j})}
\langle Q_{ij},M_{ij}\rangle
+
\frac{\varepsilon_2}{P_{ij}}
\sum_{k,l}Q_{ij}(k,l)(\log Q_{ij}(k,l)-1).
$$

### Sinkhorn 结构

$$
Q=\operatorname{diag}(u)K\operatorname{diag}(v),
$$

$$
K_{kl}=\exp\left(-\frac{M_{kl}}{\eta}\right).
$$

### Sinkhorn 迭代

$$
u=\frac{a}{Kv},
$$

$$
v=\frac{b}{K^\top u}.
$$

### 簇级更新问题

$$
\min_{P\in B_S}
\sum_{i,j}P_{ij}C_{ij}
+
\varepsilon_1\sum_{i,j}P_{ij}(\log P_{ij}-1).
$$

### 簇级核矩阵

$$
K^P_{ij}=\exp\left(-\frac{C_{ij}}{\varepsilon_1}\right).
$$

---

## 14. 下一步学习建议

下一节最自然的学习内容是：

# Algorithm 1 逐行精读

到那时，我们会把：

- $R_{ij}$ 的更新；
- $Q_{ij}$ 的更新；
- $P$ 的更新；
- 全局 $R$ 的更新；
- $\Lambda_{ij}$ 的更新；

全部串起来，让你能够真正“看懂论文算法每一行到底在干什么”。
