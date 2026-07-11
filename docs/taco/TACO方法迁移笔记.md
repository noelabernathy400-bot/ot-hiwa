# TACO 方法迁移笔记：移植到 HiWA / 猕猴神经元实验时要注意什么

## 0. 核心判断

TACO 的方法思想确实是可移植的。

它最有价值的地方不是某一个特定应用场景，而是这条通用方法链：

```text
原始数据
→ 表征空间
→ soft assignment
→ group prototype
→ group-level OT
→ instance-level OT
→ 下游任务
```

这条链可以从文本-几何对齐迁移到神经元-运动行为对齐。

但是要注意：

> 可移植不等于可以直接照搬。

TACO 能迁移，是因为它提供了一种通用的“软分组 + 原型点 + 层级 OT”框架；但具体到猕猴神经元实验，你必须重新设计：

1. 表征空间怎么构造；
2. group 是什么；
3. prototype 怎么定义；
4. group cost 怎么算；
5. 是否还保留 HiWA 的簇内分布结构；
6. 最终任务指标是什么；
7. 对齐是否真的提升神经解码。

---

## 1. 移植时最重要的问题：你到底在对齐什么

TACO 原来的思想大致是：

$$
\text{text representation}
\longleftrightarrow
\text{geometry representation}
$$

如果迁移到猕猴神经元实验，你要先明确：

$$
\text{neural representation}
\longleftrightarrow
\text{movement representation}
$$

也就是说，不能只说：

> 我要把 TACO 用到神经元数据上。

而要说清楚：

> 我用 TACO-style soft-prototype OT，对齐神经活动表征空间和运动行为表征空间。

在 HiWA 原论文中，神经部分大致是：

$$
\text{高维神经放电}
\rightarrow
\text{Factor Analysis}
\rightarrow
\mathbb{R}^3
$$

然后和运动空间中的 3D 表示对齐。

如果你移植 TACO，可以改成：

$$
\text{高维神经放电}
\rightarrow
\text{neural encoder}
\rightarrow
\text{neural representation}
\rightarrow
\text{soft prototypes}
$$

运动侧也构造：

$$
\text{movement variables}
\rightarrow
\text{movement representation}
\rightarrow
\text{movement prototypes}
$$

然后做：

$$
\text{neural prototypes}
\longleftrightarrow
\text{movement prototypes}
$$

---

## 2. 注意事项一：不能把 prototype 当成真实神经簇

TACO 中一个 group 往往用一个 prototype 点表示：

$$
c_k =
\frac{
\sum_i a_{ik}z_i
}{
\sum_i a_{ik}
}
$$

其中：

- $z_i$ 是样本表示；
- $a_{ik}$ 是样本 $i$ 属于第 $k$ 个 group 的软权重；
- $c_k$ 是第 $k$ 个 group 的代表点。

这个 prototype 只是一个**加权平均中心**。

它不是：

- 真实存在的某个神经元；
- 真实存在的某个 trial；
- 必然具有明确生理含义的功能群；
- 必然对应某个运动方向。

所以移植时要小心：

> prototype 是模型学出来的潜在代表点，不一定天然等于神经科学中的真实功能模块。

如果你要解释它，需要通过后续分析验证。例如：

1. 某个 prototype 是否主要对应某个运动方向；
2. 某个 prototype 是否对应准备期或执行期；
3. 某个 prototype 是否和速度、方向、时间阶段相关；
4. prototype 的 assignment 是否稳定。

---

## 3. 注意事项二：soft group 的数量 $K$ 不能随便定

TACO-style 方法通常需要预设 group 数量：

$$
K
$$

这和 HiWA 里“有多少个方向簇”不同。

HiWA 猕猴实验中，如果有 8 个 reach directions，那么自然可以有：

$$
K=8
$$

但 TACO-style soft groups 不一定必须等于方向数。

### 3.1 $K$ 等于运动方向数

如果有 8 个方向，就设：

$$
K=8
$$

优点是解释简单：

> 每个 soft group 大致对应一个方向模式。

缺点是可能太粗，因为神经活动不仅编码方向，还可能编码速度、时间阶段、准备状态等。

### 3.2 $K$ 大于运动方向数

例如：

$$
K=16
$$

或者：

$$
K=32
$$

这表示每个方向内部可能有多个潜在模式。

例如：

```text
90 度方向
→ 准备期 prototype
→ 启动期 prototype
→ 执行期 prototype
→ 高速运动 prototype
```

优点是能保留更多神经结构。

缺点是解释更复杂，也更容易过拟合。

### 3.3 $K$ 小于运动方向数

例如：

$$
K=4
$$

这表示模型只学习几个粗粒度运动模式。

优点是稳定、简单。

缺点是可能丢掉方向细节。

### 3.4 建议

不要只选一个 $K$。

应该做超参数实验：

$$
K\in\{4,8,16,32\}
$$

然后比较：

- decoding accuracy；
- group correspondence matrix；
- assignment entropy；
- prototype stability；
- runtime；
- confusion matrix。

---

## 4. 注意事项三：表征空间比 OT 本身更关键

TACO 和 HiWA 都有一个共同问题：

> OT 对齐只能在你给它的空间里工作。

如果神经表征空间本身没有保留运动方向信息，那么后面的 OT 再精巧也没用。

原始 HiWA 使用 Factor Analysis：

$$
Y_{\text{neural}}
\rightarrow
Z_{\text{neural}}^{3D}
$$

但你可以尝试：

```text
Factor Analysis
PCA
UMAP
Isomap
Autoencoder
GPFA
Supervised encoder
Contrastive encoder
```

移植 TACO 时，最重要的问题之一是：

> 你要在哪个 representation space 里学习 soft prototypes？

如果表征空间差，prototype 也会差。

所以你的实验路线应该先做：

```text
不同降维 / 表征方法
→ 同一个 HiWA / TACO-style OT
→ 比较结果
```

例如：

| 方法 | 表征方式 | 是否改 OT |
|---|---|---|
| FA-HiWA | Factor Analysis | 否 |
| PCA-HiWA | PCA | 否 |
| AE-HiWA | Autoencoder | 否 |
| Soft-Prototype HiWA | soft prototype | 是 |
| Task-aware Soft-Prototype HiWA | soft prototype + 任务损失 | 是 |

---

## 5. 注意事项四：source 和 target 的维度要一致，或者明确改模型

如果你要做 OT 对齐，通常需要 source 和 target 在同一个维度空间：

$$
z_i^s\in\mathbb{R}^d
$$

$$
z_j^t\in\mathbb{R}^d
$$

如果神经表示是 10 维，而运动表示是 3 维：

$$
z_i^{neural}\in\mathbb{R}^{10}
$$

$$
z_j^{move}\in\mathbb{R}^{3}
$$

那么不能直接套原来的欧氏距离：

$$
\|z_i^{neural}-z_j^{move}\|^2
$$

因为两个向量维度不同。

你有三种选择。

### 5.1 都降到 3 维

这是最稳的：

$$
z_i^{neural}\in\mathbb{R}^3
$$

$$
z_j^{move}\in\mathbb{R}^3
$$

优点是最接近 HiWA 原实验。

### 5.2 都构造成更高维

例如都构造成 10 维：

$$
z_i^{neural}\in\mathbb{R}^{10}
$$

$$
z_j^{move}\in\mathbb{R}^{10}
$$

这需要你为 movement 侧构造更丰富的特征，例如：

$$
(v_x,v_y,\|v\|,a_x,a_y,\|a\|,\sin\theta,\cos\theta,t,\text{phase})
$$

### 5.3 学一个跨维映射

例如：

$$
W:\mathbb{R}^{10}\rightarrow\mathbb{R}^3
$$

然后：

$$
Wz_i^{neural}\approx z_j^{move}
$$

这更复杂，已经不是简单移植 TACO，而是改造对齐模型。

建议你一开始不要选这条。

---

## 6. 注意事项五：group cost 的定义决定方法本质

TACO-style 方法中，group cost 通常是 prototype 距离：

$$
D_{kl}^{group}
=
d(c_k^s,c_l^t)
$$

例如：

$$
D_{kl}^{group}
=
\|c_k^s-c_l^t\|^2
$$

而 HiWA 的 group cost 通常来自两个 cluster empirical distributions 的 OT 代价：

$$
D_{kl}^{group}
\approx
W_2^2(\mu_k^s,\nu_l^t)
$$

这就是两者最本质的区别。

迁移时你要决定采用哪一种。

### 6.1 完全 TACO-style

每个 group 只用一个 prototype：

$$
\mu_k=\delta_{c_k}
$$

优点：

- 计算简单；
- 容易端到端训练；
- 适合 soft assignment；
- 适合大规模数据。

缺点：

- 丢掉组内分布形状；
- 可能过度简化神经数据；
- 解释性不如 HiWA 的经验分布。

### 6.2 保留 HiWA-style

每个 group 仍然是一整个经验分布：

$$
\mu_k=\sum_i w_{ki}\delta_{z_i}
$$

优点：

- 保留组内结构；
- 更接近原始 HiWA；
- 对神经点云更精细。

缺点：

- 计算更重；
- soft assignment 改造更复杂。

### 6.3 折中：multi-prototype

每个 group 用多个 prototype 表示：

$$
\mu_k
\approx
\sum_{r=1}^{R}
w_{kr}\delta_{c_{kr}}
$$

这很适合你的项目。

因为它介于 TACO 和 HiWA 之间：

```text
TACO：一个 group 一个 prototype，太粗
HiWA：一个 cluster 保留所有点，太重
Multi-prototype：一个 group 几个 prototype，折中
```

这可以作为你的创新方向之一：

> Multi-Prototype Soft-HiWA。

---

## 7. 注意事项六：soft assignment 可能塌缩

soft assignment 看起来很灵活，但它可能出现塌缩。

例如所有样本都分到同一个 group：

$$
a_i=(1,0,0,\dots,0)
$$

或者所有样本平均分到所有 group：

$$
a_i=
\left(
\frac{1}{K},\frac{1}{K},\dots,\frac{1}{K}
\right)
$$

第一种问题是 group collapse。  
第二种问题是 group indistinguishability。

### 7.1 防止所有样本挤到一个 group

可以加入 group balance regularization。

定义第 $k$ 个 group 的平均质量：

$$
\alpha_k=
\frac{1}{n}
\sum_i a_{ik}
$$

理想情况下：

$$
\alpha_k\approx\frac{1}{K}
$$

可以加：

$$
\mathcal{L}_{balance}
=
\sum_{k=1}^{K}
\left(
\alpha_k-\frac{1}{K}
\right)^2
$$

### 7.2 防止 assignment 太平均

如果每个样本都平均分到所有 group，则每个 group 没有区分度。

可以控制每个样本的 assignment entropy：

$$
H(a_i)
=
-\sum_{k=1}^{K}a_{ik}\log a_{ik}
$$

如果希望每个样本有主要归属，可以降低样本级熵。

但要小心：熵太低会变成 hard cluster。

所以要在“太硬”和“太软”之间找平衡。

---

## 8. 注意事项七：group-level OT 和 instance-level OT 不一定都要立刻做

完整 TACO-style 结构是：

```text
soft assignment
→ prototype
→ group-level OT
→ group-guided instance-level OT
```

但做项目时可以分阶段。

### 8.1 第一阶段：只做 prototype-level OT

先不做 instance-level OT。

只比较：

$$
D_{kl}^{group}
=
d(c_k^{neural},c_l^{move})
$$

然后做：

$$
P^*
=
\arg\min_{P\in U(\alpha,\beta)}
\langle D^{group},P\rangle+\varepsilon H(P)
$$

看 $P^*$ 是否能恢复合理的方向对应关系。

### 8.2 第二阶段：用 $P^*$ 指导样本匹配

再构造：

$$
W=A^{neural}P^*(A^{move})^\top
$$

然后用 $W$ 修正 instance cost：

$$
\widetilde{D}_{ij}
=
D_{ij}
-
\lambda\log(W_{ij}+\delta)
$$

再做 instance-level OT。

### 8.3 第三阶段：加入任务损失

最后加入 movement direction decoding loss：

$$
\mathcal{L}_{total}
=
\mathcal{L}_{align}
+
\lambda\mathcal{L}_{decode}
$$

这样逐步推进，比较稳。

---

## 9. 注意事项八：不要把无监督任务偷偷变成监督任务

原始 HiWA 在猕猴实验中有 unsupervised alignment / decoding 的味道。

如果你在 cost 里加入真实标签：

$$
\mathbf{1}(y_i\neq y_j)
$$

那么你就使用了标签信息。

这不是不可以，但你必须说清楚：

```text
这是 supervised 或 semi-supervised 方法。
```

否则比较不公平。

### 9.1 无监督版本

只用几何信息和 soft assignment：

$$
D_{ij}=d(z_i^s,z_j^t)
$$

### 9.2 半监督版本

只用少量标签修正 cost：

$$
D_{ij}
=
d(z_i^s,z_j^t)
+
\lambda\mathbf{1}(y_i\neq y_j)
$$

### 9.3 监督版本

直接用 decoding loss：

$$
\mathcal{L}_{decode}
=
\mathrm{CE}(f(z_i),y_i)
$$

你的实验设计里必须把这三种分开。

---

## 10. 注意事项九：OT 的质量守恒假设可能不适合神经数据

标准 OT 假设两边总质量完全匹配：

$$
T\mathbf{1}=p
$$

$$
T^\top\mathbf{1}=q
$$

这表示：

```text
source 的所有质量都必须被运走
target 的所有质量都必须被填满
```

但神经数据中可能有：

- outlier；
- 缺失 trial；
- 某些方向样本不足；
- 某些 session 中没有对应模式；
- 噪声神经状态；
- 电极漂移导致的不可对应样本。

如果强行全部匹配，模型可能会被 outlier 拖偏。

所以后续可以考虑：

```text
partial OT
unbalanced OT
```

### 10.1 Partial OT

只匹配一部分质量：

$$
\sum_{i,j}T_{ij}=\rho
$$

其中：

$$
0<\rho<1
$$

表示只匹配最可靠的 $\rho$ 部分质量。

### 10.2 Unbalanced OT

允许边缘质量不完全匹配：

$$
T\mathbf{1}\neq p
$$

$$
T^\top\mathbf{1}\neq q
$$

但对偏离加惩罚。

这对跨 session 神经数据非常有意义。

---

## 11. 注意事项十：TACO 的“可移植性”不等于保留 HiWA 的所有优点

TACO-style prototype 很轻，但它会丢掉 HiWA 的一部分优势。

HiWA 的强项是：

> 一个 cluster 是经验分布，保留了簇内点云结构。

TACO-style prototype 的弱点是：

> 一个 group 被压缩成一个点，组内结构被弱化。

所以如果你的神经数据中簇内结构很重要，例如同一个方向内部有明显的时间轨迹或速度变化，那么一个 prototype 可能不够。

这时可以考虑：

1. 一个 group 多个 prototype；
2. group 内再做 local OT；
3. 保留 HiWA 的 within-cluster transport；
4. 加入 temporal / dynamics regularization。

---

## 12. 注意事项十一：时间结构不能轻易丢

猕猴伸手任务不是静态点云。它有 trial 和 time bin。

一个 trial 可能是：

```text
静止等待
→ cue 出现
→ 运动准备
→ movement onset
→ 执行运动
→ 到达目标
```

如果你把所有 time bins 打散成独立点，可能会丢掉时间结构。

TACO-style 方法迁移时可以考虑：

### 12.1 先做静态版本

把每个 time bin 当成一个点：

$$
z_i
$$

这是最简单的。

### 12.2 加入时间特征

把时间作为额外维度：

$$
z_i'=(z_i,\alpha t_i)
$$

### 12.3 分阶段对齐

分别对齐：

```text
准备期
启动期
执行期
保持期
```

### 12.4 轨迹级对齐

把一个 trial 看成一条轨迹：

$$
z_1,z_2,\dots,z_T
$$

然后做 trajectory-level alignment。

这个方向更复杂，但很有神经科学意义。

---

## 13. 注意事项十二：评价指标不能只看 accuracy

如果只看 movement direction accuracy，可能不够。

你应该设计一组指标。

### 13.1 主要指标

```text
direction decoding accuracy
```

也就是方向分类准确率。

### 13.2 混淆矩阵

看哪些方向容易混淆。

例如：

```text
90 度是否容易被误判为 45 度或 135 度
```

### 13.3 对齐代价

比较：

$$
\mathcal{L}_{group}
$$

$$
\mathcal{L}_{inst}
$$

是否下降。

### 13.4 稳定性

重复不同随机种子：

```text
seed 1
seed 2
seed 3
...
```

观察 accuracy 和 $P^*$ 是否稳定。

### 13.5 小样本表现

减少每个方向样本数：

$$
n_k\in\{5,10,20,50\}
$$

看模型是否仍然稳定。

### 13.6 鲁棒性

人为加入噪声或 outlier，测试模型是否崩掉。

### 13.7 可解释性

看 prototype 是否对应有意义的神经模式：

```text
方向
速度
时间阶段
trial phase
```

---

## 14. 注意事项十三：一定要做 ablation study

你不能只说：

> 我把 TACO 移植到了 HiWA。

你要证明每个改动有用。

建议做以下消融实验：

| 模型 | 目的 |
|---|---|
| Original HiWA | 原始基线 |
| PCA-HiWA | 检查表征空间影响 |
| FA-HiWA | 对齐原论文 |
| GMM-HiWA | 检查自动聚类影响 |
| Soft Assignment HiWA | 检查 soft group 是否有用 |
| Prototype-HiWA | 检查 prototype 表示是否有用 |
| Multi-Prototype HiWA | 检查多个 prototype 是否更好 |
| Task-aware Soft-HiWA | 检查任务损失是否提升解码 |
| Unbalanced Soft-HiWA | 检查噪声和缺失模式鲁棒性 |

这样你的研究就不是“换了一个公式”，而是系统回答：

> 神经对齐中，表征、分组、prototype、OT 约束分别有多重要？

---

## 15. 最推荐的迁移路线

不建议一步到位做完整 TACO。

更稳的路线是：

### 阶段一：复现 HiWA

目标：

```text
跑通官方神经元实验
理解 X、Y、T、P、R、accuracy
```

你需要记录：

- 数据 shape；
- 几个 movement directions；
- 每个方向样本数；
- FA 后的 3D 分布；
- 对齐前后 accuracy；
- $P$ 是否接近合理 correspondence；
- 结果是否稳定。

### 阶段二：替换神经表征方法

保持 HiWA 主体不变，只换：

$$
\text{Factor Analysis}
$$

例如：

```text
PCA
UMAP
Autoencoder
GPFA
```

目标：

> 证明表征空间选择对 HiWA 很重要。

### 阶段三：引入 soft assignment

从 hard direction cluster 改为：

$$
a_{ik}\in[0,1]
$$

可以先用 GMM posterior 得到 soft assignment。

目标：

> 证明神经活动更适合 soft group，而不是 hard cluster。

### 阶段四：引入 prototype-level OT

计算 neural prototypes 和 movement prototypes：

$$
c_k^{neural}
$$

$$
c_l^{move}
$$

做：

$$
P^*
=
\arg\min_{P\in U(\alpha,\beta)}
\langle D^{group},P\rangle+\varepsilon H(P)
$$

目标：

> 验证 prototype-level group correspondence 是否能恢复方向结构。

### 阶段五：加入 instance-level OT

构造：

$$
W=A^{neural}P^*(A^{move})^\top
$$

再做 group-guided instance OT。

目标：

> 检查 group-level 信息是否能改善样本级对齐。

### 阶段六：加入任务损失

加入 decoding loss：

$$
\mathcal{L}_{decode}
$$

目标：

> 从“对齐得好”推进到“神经解码更准”。

---

## 16. 可以形成的研究问题

你可以把项目问题写成几个清楚的问题。

### 问题一

> HiWA 的性能是否依赖 neural latent representation 的选择？

对应实验：

```text
FA vs PCA vs UMAP vs Autoencoder
```

### 问题二

> hard cluster 是否限制了神经活动对齐？

对应实验：

```text
hard direction label vs soft assignment
```

### 问题三

> prototype-level group OT 是否能替代或补充 HiWA 的 cluster-level OT？

对应实验：

```text
HiWA distribution cluster vs TACO-style prototype group
```

### 问题四

> multi-prototype 是否比 single-prototype 更适合神经数据？

对应实验：

```text
1 prototype per group vs multiple prototypes per group
```

### 问题五

> task-aware alignment 是否提升 movement direction decoding？

对应实验：

```text
alignment-only vs alignment + decoding loss
```

---

## 17. 最后总结

TACO 的方法确实是可移植的，因为它提供了一种通用结构：

$$
\text{soft assignment}
\rightarrow
\text{prototype}
\rightarrow
\text{group OT}
\rightarrow
\text{instance OT}
$$

但是移植到 HiWA / 猕猴神经元实验时，必须注意：

1. **表征空间要重新设计**  
   神经数据不是文本数据，不能照搬 encoder。

2. **prototype 不是生理实体**  
   它只是 soft group 的数学代表点，需要后续解释验证。

3. **group 数量 $K$ 是关键超参数**  
   不能随便定，要系统实验。

4. **source 和 target 维度必须一致**  
   除非你显式改成跨维映射模型。

5. **group cost 决定方法本质**  
   TACO 是 prototype distance；HiWA 是 distribution-level cost。

6. **soft assignment 可能塌缩**  
   需要 balance 或 entropy 控制。

7. **不要偷用标签却声称无监督**  
   task-aware 方法要和 unsupervised 方法分开比较。

8. **标准 OT 的质量守恒可能不适合真实神经数据**  
   可以考虑 partial OT 或 unbalanced OT。

9. **时间结构可能很重要**  
   trial 和 time bin 不能长期忽略。

10. **必须做 ablation study**  
   否则无法证明你的改动真的有意义。

最适合你的创新主线可以写成：

> 原始 HiWA 依赖 hard cluster 和 cluster empirical distribution。受 TACO 的 soft-prototype hierarchical OT 启发，我们引入 soft assignment 与 prototype-level group alignment，用于构造更灵活的神经活动模式对齐方法，并进一步研究其对猕猴运动方向解码的影响。

用一句话概括：

> TACO 可以移植，但不要照搬；你真正要移植的是 soft group、prototype 和 group-guided instance alignment 这套思想，而不是某个固定公式。
