---
title: "GC-HiWA v2 完整研究笔记"
aliases:
  - "GC-HiWA v2"
  - "适用性诊断的无监督坐标对齐"
tags:
  - research/ot
  - research/pose
  - method/gc-hiwa
  - status/current
created: 2026-07-25
status: "active-paper-draft"
---

# GC-HiWA v2：完整研究笔记

> [!abstract] 当前的一句话结论
>
> GC-HiWA v2 不是“让 OT 在所有跨域任务上提升”的方法。它是一个带适用性诊断的无监督坐标对齐器：当两域共享群组几何、且坐标近似只相差一个预先声明的结构化正交变换时恢复该变换；当条件不成立时，输出弃权与原因。

> [!warning] 不可写成的结论
>
> 现有证据**不支持**“GC-HiWA 已普遍提升真实跨设备动作识别”“低 OT 代价等于语义迁移成功”“ROCA 在所有数据上都能选对分支”。真实语义动作迁移的 UWA3DII 冻结实验尚未运行。

## 0. 导航与当前材料

| 用途 | 当前文件 |
|---|---|
| 论文主稿 | `Writing/GC-HiWA_v2_论文草稿.tex` |
| 编译 PDF | `Writing/GC-HiWA_v2_final.pdf` |
| 论文的数学合同与结果 | [[GC-HiWA_v2_数学合同、适用性诊断与Panoptic验证_2026-07-23]] |
| 最终质量边界 | `Writing/终稿审查清单.md` |
| 一键证据审计 | `experiments/synthetic/verify_gc_hiwa_v2_evidence.py` |
| 真实语义数据的冻结协议 | [[真实语义动作验证的数据接入与冻结协议_2026-07-25]] |
| 历史 ROCA 草稿 | `Writing/ROCA-HiWA_paper.tex`，已标为 archived，不可作为当前主结论 |

### 0.1 已完成的工作、证据与边界

下表回答“项目已经做了什么”。“完成”表示代码、协议和可复核输出都已存在；它不把尚未完成的真实语义迁移实验写成结果。

| 工作包 | 已完成内容 | 可核查结果 | 当前边界 |
|---|---|---|---|
| 几何合同 | 将任务限定为预先声明的结构化正交族；对姿态固定为 $I_J\otimes R_3,\ R_3\in SO(3)$ | 数学合同、投影和资格检查已进入实现 | 不适用于任意线性域适配或非刚体跨域变化 |
| 无监督拟合器 | 实现软群组、组内加权 Sinkhorn、组间熵正则 OT、局部--全局 ADMM 与结构化旋转投影 | `SoftHiWA` 可在无配对、无标签条件下输出旋转或弃权 | ADMM 收敛不是全局最优或语义迁移证明 |
| 适用性诊断 | 实现局部 OT 边际误差、正交误差、单旋转拟合比、谱差、可识别性间隔与多启动一致性 | 任一硬门槛失败返回明确的 `abstain` 原因 | 固定阈值是工程合同，不是普适的统计显著性定理 |
| ROCA | 实现 $O(3)$ 的正／负行列式候选、定向四面体符号、候选资格比较与模糊弃权 | 合成的可恢复情形 15/15 选对；违例不强行选分支 | 只适合有 $d+1$ 个稳定群组的诊断分支；姿态主干只用 $SO(3)$ |
| 合成边界实验 | 建立理想旋转、混合旋转、群组平移、非正交、独立跨模态与谱伪装非正交的生成器和独立种子协议 | 15/15 可恢复接受；25/25 预声明违例弃权；谱伪装 5/5 弃权 | 这是机制验证，不是现实动作识别表现 |
| Panoptic 受控几何 | 以隐藏相机外参和同步帧做最终评分；拟合阶段完全屏蔽这些信息 | 三组共 11 次均收敛；最大旋转误差 0.0308；Recall@1=1 | 同一采集系统下的相机几何恢复，不能外推为跨设备语义泛化 |
| 冻结下游代理 | 源端无人工状态聚类后训练分类器；只在评分时使用隐藏同步帧得到目标真值 | 一个窗口由 0.333 恢复至 0.864；另一窗口为 0.667 至 0.811 | 状态不是人工动作标签，不能称为动作识别结果 |
| 负结果与范围审计 | 审计 Indy-Loco、PBMC、PAMAP2、Paderborn 及早期 MiHiA/TACO-faithful 路线 | 记录为何几何假设、切分或独立性不成立 | 负结果界定范围，不构成这些数据集上的正向结果 |
| 交付与复现 | 完成论文主稿、Obsidian 笔记、证据审计器、回归测试和关键实现索引 | 全测试 `114 passed`；一键审计输出冻结结果 | UWA3DII 完整数据尚未到位，真实跨设备语义实验尚未运行 |

因此，当前最强的完成结论是：**GC-HiWA 已成为一个可复核的、带弃权机制的无监督坐标恢复方法，并已在合成边界和受控真实相机几何上验证。** 尚未完成的工作是按冻结协议完成 UWA3DII 的真实动作语义迁移实验。

## 1. 问题从哪里来

### 1.1 实际应用

设源设备已有动作识别或姿态分析模型；目标设备从不同相机／设备坐标系产生三维姿态。目标端没有：

- 相机外参或人工标定；
- 与源端同步的配对帧；
- 可用于拟合的动作标签。

我们希望只用两域无标签姿态恢复目标坐标到源坐标的关系，使源模型可以在目标端使用。

### 1.2 为什么不是任意线性域适配

若一帧有 $J$ 个三维关节，向量维度是 $d=3J$。一个物理相机坐标变化对每个关节施加同一三维旋转 $R_3$，因此允许族是

$$
\mathcal R_{\mathrm{pose}}
=\{I_J\otimes R_3:R_3\in SO(3)\}.
$$

它只有三个自由度。若放宽为 $O(3J)$，优化器可以把左腕的坐标混入右膝；虽然数值上仍正交，却不再表示真实相机坐标关系。这是本项目最重要的建模约束之一。

### 1.3 正确的问题与错误的问题

| 问题 | 本项目的回答 |
|---|---|
| 两域是否一定能由 OT 对齐？ | 否。OT 总能给出耦合，但不能创造不存在的坐标关系。 |
| 收敛是否等于可迁移？ | 否。ADMM 收敛只表示固定点。 |
| 是否要求恢复每个目标样本的配对？ | 拟合阶段不要求，也不使用配对；受控实验仅事后计算 Recall。 |
| 若结构不成立怎么办？ | 弃权；记录哪一项检查失败。 |

## 2. 完整数学合同

### 2.1 无配对生成模型

源域和目标域样本分别为

$$
X=\{x_i\}_{i=1}^n\subset\mathbb R^d,\qquad
Y=\{y_j\}_{j=1}^m\subset\mathbb R^d.
$$

为避免行／列向量歧义，后文把样本堆叠为矩阵 $X\in\mathbb R^{n\times d}$、$Y\in\mathbb R^{m\times d}$，每一行是一个样本。平移在标准化时被去除；下式只用来陈述理想的潜在对应，不是拟合器的输入。

理想情况下存在未观察的置换 $\pi$，使得

$$
y_{\pi(i)}=x_iR_\star^\top+\varepsilon_i,
\qquad R_\star\in\mathcal R.
$$

这里：

- $\pi$ 不提供给拟合器；
- $\varepsilon_i$ 是测量噪声或轻微模型失配；
- $\mathcal R$ 在实验前固定；姿态使用 $\mathcal R_{\mathrm{pose}}$；
- 群组结构必须在两域都存在且可被无监督地匹配。

若样本数不同或不存在一一对应，上式中的 $\pi$ 不再有定义；这正是后文用带边际约束的软耦合 $Q$ 取代 $\pi$ 的原因。方法从未把 $\pi$、动作标签、相机外参或同步帧传给优化器。

### 2.2 旋转等变的标准化

逐维 z-score 需要每一坐标轴独立缩放，因而一般不满足未知旋转等变性。GC-HiWA 使用整体标量标准化：

$$
\mathcal{S}(X)=\frac{X-\mathbf{1}\bar x^\top}{s_X},
\qquad
s_X=\sqrt{\frac{1}{nd}\left\|X-\mathbf{1}\bar x^\top\right\|_F^2}.
$$

若 $Y=XR^\top+\mathbf{1}t^\top$、$R\in O(d)$，则

$$
\bar y=\bar xR^\top+t,
\qquad
\left\|(X-\mathbf{1}\bar x^\top)R^\top\right\|_F
=\left\|X-\mathbf{1}\bar x^\top\right\|_F,
$$

所以 $s_Y=s_X$ 且

$$
\boxed{\mathcal{S}(Y)=\mathcal{S}(X)R^\top.}
$$

推导只用到正交矩阵保持 Frobenius 范数。令 $X_c=X-\mathbf{1}\bar x^\top$。由 $Y=XR^\top+\mathbf{1}t^\top$ 得

$$
\bar y=\bar xR^\top+t,
\qquad
Y_c=Y-\mathbf{1}\bar y^\top=X_cR^\top.
$$

再由 $R^\top R=I$，

$$
s_Y^2
=\frac{1}{nd}\lVert X_cR^\top\rVert_F^2
=\frac{1}{nd}\operatorname{tr}(RX_c^\top X_cR^\top)
=\frac{1}{nd}\operatorname{tr}(X_c^\top X_c)
=s_X^2.
$$

取正的尺度并代回，便得到上面的等变式。这个推导也说明为什么主实验不能使用逐坐标缩放：若 $D$ 是不等的对角缩放，通常 $DR^\top\ne R^\top D$，旋转关系会在预处理阶段被破坏。

代码中这对应 `scaling_mode="global_scalar"`。`per_feature` 模式可用于非几何任务，但不可以支撑未知坐标旋转恢复的主张。

### 2.3 软群组与经验测度

对每个域学习 $K$ 个原型。源端样本对群组的软归属为 $a_{ik}$，目标端为 $b_{j\ell}$，并满足

$$
a_{ik}\ge0,\quad \sum_{k=1}^K a_{ik}=1,
\qquad
b_{j\ell}\ge0,\quad \sum_{\ell=1}^K b_{j\ell}=1.
$$

实现中先在标准化空间学习原型 $c_k$，再以原型相似度产生归属：

$$
a_{ik}=
\frac{\exp\!\left(x_i c_k^\top/\tau_g\right)}
{\sum_{r=1}^K\exp\!\left(x_i c_r^\top/\tau_g\right)},
\qquad \tau_g>0.
$$

原型通过“重构误差减去熵奖励再加 $\ell_2$ 正则”的目标学习：

$$
\mathcal{L}_{\mathrm{group}}
=\frac{1}{nd}\lVert AC-X\rVert_F^2
-\lambda_H\frac{1}{n}\sum_i H(a_i)
+\lambda_C\frac{1}{Kd}\lVert C\rVert_F^2,
\qquad
H(a_i)=-\sum_k a_{ik}\log a_{ik}.
$$

这一步只构造可软匹配的局部经验测度，不使用类别标签。温度 $\tau_g$ 控制归属的尖锐程度；熵项避免过早退化为硬分组。

归属诱导每个组的加权经验测度：

$$
\alpha_k(i)=\frac{a_{ik}}{\sum_r a_{rk}},\qquad
\mu_k^X=\sum_i\alpha_k(i)\delta_{x_i},
$$

目标域同理。`full support` 表示每个组保留所有样本，只是权重不同；它避免硬截断把样本是否存在于某组变成隐含超参数。

由于 $\sum_i\alpha_k(i)=1$，$\mu_k^X$ 是概率测度；同样 $\sum_j\beta_\ell(j)=1$。因此每一个局部 OT 都有明确、可检查的边际，而不是从硬阈值后的不等样本子集临时构造。

### 2.4 两层 OT

对每个源组 $k$ 与目标组 $\ell$，局部样本传输 $Q_{k\ell}\in\mathbb R_+^{n\times m}$ 的可行集合为

$$
\mathcal{U}(\alpha_k,\beta_\ell)
=\left\{Q\ge0:
Q\mathbf{1}_m=\alpha_k,\quad
Q^\top\mathbf{1}_n=\beta_\ell
\right\}.
$$

给定局部旋转 $R_{k\ell}$，局部熵正则 OT 是

$$
Q_{k\ell}^\star(R_{k\ell})
=\operatorname*{arg\,min}_{Q\in\mathcal{U}(\alpha_k,\beta_\ell)}
\left\langle Q,D(R_{k\ell})\right\rangle
+\varepsilon\sum_{ij}Q_{ij}(\log Q_{ij}-1),
$$

其中 $D_{ij}(R)=\lVert x_iR^\top-y_j\rVert_2^2$，$\varepsilon>0$ 是局部 Sinkhorn 温度。若局部旋转为 $R_{k\ell}$，未正则化的运输成本是

$$
C_{k\ell}(R_{k\ell})
=\left\langle Q_{k\ell},
\left[\|x_iR_{k\ell}^\top-y_j\|_2^2\right]_{ij}\right\rangle.
$$

在固定 $Q_{k\ell}$ 时，旋转子问题可化为正交 Procrustes。记

$$
M_{k\ell}=Y^\top Q_{k\ell}^\top X.
$$

因为 $R$ 正交，平方项中的 $\sum_{ij}Q_{ij}\lVert x_iR^\top\rVert^2$ 与 $R$ 无关，于是

$$
\begin{aligned}
\left\langle Q_{k\ell},D(R)\right\rangle
&=\mathrm{const}-2\sum_{ij}(Q_{k\ell})_{ij}x_iR^\top y_j^\top\\
&=\mathrm{const}-2\operatorname{tr}(R^\top M_{k\ell}).
\end{aligned}
$$

令 $M_{k\ell}=U\Sigma V^\top$ 为 SVD。无行列式限制时，最大化 $\operatorname{tr}(R^\top M_{k\ell})$ 的解为 $R=UV^\top$；若约束 $R\in SO(d)$，则

$$
R=U\operatorname{diag}\!\left(1,\ldots,1,\det(UV^\top)\right)V^\top.
$$

这给出“固定耦合更新旋转、固定旋转更新耦合”的每一步来源；整体问题仍非凸，因而需要资格检查与多启动，而不能把交替下降误写成全局恢复保证。

组层传输 $P$ 用熵正则 OT 求得：

$$
P_\gamma=
\operatorname*{arg\,min}_{P\in\mathcal{U}_K}
\langle P,C\rangle+
\gamma\sum_{k,\ell}P_{k\ell}(\log P_{k\ell}-1),
$$

其中 $\mathcal U_K$ 约束行、列边际均匀。有限熵下 $P_\gamma$ 通常全支持，所以它不是严格置换；只有当正确匹配有正成本间隔且 $\gamma\downarrow0$ 时，才趋近正确置换集合。

具体地，组层可行集为

$$
\mathcal{U}_K=
\left\{P\ge0:
P\mathbf{1}_K=\frac{1}{K}\mathbf{1}_K,
P^\top\mathbf{1}_K=\frac{1}{K}\mathbf{1}_K\right\}.
$$

因此 $P_{k\ell}$ 同时承担两种作用：它软匹配群组，并在采用 `transport` 共识配置时决定该组对在全局旋转更新中的权重。

### 2.5 ADMM 局部--全局共识

局部组对各自拟合 $R_{k\ell}$，但物理模型要求所有局部解同意一个全局 $R_g$。把每个局部代价记为 $f_{k\ell}$，可写为带共识约束的问题：

$$
\min_{\{R_{k\ell}\},R_g}
\sum_{k,\ell}w_{k\ell}f_{k\ell}(R_{k\ell})
+\iota_{\mathcal{R}}(R_g)
\quad\text{s.t.}\quad R_{k\ell}=R_g.
$$

这里 $\iota_{\mathcal{R}}$ 是允许旋转族的指示函数；受控姿态实验的 `transport` 配置使用归一化 $w_{k\ell}\propto P_{k\ell}$。其 scaled augmented Lagrangian 为

$$
\mathcal{L}_\mu=
\sum_{k,\ell}w_{k\ell}
\left[f_{k\ell}(R_{k\ell})
+\frac{\mu}{2}\lVert R_{k\ell}-R_g+\Lambda_{k\ell}\rVert_F^2\right]
+\iota_{\mathcal{R}}(R_g).
$$

对 $R_g$ 最小化二次项，得到先取加权平均、再投影的闭式更新：

$$
R_g\leftarrow
\operatorname{Proj}_{\mathcal{R}}
\left(\sum_{k,\ell}w_{k\ell}(R_{k\ell}+\Lambda_{k\ell})\right),
\qquad
\Lambda_{k\ell}\leftarrow\Lambda_{k\ell}+R_{k\ell}-R_g.
$$

对姿态，令 $A_{uv}\in\mathbb R^{3\times3}$ 为矩阵 $A$ 的第 $(u,v)$ 个关节块，则

$$
\operatorname{Proj}_{\mathcal R_{\mathrm{pose}}}(A)
=I_J\otimes
\operatorname{Proj}_{SO(3)}\left(\sum_{u=1}^J A_{uu}\right).
$$

该投影丢弃跨关节块，并将对角关节块的证据聚合为一个 $3\times3$ 物理相机旋转。

其推导是对任意 $A$ 解

$$
\operatorname*{arg\,min}_{R_3\in SO(3)}
\left\lVert A-I_J\otimes R_3\right\rVert_F^2.
$$

非对角关节块不含 $R_3$，只贡献常数；展开对角块后，问题等价于最大化

$$
\operatorname{tr}\!\left(R_3^\top\sum_{u=1}^J A_{uu}\right),
$$

故对 $\sum_u A_{uu}$ 做一次 $SO(3)$ Procrustes 投影即可。这解释了为什么输出不能把不同关节混成不同旋转：模型空间中只允许一个共享的 $R_3$。

## 3. ROCA：行列式分支的保守选择

### 3.1 为什么存在两个分支

一般正交群 $O(3)$ 有两个连通分支：

$$
\det(R)=+1 \quad\text{和}\quad \det(R)=-1.
$$

前者是保手性的真正旋转，后者包含反射。欧氏 OT 代价可能无法区分一个构型与其镜像；因此不能仅把较低 transport objective 当作正确分支。

### 3.2 代表元与定向四面体

软代表元为

$$
r_k^X=\frac{\sum_i a_{ik}x_i}{\sum_i a_{ik}},
\qquad
r_\ell^Y=\frac{\sum_j b_{j\ell}y_j}{\sum_j b_{j\ell}}.
$$

对于 $d=3,K=4$，匹配后的四个代表元构成四面体。以第一个代表元为锚点：

$$
V_X=\det\left[r_2^X-r_1^X, r_3^X-r_1^X, r_4^X-r_1^X\right].
$$

目标域得到 $V_Y$。若 $Y\approx XR^\top$，则体积符号满足

$$
\operatorname{sign}(V_XV_Y)\approx\det(R).
$$

推导使用列向量表示边：令

$$
B_X=
\left[(r_2^X-r_1^X)^\top,
(r_3^X-r_1^X)^\top,
(r_4^X-r_1^X)^\top\right].
$$

若匹配后的代表元严格满足 $r_k^Y=r_k^XR^\top$，则每条列边满足 $(r_k^Y-r_1^Y)^\top=R(r_k^X-r_1^X)^\top$，从而 $B_Y=RB_X$。因此

$$
V_Y=\det(B_Y)=\det(R)\det(B_X)=\det(R)V_X.
$$

当 $|V_X|$ 或 $|V_Y|$ 很小时，符号会被噪声支配；ROCA 在这种退化情形弃权，而不是把不稳定符号当成反射证据。

### 3.3 现在的 ROCA 不做什么

当前实现不是“只要四面体符号匹配就选分支”。每个候选先独立接受数值与几何资格检查；两个候选若都合格，只有内部目标有预定义充分间隔才选择，否则输出 `ambiguous_multiple_qualified_candidates`。若代表元单纯形退化、谱不兼容或几何不可识别，则弃权。

> [!note] 适用范围
>
> 定向单纯形 ROCA 严格需要 $d+1$ 个稳定群组。当前三维姿态的主干本身不需要 ROCA，因为物理相机坐标变化已限制为 $SO(3)$；ROCA 是一般 $O(3)$ 合成诊断分支，而不是对姿态任务强加反射候选。

## 4. 何时可恢复，何时不可恢复

### 4.1 已知正确耦合下的唯一性

若正确加权耦合 $Q_\star$ 已知、模型无噪声且源的加权协方差满秩，则

$$
\hat R=\operatorname*{arg\,min}_{R\in\mathcal{R}}
\sum_{ij}(Q_\star)_{ij}\|x_iR^\top-y_j\|_2^2
$$

的唯一最优解为 $R_\star$。

更完整地说，令中心化后的源样本为 $z_i$，并定义正确耦合诱导的 Gram 矩阵

$$
G=\sum_{i,j}(Q_\star)_{ij}z_i^\top z_i.
$$

“加权协方差满秩”即 $G\succ0$。因为无噪声且耦合正确，耦合支持上的目标点满足 $y_j=z_iR_\star^\top$，所以 $R_\star$ 的目标值为零。若另一个 $R$ 也达到零，则非负项之和为零，故对所有 $(Q_\star)_{ij}>0$ 有

$$
z_i(R-R_\star)^\top=0.
$$

两边平方、按耦合权重求和：

$$
\begin{aligned}
0
&=\sum_{ij}(Q_\star)_{ij}
\left\lVert z_i(R-R_\star)^\top\right\rVert_2^2\\
&=\operatorname{tr}\!\left((R-R_\star)^\top G(R-R_\star)\right).
\end{aligned}
$$

由于 $G\succ0$，上式仅在 $R-R_\star=0$ 时为零，故 $R=R_\star$。这只证明“给定正确耦合”时的唯一性，**不**证明 OT--ADMM 总能找到该耦合，更不证明无标签数据一定具有这样的耦合。

### 4.2 不存在精确解的定义

定义允许族下最小配对残差

$$
\rho_{\mathcal R}^2=
\inf_{R\in\mathcal R,\pi}
\frac1n\sum_i\|x_iR^\top-y_{\pi(i)}\|_2^2.
$$

若 $\rho_{\mathcal R}>0$，允许族内不存在精确解。多旋转混合、群组平移、显著非正交缩放和独立跨模态目标都属于这个范畴。没有无监督算法能从数据中“恢复”一个并不存在的共同旋转。

### 4.3 谱门槛与可识别性门槛

中心化并 Frobenius 归一化的协方差为

$$
\widetilde\Sigma_X=\frac{X_c^\top X_c}{\|X_c\|_F^2}.
$$

正交关系必然保持特征值，因此定义

$$
\delta_{\mathrm{spec}}=
\frac{\|\lambda(\widetilde\Sigma_X)-\lambda(\widetilde\Sigma_Y)\|_2}
{(\|\lambda(\widetilde\Sigma_X)\|_2+\|\lambda(\widetilde\Sigma_Y)\|_2)/2}.
$$

必要性的推导很直接。若中心化、整体缩放后的数据精确满足 $Y_c=X_cR^\top$，则

$$
\widetilde\Sigma_Y
=\frac{Y_c^\top Y_c}{\lVert Y_c\rVert_F^2}
=\frac{RX_c^\top X_cR^\top}{\lVert X_c\rVert_F^2}
=R\widetilde\Sigma_XR^\top.
$$

这是相似变换，故两者特征值完全相同，理想情况下 $\delta_{\mathrm{spec}}=0$。反过来，谱相同只说明存在某种保持该二阶结构的变换；它既不保证样本级耦合存在，也不保证该变换是共同的物理旋转。这正是谱伪装反例仍会被其他门槛拒绝的数学原因。

若它大，合同必然不成立；若它小，合同仍可能不成立。为了识别协方差的近轴对称／近球对称退化，定义

$$
\kappa_{\mathrm{id}}=
\frac{\min_r(\lambda_{r+1}-\lambda_r)}{\|\lambda(\widetilde\Sigma_X)\|_2}.
$$

当 $\kappa_{\mathrm{id}}=0$，仅依靠二阶统计量不能区分重特征子空间内的旋转。小 $\kappa_{\mathrm{id}}$ 不证明完整分布没有高阶可识别信息，但在没有标签与配对时，是保守弃权的充分理由。

更具体地，若 $\widetilde\Sigma_X=U\Lambda U^\top$ 含有重复特征值，则在对应特征子空间内任取正交矩阵 $H$ 都有 $H\Lambda H^\top=\Lambda$。二阶统计无法区分这些 $H$，所以由协方差导出的旋转不唯一。$\kappa_{\mathrm{id}}$ 用最小相邻谱间隔量化这种不稳定性；它是“证据不足时不输出”的诊断量，而不是把高阶可识别性宣判为不存在的定理。

## 5. 部署时的资格检查

| 检查 | 当前冻结阈值 | 失败含义 |
|---|---:|---|
| ADMM 收敛 | 必须为真 | 没有可靠数值固定点 |
| 局部 Sinkhorn 边际误差 | $\le 10^{-3}$ | 局部 OT 不可行／不可靠 |
| 正交误差 | $\le10^{-6}$ | 输出不在声明的允许族 |
| 单一旋转拟合比 $\eta$ | $\le1.5$ | 跨域拟合明显差于域内自耦合 |
| 谱差 $\delta_{\mathrm{spec}}$ | $\le 0.05$ | 违反必要的正交不变量 |
| 几何间隔 $\kappa_{\mathrm{id}}$ | $\ge 0.02$ | 输出的唯一旋转不稳定 |
| 姿态结构 | `repeated_3d_blocks` | 输出不是一个共享相机旋转 |
| 多启动 | 至少 3 次；$80\%$ 在 $5^\circ$ 内 | 优化器局部解不稳定 |

拟合比的定义是先由 $P$ 得到一一组匹配，再把跨域匹配成本与两个域的域内自耦合成本比较：

$$
\eta=
\frac{L_{\mathrm{cross}}}
{\frac{1}{2}(L_{X,\mathrm{self}}+L_{Y,\mathrm{self}})}.
$$

最终输出不是无条件的 $\hat R$，而是一个资格化决策。令 $g_1,\ldots,g_s$ 分别表示上表中的硬检查，$\mathcal{C}$ 表示多启动形成的稳定簇，定义

$$
\operatorname{Decision}(X,Y)=
\begin{cases}
(\mathrm{accept},\hat R),
& \bigwedge_{r=1}^{s}g_r(X,Y,\hat R)=1
\ \land\ |\mathcal{C}|/N_{\mathrm{restart}}\ge0.8,\\
(\mathrm{abstain},\mathcal{E}), & \text{otherwise},
\end{cases}
$$

其中 $\mathcal{E}$ 是失败检查的名称集合。这个规则的逻辑方向很重要：接受表示“没有发现与合同冲突的证据”，不表示已经数学证明两域语义等价；弃权则是模型的有效输出，而不是程序错误。

任何一项失败均返回 `abstain`。接受只表示“当前检查未发现合同冲突”，不等于语义等价证明。

## 6. 实验：哪些结果支持什么

### 6.1 合成边界

拟合器看不到群组标签、配对、真实旋转和真实 determinant；这些量只在事后算旋转误差、MSE、ARI 和选择正确性。

| 独立实验 | 结果 | 能支持的结论 |
|---|---|---|
| 种子 1961--1965：理想旋转、噪声 0.05、噪声 0.10 | 核心 15/15 接受；ROCA 15/15 接受并选对分支 | 合同满足时，固定主干可恢复 |
| 同一独立批次：混合旋转、群组平移、非正交、独立跨模态 | 核心与 ROCA 均 25/25 弃权 | 当前已设计的违例可被组合门槛阻断 |
| 种子 2201--2205：谱伪装非正交 | 核心与 ROCA 均 5/5 弃权 | 谱检查不是充分条件；拟合与可行性检查仍必要 |

谱伪装构造为

$$
A=S^{1/2}QS^{-1/2},\quad S=X_c^\top X_c,\quad Q\in SO(3).
$$

它满足 $ASA^\top=S$，故两域谱差最大仅 $1.16\times10^{-15}$，但一般 $A\notin O(3)$。五次中 $\eta$ 在 4.34--12.36，核心门槛因为非收敛和 `single_rotation_fit_excessive` 弃权。这个反例防止我们把“谱相同”误读成“存在共同旋转”。

### 6.2 自动群组数是否值得加进模型

曾测试固定 $K=4$ 与事后知道真 $K$ 的 oracle 控制，真群数覆盖 2--6。五群组的首次小样本筛查看似给 oracle 带来优势，但独立种子 2104--2108 中固定 $K=4$ 的平均旋转误差为 0.0109，而 oracle $K=5$ 有一次错误局部解、平均误差为 0.5429。不存在稳定、可重复的缺口。

结论：当前**不**引入截断 DP / DP-GMM 自动聚类。它会改变群组 OT 的边际与 ROCA 的 $d+1$ 单纯形前提，却没有可靠收益。这是主动停止无价值复杂化，而不是未完成的功能。

### 6.3 Panoptic Studio：受控的真实几何

Panoptic 使用多相机三维骨架。实验处理为：

1. 每帧 19 个关节按置信度中心化，消除平移；
2. 转到两个相机坐标系；
3. 用共同标量缩放；
4. 打乱目标帧；
5. 拟合时屏蔽同步帧、相机外参、ID 和标签；
6. 只在最终评分时打开这些真值。

| 设置 | 数值结果 | 解释 |
|---|---|---|
| 使用无结构 $O(57)$ | 100 轮未收敛；旋转误差 10.31 | 高维任意正交不代表相机变换 |
| pose1 00\_00 $\to$ 00\_01，5 次 | 5/5 收敛；平均误差 0.0154；最大 0.0254；Recall@1=1 | 结构约束后恢复 |
| pose1 00\_00 $\to$ 00\_05 与 00\_21，各 3 次 | 6/6 收敛；最大误差 0.0308；Recall@1=1 | 不依赖单一相机对 |
| 三组共 11 次 | 全部收敛；最大 MSE $3.23\times10^{-6}$ | 受控多配置一致 |
| 加入新资格门槛的 seed 1601 | 误差 0.00818；MSE $2.79\times10^{-7}$；$\kappa_{id}=0.133$ | 新门槛没有破坏有效恢复 |

### 6.4 多启动为什么必要

在 pose2 独立窗口，五次运行中四次形成 $5^\circ$ 内稳定簇；代表解误差 0.00575、Recall@1=1。剩下一次虽然单次门槛局部通过，但旋转误差 0.716、Recall@1=0.523，且与稳定簇约差 $6.7^\circ$，因此被多启动规则排除。

这个结果不能被简化为“多跑几次选最好”。选择时没有使用真旋转或 Recall，而是只用无标签重启之间的物理旋转一致性。

### 6.5 冻结源模型的端到端代理

源端按时间划分，先对源姿态聚出四个状态并训练逻辑回归；目标状态标签只在最终评分时由隐藏同步帧获得。

| 窗口 | 源留出平衡准确率 | 目标原坐标 | 目标经 GC-HiWA |
|---|---:|---:|---:|
| 主体 0，帧 141--339 | 0.864 | 0.333 | 0.864 |
| 主体 2，帧 10000--10099 | 0.822 | 0.667 | 0.811 |

这是“源坐标模型经过恢复的坐标后能重新使用”的受控证据。状态来自源姿态，不是人工动作语义，所以不能被升级为跨设备动作识别结果。

## 7. 其他数据集：做过什么，为什么不作为主结果

| 数据／方向 | 已做工作 | 结果与正确定位 |
|---|---|---|
| MiHiA / TACO-faithful 神经--运动 | 软群组、Hard/Soft HiWA、早期 ROCA 开发与跨子样本审计 | 有开发集信号，但独立性不足；仅作方法开发史，不作 v2 主证据 |
| Indy-Loco 跨会话神经--速度 | 对比 source-only、Hard、Soft-GCOT 与显式 paired Procrustes oracle | oracle 也只能达到很低的 $R^2$；全局正交几何不适用，是重要反证 |
| PBMC RNA--ATAC | 公共特征空间、Hard/Soft 对齐、paired recall | 共享特征不等于近似正交几何；当前低 recall，不能作正向例子 |
| PAMAP2 传感器 | 单受试者时间域安全协议、task-aware OT 与弱配对 | task-aware OT 无稳定收益；弱配对的单受试者信号不能泛化 |
| Paderborn 轴承 | 工况／实体切分审计 | 一种切分太容易，另一种没有信号；实验任务本身不适合验证方法 |

> [!important] 对负结果的用法
>
> 这些数据不是“失败后删掉”的材料。它们界定了假设边界：当表示不共享正交几何，调 Sinkhorn、加深编码器或增大 OT 预算都不能将其变成坐标恢复问题。

## 8. 联合优化、编码器与原型更新为什么不进入主结果

项目曾实现或讨论编码器、交替联合优化、原型反向更新和 task-aware OT。它们可以在数据上进行更灵活的表示变换，但也会混淆两件事：

1. 是否已经从原坐标中恢复了物理坐标关系；
2. 是否通过学习表示让某个下游任务变容易。

现有实验没有显示这些附加模块对主任务有稳定、独立的增益。因此 v2 主干刻意固定为无学习编码器、无联合原型更新的几何版本。以后若重启该方向，必须先在已知真旋转合成数据上验证不破坏恢复，再以冻结消融比较其真实语义收益。

## 9. 可复现性与一键审计

### 9.1 关键代码

| 组件 | 路径 |
|---|---|
| 软群组 | `src/cc_hiwa/soft_groups.py` |
| 层级 OT 与结构化旋转 | `src/cc_hiwa/soft_hiwa.py` |
| 资格门槛与多启动 | `src/cc_hiwa/alignment_qualification.py` |
| ROCA 候选资格 | `src/cc_hiwa/roca_qualification.py` |
| 合成生成器 | `src/cc_hiwa/synthetic_boundary.py` |
| 合成边界运行器 | `experiments/synthetic/run_gc_hiwa_boundary_benchmark.py` |
| Panoptic 运行器 | `experiments/panoptic/run_camera_coordinate_alignment.py` |

### 9.2 推荐检查命令

```powershell
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
python -m pytest tests -q
python experiments/synthetic/verify_gc_hiwa_v2_evidence.py
```

当前完整测试为 **114 passed**。单线程环境变量用于规避 Windows 下 Torch 多线程运行时的偶发 abort；它不改变 GC-HiWA 的方法超参数。

证据审计器应输出：

- 合成独立批次：15 个可恢复运行全部接受；25 个预声明违例全部弃权；
- 谱伪装非正交：5 个运行核心与 ROCA 全部弃权；
- Panoptic：结构化资格接受，旋转误差 0.00818，Recall@1 为 1。

## 10. 论文应如何写

### 10.1 可以作为主结论的句子

> 在共享群组几何且存在单一结构化正交坐标关系的受控条件下，GC-HiWA 可从无标签、无配对姿态中恢复三维相机坐标关系；其组合资格门槛会在本文设计的错配、非正交和跨模态边界中弃权。

### 10.2 必须保留的限制

- 恢复定理假设正确耦合，不是端到端 OT--ADMM 的全局收敛证明；
- 阈值是固定工程合同，不是普适统计显著性界；
- Panoptic 是同一采集系统下的受控几何证据；
- 下游代理不是有人工语义标签的动作识别；
- 没有完整 UWA3DII 数据前，不能声称真实跨设备语义泛化。

## 11. 下一步：UWA3DII 冻结语义实验

候选数据是 UWA3D Multiview Activity II：30 个动作、10 名受试者、4 个 Kinect 视角。其不同视角为独立表演，不是同步配对，因此恰好考验无标签语义迁移，但不能评价帧级物理旋转误差。

冻结协议：

1. 按受试者划分训练／测试，禁止随机帧泄露；
2. 在源视角训练真实动作分类器；
3. 只用训练受试者的无标签目标视角数据拟合 GC-HiWA；
4. 用未见受试者的目标视角测试；
5. 同时报告 source-only、GC-HiWA 映射、标定／配对上界、接受率、弃权率和错误接受；
6. 不存在的同步帧和相机真值一律不伪造为指标。

当前 `data/raw/uwa3dii/UWA3DII_Skeleton.zip` 是下载超时留下的不完整文件，不能使用。完整数据到位后，实验从数据结构审计开始，不需要再修改方法合同。

## 12. 最终定位

GC-HiWA 的价值不是输出一个总会“看起来合理”的旋转，而是把它知道什么、不能知道什么，以及何时不该输出，全部变成可检查的数学与实验对象。对于无标定跨视角三维姿态，这比追求一个在任何数据集上都微小提升的 OT 变体更可信，也更接近实际部署所需的安全行为。
