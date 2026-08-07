# GC-HiWA v2：实现规格（可直接交给代码助手）

## 🧭 0. 任务目标

请实现 **GC-HiWA v2 的固定主干**，用于两个无标签数据域的分层最优传输对齐。

本版本的研究问题是：当两个域具有相同的多组结构，且可由近似全局正交变换对应时，能否恢复组对应、全局正交变换和样本级传输；当该假设被噪声、混合变换、群组变化或非正交形变破坏时，模型应暴露失败，而不能伪造成功。

必须遵守以下范围：

- 不使用目标域标签、配对样本 ID、准确率或 $R^2$ 进行训练、选参或分支选择。
- 默认使用 full-support 局部 OT；稀疏支持只能作为显式的计算近似。
- 可学习编码器、联合原型更新、端到端微分都不是本次主干；先不要实现或启用它们。
- 不要使用 $\varepsilon/P_{ij}$ 作为局部 Sinkhorn 温度。内层和外层熵正则必须是独立、固定的正数。

> 交付格式说明：本文只使用标准 Markdown、LaTeX 数学和可选 Mermaid 图。若阅读器不渲染 Mermaid，可直接按后续伪代码实现；不依赖任何本地文件或历史对话。

```mermaid
flowchart LR
    accTitle: GC HiWA Core Flow
    accDescr: The fixed GC HiWA core maps raw data through fixed adapters, learns soft groups, solves hierarchical transport and ADMM consensus for both determinant branches, then performs qualified ROCA selection or abstains.

    raw_data["原始数据 X 与 Y"] --> adapters["固定数据适配器"]
    adapters --> groups["冻结软分组 A 与 B"]
    groups --> local_ot["组内 OT Q_ij"]
    local_ot --> group_ot["组间 OT P"]
    group_ot --> admm["局部旋转与 ADMM 共识"]
    admm --> branches["det -1 与 +1 候选"]
    branches --> roca["ROCA 资格选择或弃权"]

    classDef core fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a5f
    classDef decision fill:#fef9c3,stroke:#ca8a04,stroke-width:2px,color:#713f12
    class raw_data,adapters,groups,local_ot,group_ot,admm core
    class branches,roca decision
```

## 📥 1. 输入、输出与固定适配器

输入为两个无标签数据矩阵：

$$
X^{raw}\in\mathbb{R}^{N\times p_X},\qquad
Y^{raw}\in\mathbb{R}^{M\times p_Y}.
$$

先用固定数据适配器映射到相同维度 $d$：

$$
Z^X=h_X(X^{raw})\in\mathbb{R}^{N\times d},\qquad
Z^Y=h_Y(Y^{raw})\in\mathbb{R}^{M\times d}.
$$

本次实现中，$h_X,h_Y$ 只能是以下之一：

1. 恒等映射：原始特征已经同维且可直接比较；
2. 固定标准化；
3. 在训练数据上单独拟合、随后冻结的 PCA 或其他固定降维器。

不要训练 MLP 编码器。若未来启用可学习编码器，必须作为单独实验，并额外加入重建、方差和几何保持损失以避免表示塌缩。

输出至少包括：

- 全局正交变换 $R$；
- 组间传输 $P\in\mathbb{R}^{K\times K}$；
- 每个组对的样本级传输 $Q_{ij}\in\mathbb{R}^{N\times M}$；
- 源域和目标域软归属矩阵 $A,B$；
- 局部旋转 $R_{ij}$ 与 ADMM 对偶变量 $U_{ij}$；
- 完整数值诊断和 ROCA 决策/弃权原因。

## 🧩 2. 软分组与群组经验测度

设组数为 $K$。在固定表示空间中学习源域和目标域原型：

$$
C^X=\{c_1^X,\ldots,c_K^X\},\qquad
C^Y=\{c_1^Y,\ldots,c_K^Y\}.
$$

使用温度为 $\tau>0$ 的 softmax 得到软归属：

$$
A_{ki}=\frac{\exp(\bar z_k^{X\top}c_i^X/\tau)}
{\sum_{r=1}^{K}\exp(\bar z_k^{X\top}c_r^X/\tau)},
$$

$$
B_{lj}=\frac{\exp(\bar z_l^{Y\top}c_j^Y/\tau)}
{\sum_{r=1}^{K}\exp(\bar z_l^{Y\top}c_r^Y/\tau)}.
$$

其中 $\bar z$ 表示按域标准化后的特征。每行必须和为 1。

主干中的原型只在初始化阶段学习，然后冻结。可使用以下无监督目标：

$$
\mathcal{L}_{proto}=\mathcal{L}_{rec}
+\lambda_{bal}\mathcal{L}_{bal}
+\lambda_{ent}\mathcal{L}_{ent}
+\lambda_{sep}\mathcal{L}_{sep}.
$$

原型分离项必须有界。推荐余弦 hinge：

$$
\mathcal{L}_{sep}=
\frac{1}{|\mathcal{P}|}
\sum_{(r,s)\in\mathcal{P}}
\left[\max\left(0,
\frac{c_r^\top c_s}{\|c_r\|\|c_s\|}-m\right)\right]^2.
$$

禁止使用负的平方距离分离项，例如 $-\sum_{r\neq s}\|c_r-c_s\|^2$；该项无下界，会驱动原型发散。

对每个组构造非均匀样本边缘：

$$
a_k^{(i)}=\frac{A_{ki}}{\sum_{r=1}^{N}A_{ri}},
\qquad
b_l^{(j)}=\frac{B_{lj}}{\sum_{r=1}^{M}B_{rj}}.
$$

因此 $a^{(i)}\in\Delta_N$，$b^{(j)}\in\Delta_M$。这些边缘必须显式传给 Sinkhorn，不能在局部 OT 中替换成均匀权重。

## 🔁 3. 双层 GCOT

定义负熵正则：

$$
\Omega(T)=\sum_{r,s}T_{rs}\bigl(\log(T_{rs}+10^{-12})-1\bigr).
$$

设 $R\in O(d)$。对每个组对 $(i,j)$，局部成本为：

$$
D_R(k,l)=\|Rz_k^X-z_l^Y\|_2^2.
$$

内层样本级 OT：

$$
Q_{ij}^{*}=
\arg\min_{Q\in\mathcal{U}(a^{(i)},b^{(j)})}
\langle D_R,Q\rangle+\varepsilon_{in}\Omega(Q),
$$

其中 $\varepsilon_{in}>0$ 是固定超参数。实现时使用稳定的非均匀 Sinkhorn，并检查：

$$
Q_{ij}\mathbf{1}=a^{(i)},\qquad
Q_{ij}^{\top}\mathbf{1}=b^{(j)}.
$$

令：

$$
G_{ij}=\langle D_R,Q_{ij}^{*}\rangle+\varepsilon_{in}\Omega(Q_{ij}^{*}).
$$

外层组级 OT：

$$
P^{*}=
\arg\min_{P\in\mathcal{U}(u,u)}
\sum_{i,j}P_{ij}G_{ij}+\varepsilon_{out}\Omega(P),
\qquad u=\frac{1}{K}\mathbf{1}_K.
$$

其中 $\varepsilon_{out}>0$ 与 $\varepsilon_{in}>0$ 独立设置。必须检查 $P$ 的行、列边缘误差。

## ⚙️ 4. HiWA 风格 ADMM 旋转共识

为每个组对维护局部旋转 $R_{ij}$ 和对偶变量 $U_{ij}$，并维护一个全局正交旋转 $R$。

每轮迭代执行：

1. 固定当前 $P,R,U$，交替更新每个组对的 $R_{ij}$ 与 $Q_{ij}$；
2. 从局部 OT 代价更新外层传输 $P$；
3. 计算传输加权共识矩阵：

$$
M=\sum_{i,j}P_{ij}(R_{ij}+U_{ij});
$$

4. 将 $M$ 投影到正交群：若 $M=U\Sigma V^\top$，则 $R=UV^\top$；
5. 更新对偶变量：

$$
U_{ij}\leftarrow U_{ij}+R_{ij}-R.
$$

本版本默认使用 $P_{ij}$ 加权的共识。均匀平均只保留为消融对照，不应静默替代主干。

每轮必须记录：

- 全局旋转变化量；
- ADMM 原始残差 $\max_{ij}\|R_{ij}-R\|_F$；
- ADMM 对偶残差；
- 每个 $Q_{ij}$ 的边缘误差；
- $P$ 的行、列边缘误差；
- $\|R^\top R-I\|_F$；
- 群组质量、分配熵和空群情况。

只有全局残差与原始残差同时低于预先设置的阈值时，才标记为收敛。

## 🧭 5. ROCA：行列式分支资格选择

ROCA 只解决 $O(d)$ 中 $\det(R)=+1$ 与 $\det(R)=-1$ 两个分支的歧义。它不能修复非正交、群组不一致或错误表示空间。

必须分别完整求解两个候选：

$$
R^{(+)}:\det(R)=+1,
\qquad
R^{(-)}:\det(R)=-1.
$$

对每个候选，必须使用该候选自己的 $P$，不能平均两个候选的 $P$。有向单纯形只用于检查候选内部是否自洽；它不能单独区分两个都自洽的候选。步骤如下：

1. 根据该候选的 $P$ 求源组到目标组的最大权重一一匹配；
2. 用匹配后的组代表元计算有向单纯形符号；
3. 检查该符号是否与候选的行列式符号一致；
4. 检查该候选是否 ADMM 收敛，且局部 OT 边缘误差不超过阈值；
5. 若只有一个候选同时通过上述检查，选择它；若两个都失败，返回 `abstain`；
6. 若两个候选都通过，比较它们各自的无监督传输目标值 $J^{(+)}$ 和 $J^{(-)}$。仅当相对差距

   $$
   \frac{|J^{(+)}-J^{(-)}|}{\min(J^{(+)},J^{(-)})} \ge 0.05
   $$

   时，选择目标值较低的候选；否则返回 `abstain`。不得用 Accuracy、$R^2$、真实旋转或测试标签强行选择。

这一规则只能在单一正交几何仍合理时消除候选平局。若数据存在混合变换、群组不一致或非正交变形，较低目标值也可能只是“更好地拟合了错误模型”；它不能作为模型适用性的证明。

有向单纯形 ROCA 仅适用于 $K=d+1$ 且代表元不退化的情况。若不满足，应返回 `not_applicable`，不要伪造符号。

## 💻 6. 伪代码

```text
input: X_raw, Y_raw, fixed adapters h_X/h_Y, K, tau,
       epsilon_in, epsilon_out, mu, tolerances

Z_X = h_X(X_raw)
Z_Y = h_Y(Y_raw)
A, C_X = learn_and_freeze_soft_groups(Z_X, K, tau)
B, C_Y = learn_and_freeze_soft_groups(Z_Y, K, tau)
build a^(i) from A and b^(j) from B

for sign in {-1, +1}:
    initialize R with determinant sign, P uniformly, R_ij, U_ij
    repeat until global and primal ADMM residuals both converge:
        for every group pair (i, j):
            alternate local Procrustes update of R_ij and
            weighted Sinkhorn update of Q_ij using fixed epsilon_in
        update group costs G_ij
        update P by outer Sinkhorn using fixed epsilon_out
        R = orthogonal_projection(sum_ij P_ij * (R_ij + U_ij), sign)
        U_ij = U_ij + R_ij - R
        record all diagnostics
    save candidate(sign, R, P, Q, diagnostics)

decision = qualify_roca_candidates(candidate(-1), candidate(+1))
if decision selects one candidate:
    return selected candidate
else:
    return abstain plus both candidates and diagnostics
```

## 🧪 7. 合成验证必须先做

实现完成后，先运行已知真值的合成基准，再运行真实数据。合成数据必须覆盖：

1. 理想正交旋转；
2. 理想反射；
3. 逐级增加的加性噪声；
4. 一部分样本来自另一正交变换的混合变换；
5. 一个目标群中心偏移造成的群组不一致；
6. 非正交缩放/剪切。

拟合时不能读取真旋转、真实群标签、配对目标行或行列式真值。它们只用于事后报告：

- 旋转 Frobenius 误差；
- 与隐藏配对目标的 MSE；
- 群组恢复 ARI；
- 收敛率；
- ROCA 正确选择率与弃权率。

现有实验证据表明：传输加权共识在理想条件下比均匀共识稳定；可学习编码器和联合优化尚未显示稳定增益。因此先完成本规格的固定主干，不要将编码器、联合优化或预测头混入首版结果。

## ⛔ 8. 不要做的事情

- 不要把两个候选分支的 $P$ 平均后再做 ROCA 匹配；
- 不要用 `$epsilon / P_ij` 调节局部 Sinkhorn 温度；
- 不要在主干中训练编码器或联合更新原型；
- 不要根据真实数据的 Accuracy 或 $R^2$ 选择超参数、ROCA 分支或是否保留一次运行；
- 不要只因旋转矩阵变化很小就宣布成功，必须同时检查 ADMM 原始残差和 OT 边缘误差；
- 不要在真实数据失败后直接调参，先检查它是否满足近似正交、共享群组结构的前提。
