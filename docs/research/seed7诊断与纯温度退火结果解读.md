# Seed 7 诊断与纯温度退火结果解读

## 0. 最终结论

这次结果非常清楚：

> **纯温度退火没有救回来方向准确率。**

更具体地说：

```text
固定 prototypes 之后，seed 7 仍然崩。
因此，之前退火失败不是因为每个温度重新学习 prototypes 导致的 prototype identity shift。
更可能的问题是 Soft-HiWA 的旋转共识结构本身不适合当前软组重叠。
```

所以现在不建议继续围绕温度退火反复调路径。

下一步应该转向：

```text
P_ij 加权全局旋转共识
```

也就是让可信组对对全局旋转说话更多，让不可信组对说话更少。

---

## 1. 这次实验回答了什么问题

上一轮退火实验有一个漏洞：

```text
tau=0.25、0.35、0.50 每个阶段都重新学习 prototypes。
```

这导致我们不能确定退火失败是因为：

```text
soft assignment 的温度路径不行
```

还是因为：

```text
不同温度阶段的 prototypes 发生了 identity shift
```

所以这次做了两件事：

1. **Seed 7 诊断**  
   解释 seed 7 为什么 $R^2$ 高，但是 direction accuracy 很低。

2. **纯温度退火**  
   先在 $\tau=0.25$ 学一套固定 prototypes，然后只改变 temperature，不再重新学习 prototypes。

这一步的目的非常明确：

> 如果固定 prototypes 后 seed 7 不崩，说明之前问题可能来自 prototype identity shift；  
> 如果固定 prototypes 后 seed 7 仍然崩，说明问题更可能来自 Soft-HiWA 优化器本身。

结果是第二种。

---

## 2. Seed 7 诊断结论

Seed 7 的核心现象是：

```text
R² 很高，但 direction accuracy 很低。
```

这说明它不是普通的几何对齐失败。

如果几何完全失败，$R^2$ 应该也很差。  
但现在 $R^2$ 不差，甚至变高，说明连续运动几何被对齐得不错。

真正坏掉的是：

```text
离散方向语义
```

也就是 direction correspondence。

---

## 3. Seed 7 不是 prototype identity shift

诊断报告检查了不同温度阶段的 prototype matching。

结果是：

| Transition | Neural cosine similarity | Movement cosine similarity |
|---|---:|---:|
| $\tau=0.25\rightarrow0.35$ | 0.9997 | 0.9980 |
| $\tau=0.25\rightarrow0.50$ | 0.9988 | 0.9852 |

这些相似度都很高。

这说明：

```text
prototypes 在不同温度之间基本稳定；
没有明显 group identity shift。
```

因此，seed 7 的崩塌不太可能是因为 prototype 重学习导致语义漂移。

---

## 4. Seed 7 的错误模式

Seed 7 在 annealed $\tau=0.50$ 下的主要错误是：

| True direction | Predicted as | Count |
|---:|---:|---:|
| 0 | 1 | 187 |
| 1 | 0 | 69 |
| 2 | 1 | 55 |
| 3 | 1 | 47 |
| 1 | 3 | 44 |
| 2 | 3 | 39 |

每类准确率：

| Direction | Accuracy |
|---:|---:|
| 0 | 0.044 |
| 1 | 0.333 |
| 2 | 0.365 |
| 3 | 0.503 |

这非常关键。

Direction 0 几乎完全坏掉：

$$
0.044
$$

这不是一个小波动，而是方向语义严重错配。

诊断报告认为这不是简单的两类方向交换，而更像：

```text
class 2 dispersed across multiple targets
```

同时 direction 0 大量被预测成 direction 1。

所以现在的问题不是某两个方向整齐互换，而是：

```text
某些方向语义被软优化后的邻域结构打散了。
```

---

## 5. 纯温度退火总体结果

最终 $\tau=0.50$ 的汇总结果如下：

| 方法 | Direction accuracy | Movement $R^2$ | 收敛 |
|---|---:|---:|---:|
| Prototype hard | $0.5441\pm0.0745$ | $0.576\pm0.017$ | 5/5 |
| Fixed soft warm | $0.5441\pm0.0745$ | $0.577\pm0.016$ | 5/5 |
| Pure annealed final | $0.5101\pm0.0906$ | $0.601\pm0.005$ | 5/5 |

结论：

```text
方向准确率下降；
R² 明显提高且更稳定；
所有种子都收敛。
```

这说明 pure annealing 做到了一件事：

> 它让连续运动几何更稳定。

但它没有做到我们真正想要的事：

> 提高方向解码准确率。

---

## 6. 逐种子结果

| Seed | Hard Acc | Fixed Warm Acc | Pure Annealed Acc | Hard $R^2$ | Fixed Warm $R^2$ | Pure Annealed $R^2$ |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.5778 | 0.5778 | 0.5778 | 0.569 | 0.593 | 0.597 |
| 6 | 0.5762 | 0.5778 | 0.5762 | 0.575 | 0.571 | 0.607 |
| 7 | 0.5746 | 0.5795 | 0.4093 | 0.551 | 0.585 | 0.603 |
| 8 | 0.4109 | 0.4109 | 0.4125 | 0.594 | 0.584 | 0.597 |
| 9 | 0.5811 | 0.5746 | 0.5746 | 0.588 | 0.552 | 0.598 |

这个表的解释非常重要。

Seed 5、6、9 基本没坏。  
Seed 8 本来 hard 就很低，pure annealing 也差不多。  
真正导致平均 accuracy 下降的是 seed 7。

所以 seed 7 不是随机小异常，而是暴露了一个机制问题：

> Soft-HiWA 可以找到 $R^2$ 很好的连续几何解，但这个解不一定保留方向语义。

---

## 7. 为什么 $R^2$ 上升但 accuracy 下降

这里要特别小心。

$R^2$ 衡量的是连续运动变量的拟合：

$$
R^2
$$

方向准确率衡量的是离散方向分类：

$$
\operatorname{Accuracy}
$$

二者不是同一个目标。

退火后的 Soft-HiWA 可能学到了一个旋转，使得整体连续速度场更平滑、更接近，因此 $R^2$ 高。

但这个旋转可能让方向簇之间的最近邻关系改变，导致 direction label 被打乱。

所以这次结果告诉我们：

> 连续几何对齐好，不等于方向语义对齐好。

这对论文写作其实很重要。它说明神经数据对齐有两个层面：

```text
continuous kinematic alignment
discrete direction semantic alignment
```

当前 pure annealing 更偏向前者，而不是后者。

---

## 8. 纯退火为什么仍然失败

这次已经排除了 prototype identity shift。

那么失败更可能来自：

```text
SoftHiWA 的旋转共识结构
```

也就是说，当前 Soft-HiWA 的全局旋转更新可能没有区分：

```text
高可信组对
低可信组对
```

在 soft group 中，同一个样本可能参与多个组对。  
某些组对其实是不可靠的，但它们仍然参与全局旋转共识。

这会导致：

```text
低可信局部旋转干扰全局旋转
→ 得到一个连续几何上不错但方向语义错的 R
```

这就是为什么下一步应该做 $P_{ij}$ 加权全局旋转共识。

---

## 9. 下一步：P_ij 加权全局旋转共识

当前全局旋转共识可以理解成所有组对差不多平等地影响 $R_g$。

但 group-level transport matrix：

$$
P_{ij}
$$

本来就表示第 $i$ 个 source group 和第 $j$ 个 target group 的对应强度。

如果：

$$
P_{ij}
$$

很大，说明这个组对可信。

如果：

$$
P_{ij}
$$

很小，说明这个组对不可信。

因此，全局旋转更新应该更像：

$$
R_g
=
\operatorname{Proj}_{\mathcal O(d)}
\left(
\sum_{i,j}P_{ij}(R_{ij}+L_{ij})
\right).
$$

直观解释是：

> 可信组对多影响全局旋转；不可信组对少影响全局旋转。

这正好针对当前失败机制。

---

## 10. 现在不建议做什么

### 10.1 不建议继续调退火路径

例如继续试：

$$
0.15\rightarrow0.25\rightarrow0.35\rightarrow0.50
$$

或者：

$$
0.25\rightarrow0.30\rightarrow0.35\rightarrow0.40\rightarrow0.50
$$

现在已经知道问题不是 prototype drift。继续调路径可能变成无意义调参。

### 10.2 不建议马上扩大种子

当前机制还没有变。扩大种子只会更精确地验证：

```text
pure annealing 平均 accuracy 不如 hard baseline。
```

### 10.3 不建议马上做 $K=5$

$K=5$ 仍然值得做，但现在更核心的问题是：

```text
soft group 重叠如何参与全局旋转共识
```

如果不解决这个问题，$K=5$ 可能会引入更多组对，让共识问题更复杂。

### 10.4 不建议马上加标签损失

加标签损失可能会修复 direction accuracy，但那会让问题变成 supervised alignment。

目前应该先把无监督 Soft-HiWA 的优化结构搞清楚。

---

## 11. 下一条给 Codex / DeepSeek 的任务方向

下一步任务应该是：

```text
实现并消融 P_ij 加权全局旋转共识。
```

实验边界要保持干净：

```text
不改 K
不加标签
不扩大种子
不继续调退火路径
不同时加入其他机制
```

比较对象至少包括：

| 方法 | 说明 |
|---|---|
| Prototype hard | 原型硬化基线 |
| Fixed soft warm | 当前稳定软基线 |
| Pure annealed | 当前退火基线 |
| P-weighted fixed soft warm | 只加 $P_{ij}$ 加权共识 |
| P-weighted pure annealed | 退火 + $P_{ij}$ 加权共识 |

但第一轮可以先只做：

```text
P-weighted fixed soft warm
```

如果它能修复 seed 7 或提高平均 accuracy，再考虑和 pure annealing 结合。

---

## 12. 当前阶段性结论怎么写

可以这样写：

> 进一步的 seed 7 诊断与纯温度退火实验表明，退火失败并非主要由不同温度阶段重新学习 prototypes 所导致。Prototype matching 显示 neural 和 movement prototypes 在不同温度之间高度稳定，固定 prototypes 的 pure annealing 仍然出现 seed 7 方向准确率崩塌。Pure annealing 在 held-out seeds 上将 movement $R^2$ 从约 0.577 提高到约 0.601，但方向准确率从约 0.544 降至约 0.510，说明连续运动几何对齐和离散方向语义对齐发生分离。当前证据指向 Soft-HiWA 的全局旋转共识机制：低可信软组对可能干扰全局旋转，导致几何拟合较好但方向语义错配。因此下一步应优先研究 $P_{ij}$ 加权全局旋转共识，而不是继续调退火路径。

---

## 13. 一句话总结

> 纯温度退火把 prototype 漂移这个嫌疑排除了，但仍然没有救回方向准确率。它提高了 $R^2$，却让 direction accuracy 下降，说明问题已经从“温度怎么调”转向“Soft-HiWA 的软组对如何参与全局旋转共识”。下一步应该做 $P_{ij}$ 加权全局旋转共识。
