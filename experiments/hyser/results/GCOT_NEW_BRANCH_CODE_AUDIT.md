# Hyser Hard HiWA：核对新上传的 GCOT 可复现代码

日期：2026-11-17。指定来源为 GitHub 分支：experiments/dandi-000138-4d-hard-clustering-20261117，目录：research/GCOT-reproducible-20261018/GC-OT。

## 对照范围与结论

- 已阅读该分支的 gcot/alignment.py、gcot/baselines.py、gcot/numerics.py、gcot/grouping.py、gcot/preprocessing.py、experiments/neural/core.py、evaluation/metrics.py，以及 reference/GCOT-copy-PyHiWA/src/hiwa.py。
- 原 Hyser 250 点实验使用 K-means(K=4) 和恒等辅助投影、手工固定 mu=.05, tol=.01, 局部40轮、样本 Sinkhorn80轮、簇级300轮。该流程不是新包中的完整 HiWA-GMM 或 HiWA-SSC；不能当作同一基线结果。
- 新包 hiwa_gmm 用两个域独立 GMM、源域 Isomap(2,12) 与目标前二维的辅助投影，HiWA 旋转仍作用于原三维输入。另有 hiwa_ssc 用 SSC。直接 K-means 是一个自行设定的硬分组变体。
- 新包的 HiWA 默认配置：mu=.005, tol=.1, maxiter=300, inner_maxiter=100, epsilon_group=.2, epsilon_sample=.1, sinkhorn=150, group_sinkhorn=1000；而原 Hyser 扫参选用了较严格但不同的参数。HiWA 缩放1/sqrt(d)与 penalty mu/d 的核心更新一致。
- 新包 SolverConfig(stopping='legacy') 只要求全局 R 两轮变化小于0.1，不能理解成 ADMM primal 共识也足够小。源码还提供 stopping='admm' 严格规则，应报告所有残差，而非只报告一个收敛字样。
- 原 PyHiWA 参考代码中的 _eval_R2 缺平方，在中心化数据上可错误地返回1；新 GC-OT/evaluation/metrics.py 的 regression_metrics 用正确的平方误差。这个 bug 只影响参考实现的评价输出，不直接改变 OT 或旋转的更新。

## 独立数学核验：固定旧特征的正交 Oracle

从此前真实 Hyser 250窗口的 all250_input.npz 读取 X、Y 与已知同步索引；不依赖任何 HiWA 求解器。

最小可能的正交平方误差：849.2851987；目标 SST：747。故在当前零时延、双域独立白化的三维特征上，最佳正交配对 R² = -0.1369279768。这是 Procrustes SVD 的全局最小平方误差，任何 Hard/Soft HiWA 仅用正交 R 均无法超过该上界。

对真实同步索引诊断性扫描：将力量相对肌电后移0.1s，配对正交 Oracle R² 提高至 +0.0562；无约束线性映射 R²：0时延 +0.2607、+0.1s +0.3736。时滞是后验选择，不能当成经过独立验证的生理时延结论，也不能当成无监督恢复结果。

## 新代码数值公式的本地对照

计算环境不能直接下载 GitHub 原文件，因此下列结果由 gcot/numerics.py 和 gcot/alignment.py 所示数值更新**手工忠实转录**并在同一份真实250点特征上运行；不是官方源文件字节级运行。随机种子均为42，HiWA新包默认 tol=.1、mu=.005 与150/1000次 Sinkhorn。

| 实验路径 | 外层轮次 | R²（真实同步250点） | 平均配对距离 | 末次全局R变化 | 末次局部整体共识残差 |
|---|---:|---:|---:|---:|---:|
| 新包更新 + 旧 K-means 分组 | 88 | -0.887570 | 2.211619 | 0.089311 | 0.911022 |
| 新包 HiWA-GMM / Isomap 路径 | 43 | -1.168068 | 2.397899 | 0.073069 | 1.868018 |

两次均符合 **legacy 旋转变化阈值**，但局部共识残差远大于常用严格ADMM阈值。因此这不是两个已严格求解到局部共识的可靠性能对比。需要在确切环境直接运行原文件，并检查完整迭代轨迹。

另外进行了已知真实正交旋转的四簇合成 sanity test：转录更新在 seed1、seed2 的相对 Frobenius 恢复误差分别为0.001087、0.002458，而 seed0 误差1.275677。它表明更新公式具备恢复旋转的能力，但非凸问题有明显初始化敏感性；合成测试不属于真实Hyser证据。

## 正式下一步

新加 scripts/run_gcot_branch_hard.py 会**真正导入该分支里的 gcot 源代码**，在此前准备的全250点输入上分别运行 hard_measures+K-means 和 hiwa_gmm(Isomap)，输出 R²、relative squared error、平均距离及所有残差。这份脚本尚未在具有完整分支源码的环境直接执行，任何旧结果不能以它的名义报告。需要从指定分支创建 worktree，指向其 GC-OT 根目录，再执行。

研究判断：现已确认协议与评价代码问题，但**不能仅归因于代码错误**。旧的零时延三维表示在正交 oracle 下也只有负R²；调整特征、分组、配对时延的诊断和数值求解应分别进行。
