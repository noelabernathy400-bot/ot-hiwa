# 2026-07-10 Joint-Prototype-HiWA 最小实现与实验报告

## 目标

在 HiWA / TACO-inspired Soft-Prototype HiWA / ROCA 主线内，实现最小版 TACO-style joint prototype optimization：固定神经与运动 3D embedding，只优化 prototypes 及其诱导的 soft assignments，并用 alignment-aware representative loss 把 frozen ROCA 选出的 rotation 反馈到 prototype/assignment。

## 改动文件

- 新增 `Projects/最优传输/Experiments/hiwa_python_reproduction/scripts/joint_prototypes.py`
- 新增 `Projects/最优传输/Experiments/hiwa_python_reproduction/scripts/run_joint_prototype_hiwa.py`
- 新增 `Projects/最优传输/Experiments/hiwa_python_reproduction/tests/test_joint_prototypes.py`
- 新增 `Projects/最优传输/Experiments/hiwa_python_reproduction/results/joint_prototype_hiwa_joint_proto_minimal_seeds_50_54.json`
- 新增 `Projects/最优传输/Experiments/hiwa_python_reproduction/results/joint_prototype_hiwa_summary_joint_proto_minimal_seeds_50_54.json`
- 新增 `Projects/最优传输/Experiments/hiwa_python_reproduction/figures/joint_prototype_hiwa_joint_proto_minimal_seeds_50_54.png`
- 新增 `Projects/最优传输/最优传输/文本笔记/代码方面/Joint-Prototype-HiWA最小实验记录_joint_proto_minimal_seeds_50_54.md`

## 关键实现决定

- 没有修改原作者 PyHiWA 源码，也没有覆盖既有 JSON / PNG / Markdown。
- 没有引入 PBMC、图像文本、新数据集、自动 K、深度 encoder 或 direction label loss。
- assignment 仍沿用现有 Soft-Prototype HiWA 的标准化 embedding 空间内积 softmax，保证与 frozen baseline 可比。
- representative alignment loss 使用原始固定 3D embedding 的 soft representative，固定 rotation 来自同 seed 的 frozen ROCA label-free selection。
- 最终评价仍通过 det ±1 候选 + ROCA representative orientation selector；direction accuracy 与 $R^2$ 只用于最终评估，不用于优化、选择或调参。

## 验证

- `python -m unittest Experiments.hiwa_python_reproduction.tests.test_joint_prototypes`
  - 结果：通过，2 个测试。
- `python -m py_compile Experiments\hiwa_python_reproduction\scripts\joint_prototypes.py Experiments\hiwa_python_reproduction\scripts\run_joint_prototype_hiwa.py`
  - 结果：通过。
- `python Experiments\hiwa_python_reproduction\scripts\run_joint_prototype_hiwa.py --seeds 50 51 52 53 54 --profile quick --tag joint_proto_minimal_seeds_50_54`
  - 结果：完成并写出结果、汇总、图和实验笔记。

## 实验结果摘要

| 指标 | Frozen ROCA | Joint prototypes + ROCA | Delta |
|---|---:|---:|---:|
| mean direction accuracy | 0.5743 | 0.5740 | -0.0003 |
| mean movement R2 | 0.6008 | 0.5839 | -0.0169 |
| mean joint objective delta |  |  | -0.014783 |

每个 seed 的 joint objective 均下降，但下游评价没有同步提升：direction accuracy 基本持平，movement R2 平均下降。

## 初步判断

这是一个有价值的机制性负结果/混合结果：当前最小 alignment-aware prototype objective 可以被优化，但它还没有证明能改善 Soft-HiWA 的 sample-level transport 或最终运动几何。下一步不应发散到新数据集或大规模调参，而应先诊断 loss 分量尺度、group mass/entropy 是否稳定，以及固定 frozen ROCA rotation 是否把 prototype 优化锁在过窄的局部目标里。

## 风险与跳过事项

- 当前 runner 使用 `quick` profile，是第一阶段机制验证，不代表最终论文主结果。
- 当前 loss 权重是单组预设，不是通过标签选择得到；尚未证明是最佳权重。
- 未更新全局交接文件，避免碰到已有大量未提交系统改动；本报告作为本次任务的正式交接记录。

## 下一步建议

1. 先读 `joint_prototype_hiwa_summary_joint_proto_minimal_seeds_50_54.json` 与完整 raw JSON，检查每个 seed 的 entropy、group mass、alignment loss 是否出现异常。
2. 做一个只读诊断表：比较 frozen vs joint 的 group representatives、mass、entropy、ROCA determinant selector 是否改变。
3. 若诊断显示 objective 尺度失衡，再做极小范围的 loss scale sanity check；不要扩展 PBMC、encoder、新数据集或自动 K。
