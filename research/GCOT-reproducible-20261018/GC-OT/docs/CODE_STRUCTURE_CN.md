# 代码结构

工程采用单一职责分层：

```text
gcot/                         算法库
experiments/
  reference.py               作者协议公共设置
  synthetic/core.py          合成实验逻辑
  synthetic/run.py           合成实验计算入口
  synthetic/matlab/          唯一保留的 MATLAB 基线生成源码
  neural/core.py              神经 GCOT 拟合及作者坐标约定
  neural/run.py               神经实验计算入口
visualization/                独立绘图模块
evaluation/                   指标与完整性检查
data/processed/               固定输入
results/synthetic/            合成结果
results/neural/               神经结果
tests/                        自动化测试
docs/                         说明和来源记录
```

依赖方向为：`experiments/visualization/evaluation -> gcot`。算法库不依赖实验或绘图；绘图不负责重新拟合；结果目录不包含可执行代码。

标准入口全部使用 Python 模块形式：

```powershell
.venv/Scripts/python.exe -m experiments.synthetic.run
.venv/Scripts/python.exe -m experiments.neural.run
.venv/Scripts/python.exe -m visualization.synthetic
.venv/Scripts/python.exe -m visualization.neural
.venv/Scripts/python.exe -m evaluation.validation --scope all
```
