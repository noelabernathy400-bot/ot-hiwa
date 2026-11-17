# GCOT 可分享复现包

## 先看这里
正式项目入口：`GCOT-author-reproduction/GCOT-author-reproduction/GC-OT/README.md`。
两层同名目录是现有代码的相对路径约定，请完整解压，勿单独移动 GC-OT。
包内算法、配置、数据、图表和结果均按原文件保留；没有修改算法或绘图代码。

## 目录
- `GCOT-author-reproduction/GCOT-author-reproduction/GC-OT/`：完整主项目。
  - `gcot/` 算法；`experiments/` 四条实验流水线；`visualization/` 绘图。
  - `configs/` 固定配置；`data/` 原始及预处理数据；`results/` 当前结果和图表。
  - `tests/` 测试；`evaluation/` 验证；`docs/` 说明；`vendor/` 第三方资源。
- 主项目旁的 `PyHiWA/`、`hiwa-matlab/`、`author_reference/`：原作者参考。
- 顶层 `HiWA/`：主项目按相对路径读取的作者代码，必须保留。
- 顶层 PDF：项目参考文献。
- `environment-freeze.txt`：已通过本地检查的实际环境精确版本。
- `SHA256SUMS.json`：包内文件校验；`verify_package.py`：无第三方依赖的校验器。

## 安装环境（Windows PowerShell）
已验证环境：Windows x64，Python 3.11.9，CPU PyTorch 2.6.0。
先安装 Python 3.11.9，在本文件所在目录运行：
```powershell
python verify_package.py
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r environment-freeze.txt --extra-index-url https://download.pytorch.org/whl/cpu
cd GCOT-author-reproduction/GCOT-author-reproduction/GC-OT
..\..\..\.venv\Scripts\python.exe -m pytest -q
..\..\..\.venv\Scripts\python.exe -m evaluation.validation --scope all
```
后续将下面的 `python` 替换为 `../../../.venv/Scripts/python.exe`，或先激活环境。
无需安装项目为包，所有模块命令均在正式 GC-OT 根目录执行。
原 `requirements-lock.txt` 留作历史记录；请使用顶层 `environment-freeze.txt`。
安装依赖需要联网；此包不包含 Python 安装器和离线 wheel。

## 查看原效果 / 重新运行
无需计算即可打开 `results/` 下原始 PNG/PDF/SVG 和指标 CSV；这些文件保持原字节。
```powershell
python -m visualization.synthetic
python -m visualization.neural
python -m experiments.synthetic.run --trials 50 --workers 6
python -m experiments.neural.run --stage all
python -m experiments.dandi_000688.run data/dandi_000688/raw results/dandi_000688
python -m experiments.panoptic.run data/panoptic/171204_pose2 results/panoptic
```
运行会写入结果目录，建议保留一份原压缩包以便比较。
数据已随包附带，包括 DANDI 的两份 NWB 和 Panoptic 数据；细节见对应实验 README。
重新生成 MATLAB 作者基线需要另行安装 MATLAB 及相应工具箱；现有基线已经包含。
保留原随机种子和配置；跨系统、CPU、数值库或字体差异可能导致重算数值及图像像素差异。

## 本次验证的范围
原运行环境：39 passed，1 skipped；完整性检查：300 个合成输入、2 个神经输入、
105 个作者参考文件、5 个必需结果通过。未重新执行全部耗时实验，未在新机器重装依赖。
打包时逐文件 SHA256 校验，并验证 ZIP CRC。不能据此声称跨平台逐像素一致。
本地虚拟环境、缓存、临时渲染文件和外层不完整旧项目不随包发送。
第三方源码和数据沿用其原许可证及使用条款，见各资源随附说明。
