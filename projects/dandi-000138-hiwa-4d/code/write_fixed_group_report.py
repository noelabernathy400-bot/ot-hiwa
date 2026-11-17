"""Write synchronized research log and LaTeX metrics rows from actual outputs."""
from __future__ import annotations
import json
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[1]
    path=root/"results/source_only_fixed_groups"
    group=json.loads((path/"group_protocol.json").read_text(encoding="utf-8"))
    r=json.loads((path/"frozen_label_metrics.json").read_text(encoding="utf-8"))
    def line(name,obj):
        return (f"| {name} | {100*obj['one_nn_shared_group']:.2f}% "
                f"| {obj['MAE']:.5f} | {obj['paired_relative_MSE']:.5f} "
                f"| {obj['rMSE_rotation_proxy']:.5f} | {obj['R2']:.5f} |")
    lines=[
        "# DANDI 000138 — 单次源域 K-means 固定分组 Hard HiWA",
        "",
        "**当前正式版本；之前两域独立 K-means 的结果仅作历史归档。**",
        "4D 数据来自 DANDI 000138 0.220113.0407 的已验证四维缓存：",
        "神经 162 units → 20ms spikes → sqrt → smoothing → train zscore → PCA(4)；",
        "目标保留原生位置与速度四维，无 PCA；400/100 trials 分割。",
        "",
        "## 从始至终只有一套组",
        "",
        "仅使用训练神经数据拟合一次 K-means(K=4,n_init=20,seed=42)。",
        "将每个神经时间窗标签按同步索引原封不动赋给手部时间窗；测试标签",
        "由训练源域 K-means 预测，随后同样复制给对应测试目标。",
        "HiWA 只对源域施加其全局正交矩阵，不得重新计算标签。",
        "**注意：这一步使用真实配对信息，属于带有对应监督的分组协议，",
        "不能再宣称是严格无配对分组发现。**",
        "",
        f"源域/目标域相同的训练组大小：{group['source_group_sizes']}",
        f"源数据缓存 SHA256：{group['frozen_input_sha256']}",
        f"全局标签 SHA256：{group['label_sha256']}",
        "",
        "## Hard HiWA 原版，重新拟合",
        "",
        "调用 code/hiwa_original.py 原版，采用每域每组 40 点，",
        "算法 5 seeds 42–46，outer=140，mu=.05，group gamma=.2，local gamma=.1。",
        "种子 42 为预先指定绘图案例，不根据测试指标筛选最优图。",
        "",
        "## 留出测试结果（相同四维、相同归一化、相同配对）",
        "",
        "| 变换 | 共享标签 1NN↑ | MAE↓ | 配对 rMSE↓ | 旋转 rMSE 代理↓ | R²↑ |",
        "|---|---:|---:|---:|---:|---:|",
        line("Identity",r["before_identity"])]
    for s in r["seed_results"]:
        lines.append(line(f"Hard HiWA seed {s['seed']}",s["metrics"]))
    avg=r["statistics"]
    metric_order=("one_nn_shared_group","MAE","paired_relative_MSE","rMSE_rotation_proxy","R2")
    lines.append("| **五次均值 ± 样本标准差** | "+
        " | ".join(
          (f"{100*avg[k]['mean']:.2f}% ± {100*avg[k]['sd_sample']:.2f}%" if k=="one_nn_shared_group"
           else f"{avg[k]['mean']:.5f} ± {avg[k]['sd_sample']:.5f}") for k in metric_order
        )+" |")
    lines += [
        "",
        "## 指标定义和限制",
        "",
        "- 1NN：目标测试点查询对齐源域测试点中的最近邻，比较该最近源标签",
        "与目标继承标签。目标标签由已知配对提供，不能解释为独立类别识别准确率。",
        "- MAE：测试配对样本四维坐标平均绝对误差，量纲为归一化坐标。",
        "- 配对 rMSE：SSE / sum(||target||²)，没有开根号，",
        "用于评价实际同步配对，不等同于作者真实旋转 rMSE。",
        "- 旋转 rMSE 代理：复用作者 _rMSE 函数，用训练配对最优正交",
        "Procrustes 作为参照旋转，而不是不存在的真值 Rgt。",
        "- 标准 R²：测试集 1-SSE/SST；原版内部 R² 算法非标准，",
        "这里修正评价公式，不动求解器。",
        "",
        "## 固定标签的实验图",
        "",
        "![源域传输前](source_before_test_six.png)",
        "",
        "![目标域同一标签](target_shared_test_six.png)",
        "",
        "![源域传输后](source_after_test_six.png)",
        "",
        "![三域六组投影直接比较](before_after_target_six_pairs.png)",
        "",
        "[第一二维横向三图](before_after_target_xy.png)；",
        "[各种指标比较](frozen_label_metrics.png)",
        "",
        "## 科学解释",
        "",
        "该协议使用已知配对传播标签，不能与旧版两域独立聚类",
        "当作仅改变优化器的公平消融。两域均按原各自 RMS 半径归一化。",
        "所有数据、组标签、旋转与逐种子结果均保存在本目录。",
        f"原版 PyHiWA SHA256：{r['original_solver_sha256']}",
        f"数值收敛标记：{[s['converged'] for s in r['seed_results']]}",
        ""]
    (path/"RUN_REPORT.md").write_text("\n".join(lines),encoding="utf-8")

    def texline(name,val):
        return (f"{name} & {100*val['one_nn_shared_group']:.2f}\\% "
                f"& {val['MAE']:.5f} & {val['paired_relative_MSE']:.5f} "
                f"& {val['rMSE_rotation_proxy']:.5f} & {val['R2']:.5f} \\\\")
    rows=[texline("初始源域",r["before_identity"])]
    rows.extend(texline(f"HiWA {s['seed']}",s["metrics"]) for s in r["seed_results"])
    rows.append(texline("五次均值",{k:v["mean"] for k,v in avg.items()}))
    (path/"metrics_rows.tex").write_text("\n".join(rows)+"\n",encoding="utf-8")

    pathlog=root/"EXPERIMENT_LOG.md"
    old=pathlog.read_text(encoding="utf-8")
    heading="## 2026-11-17 · 实验 C：只聚类源域，唯一组标签贯穿三个点云"
    summary=(
        heading+"\n\n**现行最新协议（旧两域独立 K-means 结果仅作历史）。** "
        "源 4D K-means 一次，目标同一时间窗继承标签，Hard HiWA 只旋转源坐标，"
        "前后不重新分组。配对标签继承带有监督信息。\n\n"
        f"四组训练样本数为 {group['source_group_sizes']}；详见 "
        "[正式实验结果](results/source_only_fixed_groups/RUN_REPORT.md)。\n\n"
        f"1NN：{100*avg['one_nn_shared_group']['mean']:.2f}%；"
        f"MAE：{avg['MAE']['mean']:.5f}；"
        f"配对 rMSE：{avg['paired_relative_MSE']['mean']:.5f}；"
        f"R²：{avg['R2']['mean']:.5f}。\n\n"
        "[原始源域](results/source_only_fixed_groups/source_before_test_six.png) / "
        "[继承标签目标域](results/source_only_fixed_groups/target_shared_test_six.png) / "
        "[旋转后源域](results/source_only_fixed_groups/source_after_test_six.png) / "
        "[三域直接比较](results/source_only_fixed_groups/before_after_target_six_pairs.png)。"
        "\n\n---\n\n")
    if heading not in old:
        old=old.replace("# 实验日志（持续更新）\n",
                        "# 实验日志（持续更新）\n\n"+summary,1)
    pathlog.write_text(old,encoding="utf-8")
    print("Created Markdown report, TeX metrics rows, updated experiment log")


if __name__=="__main__":
    main()
