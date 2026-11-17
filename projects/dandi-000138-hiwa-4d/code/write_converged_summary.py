"""Build official XeLaTeX table from the five audited 500-iteration HiWA fits.

Strictly read verified JSON; never type in or silently alter statistical
results, cluster labels, learned R, or sample correspondences.
"""
from __future__ import annotations
import json
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[1]
    folder=root/"results/source_only_fixed_groups_outer500"
    metrics=json.loads((folder/"frozen_label_metrics.json").read_text(encoding="utf-8"))
    alignment=json.loads((folder/"hard_hiwa_results.json").read_text(encoding="utf-8"))
    old=json.loads((root/"results/source_only_fixed_groups/frozen_label_metrics.json").read_text(encoding="utf-8"))
    group=json.loads((root/"results/source_only_fixed_groups/group_protocol.json").read_text(encoding="utf-8"))

    assert metrics["source_and_target_labels_identical"]
    assert metrics["paired_labels_use_supervision"]
    assert group["target_clustering_performed"] is False
    assert len(metrics["seed_results"])==5
    assert len(alignment["seed_results"])==5
    assert all(r["converged"] for r in metrics["seed_results"])
    assert all(r["strict_converged"] for r in alignment["seed_results"])

    def row(label,v):
        return (f"{label} & {100*v['one_nn_shared_group']:.2f}\\% "
                f"& {v['MAE']:.5f} & {v['paired_relative_MSE']:.5f} "
                f"& {v['rMSE_rotation_proxy']:.5f} & {v['R2']:.5f} \\\\")
    rows=[row("传输前",metrics["before_identity"])]
    rows.extend(row(f"HiWA {x['seed']}",x["metrics"]) for x in metrics["seed_results"])
    rows.append(row("五次均值",{k:v["mean"] for k,v in metrics["statistics"].items()}))
    (folder/"metrics_rows.tex").write_text("\n".join(rows)+"\n",encoding="utf-8")

    stat=metrics["statistics"]
    lines=[
        "# 最终审计结果：同一来源 KMeans、固定组 Hard HiWA，outer≤500",
        "",
        "所有组标签严格复用 results/source_only_fixed_groups/fixed_groups_4d.npz；",
        "源域只执行过一次 KMeans，目标依同步神经样本继承源标签；",
        "目标域没有单独 KMeans，也未在变换后重新聚类。",
        "",
        "## 测试结果",
        "",
        "| 项目 | 未对齐 | HiWA 五次均值 ± 样本 SD |",
        "|---|---:|---:|"]
    for k in ("one_nn_shared_group","MAE","paired_relative_MSE","rMSE_rotation_proxy","R2"):
        a=metrics["before_identity"][k]; s=stat[k]
        lines.append(f"| {k} | {a:.6f} | {s['mean']:.6f} ± {s['sd_sample']:.6f} |")
    lines += [
        "",
        "## 收敛审计",
        "",
        "| seed | 最终 global change | 最大 local-global residual |",
        "|---:|---:|---:|"]
    for z in alignment["seed_results"]:
        lines.append(f"| {z['seed']} | {z['global_change']:.6g} | {z['primal_max']:.6g} |")
    lines.extend(["", "全部五次均符合 global<=0.01、primal<=0.01、"
                  "簇间质量边缘差<1e-5、正交误差<1e-7。",
                  "尽管五次均在 500 次外循环时达到审计阈值，"
                  "这不能证明全局最优或统计显著性。",
                  "",
                  "**科学限制：**目标簇标签通过真实同步配对继承，"
                  "所以该实验不能称为完全无配对或无监督分组。",
                  "作者原版真实旋转 Rgt 未知，所谓旋转 rMSE 为训练配对 Procrustes 代理值。",
                  "1NN 比较共同继承标签，并不是独立真实运动类别的准确率。",
                  ""])
    (folder/"FINAL_RESULTS.md").write_text("\n".join(lines),encoding="utf-8")
    print("Generated fully audited 500-iteration metric rows and Markdown from frozen data")


if __name__=="__main__":
    main()
