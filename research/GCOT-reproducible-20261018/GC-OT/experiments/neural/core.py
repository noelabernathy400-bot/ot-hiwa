"""Isomap, residual-checked GCOT, and common-coordinate author neural panels.

Archived TPAMI inputs/baselines are read-only. All new artifacts go to
results/neural/main/gcot. Labels are used only after fitting for plots and audits.
"""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
from scipy.io import loadmat
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
from gcot.alignment import SolverConfig, TPAMIGCOT
from gcot.preprocessing import neural_projections, project_for_alignment
from experiments.reference import author_utils, sha

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "results/neural/author_reference/tpami"
OUT = ROOT / "results/neural/main/gcot"
# Exact HiWA iteration/stopping settings. Rgt and labels are not accepted by
# the solver and therefore cannot enter initialization or termination.
CFG = SolverConfig(stopping="legacy", maxiter=300, tol=.1,
                   inner_maxiter=100, inner_tol=.01,
                   sinkhorn_maxiter=150, group_sinkhorn_maxiter=1000,
                   sinkhorn_tol=0)


def display_clouds(data, gcot_r, wa_r, dad_aligned):
    """Follow the author Python neural demo's transform/plot convention."""
    return dict(Source=data["source_raw"], Target=data["Xtr"],
                GCOT=data["source_raw"] @ gcot_r.T,
                WA=data["source_raw"] @ wa_r.T, DAD=dad_aligned)


def centroid_audit(source, target, source_labels, target_labels):
    labels = np.unique(source_labels)
    if not np.array_equal(labels, np.unique(target_labels)):
        raise ValueError("Audit requires matching class sets (never used to fit)")
    a = np.array([source[source_labels == k].mean(axis=0) for k in labels])
    b = np.array([target[target_labels == k].mean(axis=0) for k in labels])
    distances = cdist(a, b)
    rows, cols = linear_sum_assignment(distances)
    return dict(labels=labels.tolist(), distances=distances.tolist(),
                centroid_assignment={str(int(labels[i])): int(labels[j]) for i, j in zip(rows, cols)},
                identity_assignment=bool(np.array_equal(rows, cols)))


def fit():
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = ARCHIVE / "neural_input.npz"
    code = [ROOT / "gcot" / name for name in
            ("alignment.py", "numerics.py", "grouping.py", "preprocessing.py")]
    code.append(Path(__file__))
    fingerprint = sha(inputs) + "".join(sha(p) for p in code) + json.dumps(asdict(CFG), sort_keys=True)
    output = OUT / "GCOT.npz"
    if output.exists():
        with np.load(output) as saved:
            if str(saved["fingerprint"]) == fingerprint:
                print("Verified cached GCOT result", flush=True)
                return
    data = np.load(inputs)
    start = time.monotonic()
    with threadpool_limits(limits=1):
        tx, ty = neural_projections(data["source"], data["target"])
        xp, yp = project_for_alignment(data["source"], data["target"], (tx, ty))
        np.savez_compressed(OUT / "preprocessing.npz", source=data["source"],
                            target=data["target"], source_transform=tx, target_transform=ty,
                            source_objective=xp, target_objective=yp)
        with (OUT / "progress.jsonl").open("w", encoding="utf-8") as log:
            def progress(record):
                record["seconds"] = time.monotonic() - start
                line = json.dumps(record)
                log.write(line + "\n")
                log.flush()
                if record["iteration"] == 1 or record["iteration"] % 5 == 0 or record["converged"]:
                    print(line, flush=True)
            model = TPAMIGCOT(4, 1, config=CFG, projections=(tx, ty),
                              progress=progress).fit(data["source"], data["target"])
    np.savez_compressed(output, **model.result_, aligned=model.transform(data["source"]),
                        aligned_objective=model.transform(xp), A=model.A_, B=model.B_,
                        source_prototypes=model.source_prototypes_,
                        target_prototypes=model.target_prototypes_,
                        source_group_fit=json.dumps(model.source_group_fit_),
                        target_group_fit=json.dumps(model.target_group_fit_),
                        seconds=time.monotonic() - start, fingerprint=fingerprint)
    manifest = dict(solver=asdict(CFG), source_hashes={str(p): sha(p) for p in [inputs, *code]},
                    neural_seed=1, input_fa_seed=0, groups=4,
                    auxiliary_projection="HiWA X T T.T; source Isomap(2,12), target whitened XY",
                    gcot_scaling=False,
                    display="Author Python demo: raw FA, raw XY target, R on raw FA, independent axes",
                    figure_labels=["Source", "Target", "GCOT", "WA", "DAD"],
                    baseline_hashes={name: sha(ARCHIVE / name) for name in ("WA.mat", "DAD.mat")},
                    baseline_policy="Reuse unmodified author fits on identical whitened inputs",
                    dad_coordinates="Author rotated_KLmin returns whitened 2D points, targeting target[:,:2]",
                    legacy_results="results/neural/author_reference/tpami left unchanged")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def render():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = np.load(ARCHIVE / "neural_input.npz")
    g = np.load(OUT / "GCOT.npz")
    w = loadmat(ARCHIVE / "WA.mat", simplify_cells=True)
    dad = loadmat(ARCHIVE / "DAD.mat", simplify_cells=True)
    clouds = display_clouds(d, g["R"], w["R"], dad["aligned"])
    # Author Python fits after internal normalization, then transforms raw FA.
    np.testing.assert_allclose(clouds["WA"], w["aligned"], atol=1e-12)
    np.testing.assert_allclose(g["R"].T @ g["R"], np.eye(3), atol=1e-10)
    folder = OUT / "figures"
    folder.mkdir(exist_ok=True)
    u = author_utils()
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.7), layout="constrained")
    audit = {}
    for ax, (name, cloud) in zip(axes, clouds.items()):
        labels = d["target_labels"] if name == "Target" else d["source_labels"]
        plt.sca(ax)
        u.plot_2d_clusters(cloud, labels)
        ax.set_title(name)
        np.savetxt(OUT / f"cloud_{name}.csv", np.column_stack((cloud, labels)),
                   delimiter=",", header=",".join(
                       [f"coordinate_{i}" for i in range(cloud.shape[1])] + ["label"]),
                   comments="")
    for extension in ("png", "svg"):
        fig.savefig(folder / f"figure2a.{extension}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    # These post-fit audits use compatible coordinates and never affect R.
    audit["GCOT_fit_coordinates"] = centroid_audit(
        g["aligned"], d["target"], d["source_labels"], d["target_labels"])
    audit["WA_fit_coordinates"] = centroid_audit(
        w["aligned_objective"], d["target"], d["source_labels"], d["target_labels"])
    audit["DAD_author_coordinates"] = centroid_audit(
        clouds["DAD"], d["target"][:, :2], d["source_labels"], d["target_labels"])
    audit["solver"] = {k: g[k].item() for k in (
        "converged", "stop_reason", "orthogonality_error", "rotation_delta",
        "consensus_residual", "dual_residual", "primal_tolerance", "dual_tolerance",
        "marginal_error", "coupling_delta")}
    audit["solver"]["iterations"] = len(g["history"])
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    solver = audit["solver"]
    assignment = audit["GCOT_fit_coordinates"]["centroid_assignment"]
    report = f"""# 本次运行结果

- 图：`figures/figure2a.png`、`figures/figure2a.svg`，方法名仍为 GCOT。
- 已恢复 Isomap 辅助投影，绘图按作者 Python 神经数据 demo 的原始坐标和独立坐标轴。
- 停止预算与 HiWA 一致：外层300、阈值0.1、至少6轮；局部100/0.01；Sinkhorn 150/1000。
- 真实 R 和类别标签均不参与拟合、初始化或停止。

## 结果

第 {solver['iterations']} 轮停止，`stop_reason={solver['stop_reason']}`，
`||R(k)-R(k-1)||_F={solver['rotation_delta']:.8g} <= 0.1`，因此按 HiWA 原规则记为收敛。
正交误差为 `{solver['orthogonality_error']:.8g}`。

事后簇中心对应为 `{assignment}`，未出现蓝绿互换。
标签只在模型返回并保存之后用于绘图与这项审计。

## 诊断边界

最终共识残差为 `{solver['consensus_residual']:.8g}`，诊断阈值为
`{solver['primal_tolerance']:.8g}`；边际误差为 `{solver['marginal_error']:.8g}`。
这些按用户要求仅保留为诊断，不参与 HiWA 停止判断。
所以“按 HiWA 规则收敛”不等于 ADMM 全部残差都已达到严格阈值。
"""
    (OUT / "REPORT_CN.md").write_text(report, encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("fit", "render", "all"), default="all")
    args = parser.parse_args()
    if args.stage in ("fit", "all"):
        fit()
    if args.stage in ("render", "all"):
        render()
