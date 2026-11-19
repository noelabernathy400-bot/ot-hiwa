#!/usr/bin/env python3
"""Aggregate the four APPROVED train-only 24-candidate inner validations.

Critical: do NOT use outer trial1 for model or parameter selection. Determine
inner random baseline from raw original force on distinct training trials.
This is one inner validation split and 250 training samples/domain; not a
nested 5-fold hyperparameter-selection certificate.
"""
import argparse,json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPS=("PCA3_PCA3","FA3_FA3","PCA5_RAW5","FA5_RAW5")
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--in-dir",type=Path,required=True)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    with np.load(a.data) as z:
        y=np.asarray(z["Y_force_5_raw"],float)
        tid=np.asarray(z["trial_id"],int)
    train_y=y[np.isin(tid,[3,4,5])]
    val_y=y[tid==2]
    assert train_y.shape==(750,5) and val_y.shape==(250,5)
    rng=np.random.default_rng(20261122)
    random_indices=rng.integers(0,len(train_y),size=(2000,len(val_y)))
    fake=np.linalg.norm(train_y[random_indices]-val_y[None,:,:],axis=2)
    rows=[];summary=[]
    for rep in REPS:
        file=a.in_dir/rep/"inner_preflight_all24.json"
        v=json.loads(file.read_text())
        assert len(v["runs"])==24
        assert v["outer_holdout_has_never_entered_any_fit_or_parameter_selection"]
        raw=v["runs"]
        eps=[r["physical_training_only_epsilon_5pct"] for r in raw if r.get("success")]
        assert len(eps)>0 and np.max(eps)-np.min(eps)<1e-12
        epsilon=eps[0]
        chance=float(np.mean(fake<=epsilon))
        good=[r for r in raw if r["qualified"]]
        ranked=sorted(good,key=lambda r:(-r["inner_physical_Hit_at_5pct"],
                                         r["inner_physical_error_mean"],
                                         r["K"],r["ADMM_mu"]))
        bestHard=next((r for r in ranked if r["method"]=="hard"),None)
        bestSoft=next((r for r in ranked if r["method"]=="soft"),None)
        for r in raw:
            rows.append({"rep":rep,"inner_training_n":250,"inner_val_n":250,
                "outer_hold":1,"inner_hold":2,"K":r["K"],"tau":r["tau"],
                "mu":r["ADMM_mu"],"method":r["method"],
                "qualified":r["qualified"],
                "hit5":r.get("inner_physical_Hit_at_5pct"),
                "chance_hit5":chance,
                "hit5_minus_chance":(
                    r["inner_physical_Hit_at_5pct"]-chance
                    if "inner_physical_Hit_at_5pct" in r else None),
                "error_mean":r.get("inner_physical_error_mean"),
                "epsilon_original_force_units":epsilon,
                "error_type":r.get("failure_type")})
        summary.append({"representation":rep,"inner_n_qualified":len(good),
             "candidate_n":24,"baseline_random_hit":chance,
             "epsilon_original_force_units":epsilon,
             "best_hard_K":bestHard["K"] if bestHard else None,
             "best_hard_mu":bestHard["ADMM_mu"] if bestHard else None,
             "best_hard_hit":bestHard["inner_physical_Hit_at_5pct"] if bestHard else None,
             "best_soft_K":bestSoft["K"] if bestSoft else None,
             "best_soft_mu":bestSoft["ADMM_mu"] if bestSoft else None,
             "best_soft_tau":bestSoft["tau"] if bestSoft else None,
             "best_soft_hit":bestSoft["inner_physical_Hit_at_5pct"] if bestSoft else None,
             "best_qualified_across_Hard_Soft":(
                 {"method":ranked[0]["method"],"K":ranked[0]["K"],
                  "mu":ranked[0]["ADMM_mu"],"tau":ranked[0]["tau"],
                  "inner_hit":ranked[0]["inner_physical_Hit_at_5pct"],
                  "inner_mean_physical_error":ranked[0]["inner_physical_error_mean"]}
                 if ranked else None)})
    with (a.out/"all_96_inner_candidates.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (a.out/"inner_only_four_representations_summary.json").write_text(json.dumps({
        "n_candidates":96,"inner_trial":2,"outer_trial_entirely_untouched":1,
        "candidate_numerical_validity_required":True,
        "no_external_model_selection_or_claim_of_generalization":True,
        "four_representations":summary
    },ensure_ascii=False,indent=2)+"\n")
    fig,axes=plt.subplots(1,4,figsize=(16,4.2),constrained_layout=True)
    for ax,item in zip(axes,summary):
        x=np.array([0,1])
        vals=[item["best_hard_hit"],item["best_soft_hit"]]
        ax.bar(x,vals,width=.5)
        ax.axhline(item["baseline_random_hit"],ls="--",color="#777",label="random")
        ax.set(xticks=x,xticklabels=["Best Hard","Best Soft"],
               title=item["representation"],ylabel="Inner Hit@5% (not outer test)")
        ax.legend(fontsize=8);ax.grid(axis="y",alpha=.16)
    fig.suptitle("One inner trial only, 250 train-domain samples, no outer test leakage")
    fig.savefig(a.out/"PILOT_inner_valid_hit_vs_random.png",dpi=165,bbox_inches="tight")
    plt.close(fig)
    md=["# 受控调参初步结果：仅内层一折，小样本预实验，不可据此宣布泛化提升","",
        "已运行四种PCA/FA三维和五维表示×每种24组，共96个原局部P/Q/Rkl-ADMM试验；候选K=4/5、τ=.25/.5/1、μ=.02/.05/.1。每个候选在750个训练窗口中独立抽250个源/目标训练模型，250个内部验证点检验，外测试trial1完全隔离；原始全部5根手指力量用于决定物理Hit@5%。",
        "真实原始力量随机匹配基线也已在同一个内部验证试次、同一个750个训练目标候选库计算，不能因为把阈值设大而宣称改善。",
        "|表示|有效参数组/24|训练内随机基线|最佳Hard Hit@5%|最佳Soft Hit@5%|内部最高分参数（未获外部验证）|",
        "|---|---:|---:|---:|---:|---|"]
    for r in summary:
        x=r["best_qualified_across_Hard_Soft"]
        choice=f"{x['method']},K={x['K']},τ={x['tau']},μ={x['mu']}" if x else "ABSTAIN"
        md.append(f"|{r['representation']}|{r['inner_n_qualified']}/24|{r['baseline_random_hit']:.2%}|{r['best_hard_hit']:.2%}|{r['best_soft_hit']:.2%}|{choice}|")
    md+=["","**重要限制**：这是单个内部试次、单种子、250点子抽样的参数**可行性预实验**。用户批准的严格完整嵌套模型选择，还需要在每个外折的全部内部训练试次上分别验证，并对新参数做最后完整1000点外折训练。这里没有执行这一昂贵的最终环节，不能用单个内层最佳值代替正式外测试结果。",
       "不同方法若在该内层预实验中更好，也可能只是该试次特例；任何调参后外测试改善需要新一轮独立验证。"]
    (a.out/"INNER_PARAM_PREFLIGHT_ZH.md").write_text("\n".join(md)+"\n",encoding="utf-8")
    print(json.dumps({"completed":True,"n":len(rows),"summary":summary},ensure_ascii=False),flush=True)
if __name__=="__main__":main()
