"""Evaluate saved seed=1 results only; never fit, select, or rerun a model."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.linalg import sqrtm
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
PYHIWA = ROOT.parents[2] / "HiWA" / "PyHiWA"
METHODS_SYNTHETIC = ['HiWA-SSC', 'GCOT', 'WA', 'CORAL', 'SA', 'ICP']
METHODS_NEURAL = ['GCOT', 'HiWA-GMM', 'WA', 'DAD']


def _matrix(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('Need a finite n-by-d array with at least two samples')
    return x


def _author_normal(x):
    # Literal public PyHiWA/src/utils.py normal(), not the buggy internal R2.
    covariance = np.atleast_2d(np.cov(x, rowvar=False))
    if np.linalg.eigvalsh(covariance).min() <= 0:
        raise ValueError('Author R2 requires nonsingular covariance; do not silently regularize')
    return (x-x.mean(axis=0)) @ np.linalg.inv(sqrtm(covariance))


def regression_metrics(predicted, truth):
    predicted, truth = _matrix(predicted), _matrix(truth)
    if predicted.shape != truth.shape:
        raise ValueError('Regression metrics require paired samples in the same coordinate system')
    residual = predicted-truth
    sse = np.sum(residual**2)
    sst = np.sum((truth-truth.mean(axis=0))**2)
    energy = np.sum(truth**2)
    if sst <= 0 or energy <= 0:
        raise ValueError('Undefined R2/rMSE for constant or zero-energy ground truth')
    pn, tn = _author_normal(predicted), _author_normal(truth)
    return {'r2': float(1-sse/sst),
            'author_r2': float(1-np.mean((tn-pn)**2, axis=0).sum()/np.var(tn, axis=0).sum()),
            'rmse_relative': float(sse/energy),  # Author rMSE: no square root!
            'rmse': float(np.sqrt(np.mean(residual**2))),
            'mae': float(np.mean(np.abs(residual)))}


def nearest_label_accuracy(aligned, target, source_labels, target_labels):
    aligned, target = _matrix(aligned), _matrix(target)
    source_labels, target_labels = np.asarray(source_labels).ravel(), np.asarray(target_labels).ravel()
    if aligned.shape[1] != target.shape[1] or len(aligned) != len(source_labels) or len(target) != len(target_labels):
        raise ValueError('Invalid dimensions or label counts')
    # Author neural notebook: each target queries the aligned source, k=1.
    nearest = cKDTree(aligned).query(target, k=1)[1]
    return float(np.mean(source_labels[nearest] == target_labels))


def paired_neural_truth(velocities, target_mean, target_whitening):
    velocities = _matrix(velocities)
    if velocities.shape[1] != 2:
        raise ValueError('Expected paired two-dimensional movement Xte')
    lifted = np.column_stack((velocities, np.linalg.norm(velocities, axis=1)))
    # Use Xtr-fitted preprocessing. Never fit a new transform on evaluation Xte.
    return ((lifted-target_mean) @ target_whitening)[:, :2]


def score(method, aligned, truth, target, source_labels, target_labels):
    return dict(model=method, accuracy_1nn=nearest_label_accuracy(aligned, target, source_labels, target_labels),
                **regression_metrics(aligned, truth))


def markdown_table(rows):
    lines = ['| 模型 | 1NN准确率 ↑ | R² ↑ | 作者版R² ↑ | rMSE ↓ | RMSE ↓ | MAE ↓ |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        values = [row['model'], f"{row['accuracy_1nn']:.2%}"]
        values += [f'{row[key]:.6f}' for key in ['r2','author_r2','rmse_relative','rmse','mae']]
        lines.append('| '+' | '.join(values)+' |')
    return '\n'.join(lines)


def evaluate(base, neural_base=None):
    tracked = []
    experiments = {}
    for paper_panel, script_panel in {'e':'f', 'f':'e'}.items():
        synthetic_base = base/'synthetic' if (base/'synthetic').is_dir() else base
        path = synthetic_base/f'figure1{script_panel}_seed01.mat'
        data = loadmat(path, simplify_cells=True)
        assert int(data['seed']) == 1
        truth, target = (data['Rgt']@data['XX']).T, data['YY'].T
        clouds = dict(zip(np.atleast_1d(data['methods']).tolist(), [x.T for x in data['baseline_aligned']]))
        gcot_path = path.with_suffix('.gcot.npz')
        with np.load(gcot_path) as gcot:
            clouds['GCOT'] = gcot['aligned']
        rows = [score(method, clouds[method], truth, target, data['Lx'], data['Ly'])
                for method in METHODS_SYNTHETIC]
        # Independently verify exact agreement with the previously reported rMSE.
        expected = dict(zip(np.atleast_1d(data['methods']).tolist(), data['baseline_errors']))
        for row in rows:
            if row['model'] in expected:
                np.testing.assert_allclose(row['rmse_relative'], expected[row['model']], atol=1e-12)
        experiments['figure1'+paper_panel] = dict(seed=1, dimensions=truth.shape[1],
            source_count=len(truth), target_count=len(target), rows=rows,
            regression_truth='Rgt @ XX; same source sample order',
            coordinates='full original synthetic coordinates', author_script_panel=script_panel)
        tracked += [path, gcot_path]
    neural_base = neural_base or base/'neural_single_run'
    data_path = neural_base/'data.npz'
    mat_path = PYHIWA / 'data/mihi_demo.mat'
    with np.load(data_path) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    raw = loadmat(mat_path)
    np.testing.assert_array_equal(data['source_labels'].ravel(), raw['Tte'].ravel())
    np.testing.assert_array_equal(data['target_labels'].ravel(), raw['Ttr'].ravel())
    np.testing.assert_allclose(data['target_raw'][:, :2], raw['Xtr'])
    # Check the saved preprocessing before applying it to paired evaluation movements.
    np.testing.assert_allclose((data['target_raw']-data['target_mean'])@data['target_whitening'], data['target'])
    np.testing.assert_allclose((data['source_raw']-data['source_mean'])@data['source_whitening'], data['source'])
    truth = paired_neural_truth(raw['Xte'], data['target_mean'], data['target_whitening'])
    assert len(truth) == len(data['source']) == len(raw['Yte'])
    rows = []
    for method in METHODS_NEURAL:
        path = neural_base/(method+'.npz')
        with np.load(path) as loaded:
            aligned = loaded['aligned'][:, :2]
        rows.append(score(method, aligned, truth, data['target'][:, :2], data['source_labels'], data['target_labels']))
        tracked.append(path)
    experiments['figure2a'] = dict(seed=1, dimensions=2, source_count=len(truth), target_count=len(data['target']),
        rows=rows, regression_truth='Xte paired with Yte; lift to (vx,vy,speed), apply saved Xtr whitening, take first 2 coordinates',
        coordinates='shared whitened target coordinates; dimensionless, not raw physical velocity')
    tracked += [data_path, mat_path, Path(__file__), PYHIWA / 'src/utils.py']
    hashes = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in tracked}
    return dict(experiments=experiments, inputs_sha256=hashes,
        protocol={'runs_per_model':1, 'seed':1, 'model_refitting':False,
                  '1nn':'target queries aligned source; no true-label permutation or best-run selection',
                  'r2':'1-SSE/SST; variance-weighted multioutput R2; truth centered per coordinate',
                  'author_r2':'PyHiWA/src/utils.py eval_R2: separately whiten prediction and truth before R2',
                  'rMSE':'SSE/sum(truth**2); no square root',
                  'RMSE':'sqrt(mean((prediction-truth)**2)), averaged over all samples and coordinates',
                  'MAE':'mean(abs(prediction-truth)), averaged over all samples and coordinates',
                  'figure2b':'not evaluated; omitted as requested'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path,
                        default=ROOT/'results/synthetic/author_reference/hiwa/runs')
    parser.add_argument('--neural-results', type=Path,
                        default=ROOT/'results/neural/author_reference/hiwa')
    args = parser.parse_args()
    report = evaluate(args.results, args.neural_results)
    output = ROOT/'results/synthetic/author_reference/hiwa/single_run_metrics'
    output.mkdir(exist_ok=True)
    (output/'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    titles = {'figure1e':'Figure 1e：合成高维（S=5，D=6）',
              'figure1f':'Figure 1f：合成低维（S=2，D=2）', 'figure2a':'Figure 2a：神经数据（共同二维坐标）'}
    text = '# 单次运行指标对比\n\n全部复用seed=1的已保存结果，不重新训练，不取平均或最好结果。\n'
    for key, experiment in report['experiments'].items():
        section = f"## {titles[key]}\n\n源点数={experiment['source_count']}，目标点数={experiment['target_count']}。\n\n"
        section += markdown_table(experiment['rows'])+'\n'
        (output/(key+'.md')).write_text(section)
        text += '\n'+section
    text += '''
## 口径与边界

- 1NN：每个目标点查询对齐后的源点，最近源点标签与目标真标签一致的比例；不做标签重排。
  合成数据使用全部D维，神经数据统一使用Figure 2a的共同前两维。
- R²：1-SSE/SST，SST对真值逐维去均值；等价于variance-weighted多输出R²。
  负值表示不如预测每个坐标的真值均值，未截断为0。
- 作者版R²：逐字采用PyHiWA/src/utils.py的eval_R2口径，预测和真值分别白化。
  这是额外评估归一化，不是重新拟合模型；它消除了部分均值/尺度影响，不等于标准R²。
  不采用src/hiwa.py中漏掉平方的内部_eval_R2诊断函数。
- rMSE：SSE/真值平方和，是相对均方误差，不开平方；RMSE才开平方。
- RMSE和MAE对所有点和坐标等权平均。合成误差使用真实Rgt作用于源点的结果；
  独立采样的目标YY不作为逐点回归真值。
- 神经回归真值来自与源Yte逐行配对的Xte，先提升到(vx,vy,speed)，
  再使用已保存、由Xtr拟合的目标白化矩阵，最后取前两维。未在Xte上拟合新坐标变换。
  神经RMSE/MAE是公共白化坐标下的无量纲误差，不是原始物理速度单位。
  DAD只比较最终二维输出，不使用其粗对齐第三维。
- 作者版R²的公式来自作者工具函数，但这里为公平比较统一了所有模型的评估坐标；
  神经rMSE等是新增的配对解码评估，不冒充论文原有Figure 2数值。
- 聚类对应矩阵误差不适用于所有模型，因此不列入公共表；不为没有P的模型编造P。
- 三个实验的数据、维度不同，尤其RMSE/MAE不宜跨表直接比较。
- Figure 2b不计算。全部数值只反映一次运行，不代表稳定性或统计显著性。

## 重现

在 GC-OT 目录运行 `.venv/Scripts/python.exe -m evaluation.metrics`。
完整精度、公式、输入文件SHA256见同目录metrics.json；Markdown仅显示六位小数。
'''
    text += f'\n本报告使用结果目录：`{args.results.resolve()}`；可通过 `--results` 指定历史结果目录。\n'
    (output/'REPORT.md').write_text(text)
    print(text)


if __name__ == '__main__':
    main()
