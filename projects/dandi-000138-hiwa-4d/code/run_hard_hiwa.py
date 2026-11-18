"""4D Hard HiWA using an UNMODIFIED copy of the uploaded PyHiWA implementation.

Unpaired training only; consistent held-out normalization and error denominators.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from hiwa_original import HiWA


class IdentityEmbedding:
    def fit_transform(self, x): return x


def sample_groups(train_idx, labels, n, rng):
    groups = [rng.choice(train_idx[labels[train_idx] == k], n, replace=False)
              for k in range(4)]
    return rng.permutation(np.concatenate(groups))


def scores(x, y, r, ix, iy, ref):
    paired = np.sqrt(np.mean(np.sum((x @ r.T - y) ** 2, axis=1)))
    D = cdist(x[ix] @ r.T, y[iy], metric='sqeuclidean')
    rr, cc = linear_sum_assignment(D)
    unpaired = np.sqrt(np.mean(D[rr, cc]))
    return {'paired_rmse': float(paired), 'paired_nrmse': float(paired / ref),
            'ot_rmse': float(unpaired), 'ot_nrmse': float(unpaired / ref)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clusters', required=True)
    ap.add_argument('--allow-group-supervised-diagnostic', action='store_true',
        help='Explicit opt-in to legacy paired-label group-supervised diagnostic; NOT unsupervised')
    ap.add_argument('--output', required=True)
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    ap.add_argument('--fit-per-group', type=int, default=40)
    ap.add_argument('--outer', type=int, default=140)
    ap.add_argument('--stop-tol', type=float, default=1e-9)
    ap.add_argument('--audit-tol', type=float, default=1e-2)
    ap.add_argument('--mu', type=float, default=0.05)
    ap.add_argument('--local', type=int, default=20)
    ap.add_argument('--local-sinkhorn', type=int, default=45)
    ap.add_argument('--group-sinkhorn', type=int, default=200)
    ap.add_argument('--local-gamma', type=float, default=0.1)
    ap.add_argument('--group-gamma', type=float, default=0.2)
    ap.add_argument('--ot-eval-n', type=int, default=200)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(args.clusters, allow_pickle=False)
    x = np.asarray(z['neural4'], dtype=float)
    y = np.asarray(z['hand4'], dtype=float)
    tr = np.asarray(z['train_mask'], dtype=bool)
    te = np.asarray(z['test_mask'], dtype=bool)
    lx = np.asarray(z['neural_groups'], dtype=int)
    ly = np.asarray(z['hand_groups'], dtype=int)
    # Paired source-derived labels are for visualization ONLY.
    supervised_groups = ('shared_labels' in z.files and
                         np.array_equal(lx, ly) and
                         np.array_equal(lx, np.asarray(z['shared_labels'], dtype=int)))
    if supervised_groups and not args.allow_group_supervised_diagnostic:
        raise ValueError(
            'GROUP LABEL LEAKAGE: training target groups were copied from '
            'source paired visualization labels. For strict unsupervised '
            'training use results/clustering/clustered_4d.npz with independent '
            'neural_groups/hand_groups. Reproducing old supervised diagnostic '
            'requires --allow-group-supervised-diagnostic.')
    if supervised_groups:
        print('GROUP-SUPERVISED DIAGNOSTIC (not unsupervised)', flush=True)
    assert x.shape == y.shape and x.shape[1] == 4 and (tr & te).sum() == 0
    assert np.isfinite(x[tr | te]).all() and np.isfinite(y[tr | te]).all()
    assert set(lx[tr]) == set(range(4)) and set(ly[tr]) == set(range(4))

    # One isotropic scale per domain; no behavior PCA or per-axis rewhitening.
    mx = x[tr].mean(0); my = y[tr].mean(0)
    sx = float(np.sqrt(np.mean(np.sum((x[tr] - mx) ** 2, axis=1))))
    sy = float(np.sqrt(np.mean(np.sum((y[tr] - my) ** 2, axis=1))))
    assert min(sx, sy) > 0
    x = (x - mx) / sx; y = (y - my) / sy
    xt = x[te]; yt = y[te]
    ref = float(np.sqrt(np.mean(np.sum(yt * yt, axis=1))))
    eval_rng = np.random.default_rng(20261117)
    e = min(args.ot_eval_n, len(xt))
    ix = eval_rng.choice(len(xt), size=e, replace=False)
    iy = eval_rng.choice(len(xt), size=e, replace=False)
    # No true neural/hand correspondence is used until ALL fits are complete.
    tridx = np.flatnonzero(tr)
    all_seeds = []
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        xi = sample_groups(tridx, lx, args.fit_per_group, rng)
        yi = sample_groups(tridx, ly, args.fit_per_group, rng)
        np.random.seed(seed)
        solver = HiWA(dim_red_method=IdentityEmbedding(), normalize=False,
            maxiter=args.outer, tol=args.stop_tol, mu=args.mu,
            shorn_maxiter=args.group_sinkhorn, shorn_gamma=args.group_gamma,
            sa_maxiter=args.local, sa_tol=0.01,
            sa_shorn_maxiter=args.local_sinkhorn, sa_shorn_gamma=args.local_gamma)
        original = solver._subspace_alignment_solver
        last = []

        def capture(a, b, p, t):
            r, q, c = original(a, b, p, t)
            if not(np.isfinite(r).all() and np.isfinite(q).all() and np.isfinite(c)):
                raise FloatingPointError('Non-finite local OT or rotation')
            last.append(r.copy())
            if len(last) > 16: last.pop(0)
            return r, q, c

        solver._subspace_alignment_solver = capture  # instrumentation only
        solver.fit(x[xi], lx[xi], y[yi], ly[yi],
                   X_transform=np.eye(4), Y_transform=np.eye(4))
        R = np.asarray(solver.Rg); P = np.asarray(solver.P)
        assert len(last) == 16 and np.isfinite(P).all() and np.isfinite(R).all()
        change = float(solver.diagnostics['Rg_norm'][-1])
        primal = float(max(np.linalg.norm(q-R, ord='fro') for q in last))
        marginal = float(max(np.max(abs(P.sum(0)-.25)), np.max(abs(P.sum(1)-.25))))
        orth = float(np.linalg.norm(R.T @ R - np.eye(4), ord='fro'))
        assert np.min(P) > -1e-10 and orth < 1e-7
        qual = bool(change <= args.audit_tol and primal <= args.audit_tol and marginal < 1e-5)
        m = scores(xt, yt, R, ix, iy, ref)
        record = {'seed': seed, 'iterations': len(solver.diagnostics['Rg_norm']),
            'strict_converged': qual, 'global_change': change, 'primal_max': primal,
            'dual_proxy': float(args.mu*np.sqrt(16/4)*change),
            'marginal_max': marginal, 'orthogonality_fro': orth,
            'det': float(np.linalg.det(R)), 'R': R.tolist(), 'P': P.tolist(),
            'global_trace': solver.diagnostics['Rg_norm'].tolist(), 'metrics': m}
        all_seeds.append(record)
        print(f'seed {seed}: global={change:.3g}, primal={primal:.3g}, '
              f'converged={qual}, paired_NRMSE={m["paired_nrmse"]:.5f}, '
              f'OT_NRMSE={m["ot_nrmse"]:.5f}', flush=True)

    # POSTFIT ONLY: baseline and paired oracle are evaluation diagnostics.
    baseline = scores(xt, yt, np.eye(4), ix, iy, ref)
    # POSTFIT ONLY: supervised oracle retained as a clearly labeled diagnostic;
    # never enters model selection, initialization, grouping, P, Q or R fitting.
    u, _, vt = np.linalg.svd(y[tr].T @ x[tr])
    ro = u @ vt
    oracle = scores(xt, yt, ro, ix, iy, ref)

    data = {'dataset': 'DANDI:000138', 'data_version': '0.220113.0407',
        'algorithm': 'original ZIP HiWA, unmodified',
        'original_solver_sha256': hashlib.sha256(Path(__file__).with_name('hiwa_original.py').read_bytes()).hexdigest(),
        'data_sha256': hashlib.sha256(Path(args.clusters).read_bytes()).hexdigest(),
        'parameters': vars(args), 'dim': 4, 'K': 4,
        'training_n': int(tr.sum()), 'test_n': int(te.sum()),
        'source_mean': mx.tolist(), 'target_mean': my.tolist(),
        'source_radius': sx, 'target_radius': sy, 'evaluation_target_rms': ref,
        'fit_uses_training_sample_pairing': bool(supervised_groups),
        'group_supervision_diagnostic': bool(supervised_groups),
        'fit_uses_test_data': False,
        'paired_metric': 'sqrt(mean(||R*x_i-y_i||^2)) / sqrt(mean(||y_i||^2)), held-out paired windows',
        'ot_metric': 'sqrt(minimum one-to-one test sample permutation mean(||R*x_i-y_perm(i)||^2)) / same target RMS',
        'evaluation_unpaired_subset_n': e,
        'baseline_identity': baseline,
        'supervised_train_pairing_oracle_diagnostic': oracle,
        'seed_results': all_seeds,
        'strict_converged': sum(v['strict_converged'] for v in all_seeds)}
    (out/'hard_hiwa_results.json').write_text(json.dumps(data, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
