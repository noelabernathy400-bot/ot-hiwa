"""GCOT synthetic experiment on the author's saved inputs and baselines."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key] = '1'
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.io import loadmat
from threadpoolctl import threadpool_limits
from gcot.alignment import hierarchical_alignment, weighted_measures
from gcot.grouping import soft_groups
from experiments.reference import shared_config

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/'data/processed'
SYNTHETIC_OUT = ROOT/'results/synthetic/main'
CONFIG_PATH = ROOT/'configs/experiments.json'
SPEC = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
CONFIGS = tuple(tuple(pair) for pair in SPEC['figure1']['configs'])
METHODS = tuple(SPEC['figure1']['methods'])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stamp(paths, settings):
    code = [Path(__file__), *(ROOT/'gcot').glob('*.py')]
    return hashlib.sha256(json.dumps(dict(
        files={str(p):sha(p) for p in sorted([*paths,*code])},
        settings=settings), sort_keys=True).encode()).hexdigest()


def save_npz(path, **data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_name(path.name+'.tmp.npz')
    np.savez_compressed(temp, **data)
    temp.replace(path)


def source_paths(s,d,seed):
    p = DATA/'synthetic'/f'S{s:02d}_D{d:02d}_seed{seed:02d}.mat'
    if (s,d) in ((2,2),(5,6)):
        panel = 'e' if s==2 else 'f'
        cached = ROOT/'results/synthetic/author_reference/hiwa/runs'/f'figure1{panel}_seed{seed:02d}.gcot.npz'
        return p, cached if cached.exists() else None
    if (s,d) != (5,5):
        cached = ROOT/'results/synthetic/extended/runs'/f'extended_S{s:02d}_D{d:02d}_seed{seed:02d}.python.npz'
        return p, cached
    return p, None


def synthetic_job(job):
    s,d,seed = job
    start = time.monotonic()
    p, soft_cache = source_paths(s,d,seed)
    dest = SYNTHETIC_OUT/'runs'/f'S{s:02d}_D{d:02d}_seed{seed:02d}.npz'
    cfg = shared_config()
    fingerprint = stamp([p]+([soft_cache] if soft_cache else []),asdict(cfg))
    if dest.exists():
        with np.load(dest) as z:
            if str(z['fingerprint']) == fingerprint:
                return f'skip S={s} D={d} seed={seed}'
    data = loadmat(p,simplify_cells=True)
    x,y = np.asarray(data['XX']).T,np.asarray(data['YY']).T
    assert int(data['S'])==s and x.shape[1]==d and len(x)==s*50
    truth = (data['Rgt']@data['XX']).T
    with threadpool_limits(limits=1):
        a,ca,fa = soft_groups(x,s,seed)
        b,cb,fb = soft_groups(y,s,seed+1)
        soft_aligned = None
        if soft_cache:
            with np.load(soft_cache) as old:
                if 'A' in old:
                    np.testing.assert_allclose(a,old['A'],atol=1e-12,rtol=1e-10)
                    np.testing.assert_allclose(b,old['B'],atol=1e-12,rtol=1e-10)
                    soft_aligned = old['aligned']
                else:
                    soft_aligned = old['gcot_aligned']
        validation_error = np.nan
        if soft_aligned is None or seed==1:
            soft = hierarchical_alignment(weighted_measures(x,a),weighted_measures(y,b),
                                          seed,cfg,scale=True)
            fresh_aligned = x@soft['R'].T
            if soft_aligned is not None:
                validation_error = float(np.max(np.abs(fresh_aligned-soft_aligned)))
                np.testing.assert_allclose(fresh_aligned,soft_aligned,atol=1e-9,rtol=1e-8)
            soft_aligned = fresh_aligned
    den = np.sum(truth**2)
    baseline = dict(zip(np.atleast_1d(data['methods']),np.atleast_1d(data['baseline_errors'])))
    errors = np.array([np.sum((soft_aligned-truth)**2)/den,
                       *[baseline[k] for k in METHODS[1:]]])
    if not np.isfinite(errors).all():
        raise RuntimeError(f'Nonfinite errors for {job}: {errors}')
    save_npz(dest, errors=errors, methods=METHODS, aligned=soft_aligned,
             source_prototypes=ca,target_prototypes=cb,
             source_group_fit=json.dumps(fa),target_group_fit=json.dumps(fb),
             fingerprint=fingerprint, input_path=str(p), input_sha=sha(p),
             soft_cache_path=str(soft_cache), soft_cache_validation_max_error=validation_error,
             solver_json=json.dumps(asdict(cfg)),hiwa_scaling=True,
             S=s,D=d,seed=seed,seconds=time.monotonic()-start)
    return f'done S={s} D={d} seed={seed} GCOT={errors[0]:.6g} seconds={time.monotonic()-start:.1f}'


def run_synthetic(trials=50, workers=4):
    SYNTHETIC_OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        first = [pool.submit(synthetic_job, (s, d, 1)) for s, d in CONFIGS]
        for future in as_completed(first):
            print(future.result(), flush=True)
        jobs = [pool.submit(synthetic_job, (s, d, seed))
                for seed in range(2, trials + 1) for s, d in CONFIGS]
        for future in as_completed(jobs):
            print(future.result(), flush=True)
