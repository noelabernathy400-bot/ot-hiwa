import numpy as np
from unittest.mock import patch
from gcot.alignment import GCOT, SolverConfig
from gcot.baselines import hiwa_gmm
from gcot.preprocessing import project_for_alignment
from experiments.reference import shared_config
from evaluation.validation import validate_author_inputs


def test_shared_settings_match_both_author_figure_scripts():
    import re
    from pathlib import Path
    root=Path(__file__).resolve().parents[2]/'hiwa-matlab/code'
    expected={'maxiter':300,'tol':.1,'mu':.005,'shorn.gamma':.2,'shorn.maxiter':1000,
              'WAparam.miter':100,'WAparam.sh_gamma':.1,'WAparam.sh_miter':150,'WAparam.tol':.01}
    for name in ['figure1e.m','figure1f.m']:
        text=(root/name).read_text()
        for field,value in expected.items():
            match=re.search(r'HiWAparam\.'+re.escape(field)+r'\s*=\s*([0-9.eE+-]+)',text)
            assert match is not None and float(match.group(1))==value


def test_author_budget_and_scaling_configuration():
    c=shared_config()
    assert (c.maxiter,c.tol,c.mu)==(300,.1,.005)
    assert (c.inner_maxiter,c.inner_tol,c.sinkhorn_maxiter,c.group_sinkhorn_maxiter)==(100,.01,150,1000)
    assert (c.epsilon_group,c.epsilon_sample,c.sinkhorn_tol)==(.2,.1,0)


def test_packaged_author_inputs_and_baselines_are_complete():
    records = validate_author_inputs()
    assert len(records) == 105


def test_models_share_projection_and_solver_but_keep_distinct_groups():
    rng=np.random.default_rng(7)
    x,y=rng.normal(size=(24,3)),rng.normal(size=(26,3))
    tx,ty=rng.normal(size=(3,2)),rng.normal(size=(3,2))
    xp,yp=project_for_alignment(x,y,(tx,ty))
    config=shared_config()
    calls=[]
    def stub(source,target,seed,cfg,scale=True):
        calls.append((source,target,seed,cfg,scale))
        return {'R':np.eye(3),'P':np.eye(2)/2}
    with patch('gcot.alignment.hierarchical_alignment',stub),patch('gcot.baselines.hierarchical_alignment',stub):
        model=GCOT(2,1,config=config,hiwa_scaling=True,projections=(tx,ty)).fit(x,y)
        hiwa_gmm(x,y,2,1,config,projections=(tx,ty))
    for source,target,seed,cfg,scale in calls:
        assert seed==1 and cfg is config and scale is True
        for cloud,weights in source:
            assert all(any(np.allclose(row,p) for p in xp) for row in cloud)
        for cloud,weights in target:
            assert all(any(np.allclose(row,p) for p in yp) for row in cloud)
    assert all(len(cloud)==len(x) for cloud,_ in calls[0][0])
    assert sum(len(cloud) for cloud,_ in calls[1][0])==len(x)
    np.testing.assert_allclose(model.transform(x),x)
