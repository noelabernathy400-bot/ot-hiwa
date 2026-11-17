import importlib.util
from pathlib import Path
import numpy as np
import pytest
from sklearn.metrics import r2_score
from evaluation.metrics import regression_metrics, nearest_label_accuracy, paired_neural_truth


def test_perfect_and_wrong_predictions():
    truth = np.array([[1., 0.], [0., 2.], [-1., -1.], [2., -2.]])
    scores = regression_metrics(truth, truth)
    assert scores['r2'] == pytest.approx(1)
    assert scores['author_r2'] == pytest.approx(1)
    assert scores['rmse'] == 0 and scores['mae'] == 0 and scores['rmse_relative'] == 0
    assert regression_metrics(-truth, truth)['r2'] < 0


def test_author_parity_and_standard_r2():
    rng = np.random.default_rng(9)
    truth = rng.normal(size=(50, 3))
    predicted = truth @ np.array([[1., .2, 0], [0, .5, .1], [.1, 0, 2.]]) + .3
    path = Path(__file__).resolve().parents[2]/'PyHiWA/src/utils.py'
    spec = importlib.util.spec_from_file_location('author_metric_utils', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = regression_metrics(predicted, truth)
    assert result['author_r2'] == pytest.approx(module.eval_R2(predicted, truth), abs=1e-12)
    assert result['r2'] == pytest.approx(r2_score(truth, predicted, multioutput='variance_weighted'))
    assert result['rmse_relative'] == pytest.approx(np.sum((truth-predicted)**2)/np.sum(truth**2))


def test_nearest_neighbor_direction():
    source = np.array([[0.], [10.], [11.]])
    target = np.array([[0.], [10.]])
    # Target queries source: 2/2. Reverse queries would give 2/3.
    assert nearest_label_accuracy(source, target, [0, 1, 0], [0, 1]) == 1


def test_neural_truth_uses_saved_target_transform():
    velocities = np.array([[1., 2.], [3., 4.]])
    mean = np.array([.3, .5, .9])
    transform = np.array([[2., 0, 0], [0, 3., 0], [.2, .4, 1.]])
    expected = (np.column_stack((velocities, np.linalg.norm(velocities, axis=1)))-mean)@transform
    np.testing.assert_allclose(paired_neural_truth(velocities, mean, transform), expected[:, :2])


def test_unpaired_rows_rejected():
    with pytest.raises(ValueError):
        regression_metrics(np.ones((3, 2)), np.ones((4, 2)))


def test_complete_saved_experiments():
    from evaluation.metrics import evaluate, ROOT
    base = ROOT/'results/synthetic/author_reference/hiwa/runs'
    neural = ROOT/'results/neural/author_reference/hiwa'
    if not (base/'figure1f_seed01.gcot.npz').exists():
        pytest.skip('Saved current experiment artifacts are not present')
    report = evaluate(base, neural)
    experiments = report['experiments']
    assert [len(experiments[key]['rows']) for key in ['figure1e','figure1f','figure2a']] == [6, 6, 4]
    assert [experiments[key]['dimensions'] for key in ['figure1e','figure1f','figure2a']] == [6, 2, 2]
    for experiment in experiments.values():
        assert experiment['seed'] == 1
        for row in experiment['rows']:
            assert 0 <= row['accuracy_1nn'] <= 1
            assert np.isfinite([value for key, value in row.items() if key != 'model']).all()
