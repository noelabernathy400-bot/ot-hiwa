from __future__ import annotations

import numpy as np

from hiwa.hiwa import HiWA


def test_hard_hiwa_emits_common_convergence_diagnostics() -> None:
    values = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    labels = np.asarray([0, 0, 1, 1])
    model = HiWA(
        normalize=False,
        maxiter=6,
        tol=10.0,
        shorn_maxiter=20,
        sa_maxiter=2,
        sa_shorn_maxiter=20,
    )
    model.fit(values, labels, values, labels)
    diagnostics = model.diagnostics
    lengths = {
        len(diagnostics[key])
        for key in (
            "Rg_norm",
            "admm_primal_residual",
            "admm_dual_residual",
            "transport_objective",
            "group_sinkhorn_marginal_error",
        )
    }
    assert lengths == {6}
    assert np.isfinite(diagnostics["admm_primal_residual"]).all()
    assert np.isfinite(diagnostics["admm_dual_residual"]).all()
