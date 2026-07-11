"""Public API for the component-conditioned HiWA extensions."""

from .cc_hiwa import (
    barycentric_projection,
    compatibility_from_group_transport,
    component_conditioned_cost,
    coupling_entropy,
    fit_cc_hiwa,
    soft_representatives,
    squared_euclidean_cost,
)

__all__ = [
    "barycentric_projection",
    "compatibility_from_group_transport",
    "component_conditioned_cost",
    "coupling_entropy",
    "fit_cc_hiwa",
    "soft_representatives",
    "squared_euclidean_cost",
]
