"""Group-conditioned optimal transport; row-wise sample convention."""
from .alignment import GCOT, TPAMIGCOT, hierarchical_alignment

__all__ = ["GCOT", "TPAMIGCOT", "hierarchical_alignment"]
