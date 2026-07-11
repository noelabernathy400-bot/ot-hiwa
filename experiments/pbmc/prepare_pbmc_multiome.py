"""Prepare PBMC RNA-ATAC data for the CC-HiWA interface.

This script intentionally starts from local h5ad files. It does not silently
download large data. First planned source:

- 10x-Multiome-Pbmc10k-RNA.h5ad
- 10x-Multiome-Pbmc10k-ATAC.h5ad

Output NPZ fields:

- rna_embedding
- atac_embedding
- rna_assignments
- atac_assignments
- cell_ids
- cell_type (optional, if found)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import normalize, StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rna-h5ad", required=True, type=Path)
    parser.add_argument("--atac-h5ad", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample-cells", type=int, default=1000)
    parser.add_argument("--top-rna-features", type=int, default=5000)
    parser.add_argument("--top-atac-features", type=int, default=20000)
    parser.add_argument("--embedding-dim", type=int, default=20)
    parser.add_argument("--components", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _load_anndata(path: Path):
    try:
        import anndata as ad
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: anndata. Install anndata/scanpy before reading h5ad files."
        ) from exc
    if not path.exists():
        raise FileNotFoundError(path)
    return ad.read_h5ad(path, backed="r")


def _select_features(adata, top_features: int, mode: str) -> np.ndarray | slice:
    if top_features <= 0 or top_features >= adata.n_vars:
        return slice(None)
    var = adata.var
    if mode == "rna" and "highly_variable_rank" in var:
        ranks = np.asarray(var["highly_variable_rank"], dtype=float)
        finite = np.isfinite(ranks)
        if np.any(finite):
            candidate = np.where(finite)[0]
            order = candidate[np.argsort(ranks[finite], kind="stable")]
            return np.sort(order[:top_features])
    if "n_counts" in var:
        counts = np.asarray(var["n_counts"], dtype=float)
        order = np.argsort(-counts, kind="stable")
        return np.sort(order[:top_features])
    return np.arange(min(top_features, adata.n_vars))


def _feature_count(features: np.ndarray | slice, total: int) -> int:
    if isinstance(features, slice):
        return total
    return int(len(features))


def _cell_positions(adata, cells: list[str]) -> np.ndarray:
    position_by_cell = {str(cell): index for index, cell in enumerate(adata.obs_names)}
    return np.asarray([position_by_cell[cell] for cell in cells], dtype=np.int64)


def _subset_backed_csr_matrix(adata, row_positions: np.ndarray, features: np.ndarray | slice):
    filename = getattr(adata, "filename", None)
    if filename is None:
        raise ValueError("backed AnnData object does not expose filename")
    with h5py.File(filename, "r") as handle:
        x_group = handle["X"]
        data_ds = x_group["data"]
        indices_ds = x_group["indices"]
        indptr_ds = x_group["indptr"]
        if isinstance(features, slice):
            feature_map = None
            n_features = adata.n_vars
        else:
            feature_map = np.full(adata.n_vars, -1, dtype=np.int64)
            feature_map[features] = np.arange(len(features), dtype=np.int64)
            n_features = len(features)
        data_parts = []
        index_parts = []
        indptr = [0]
        for row in row_positions:
            start = int(indptr_ds[row])
            end = int(indptr_ds[row + 1])
            row_data = np.asarray(data_ds[start:end], dtype=np.float32)
            row_indices = np.asarray(indices_ds[start:end], dtype=np.int64)
            if feature_map is not None:
                remapped = feature_map[row_indices]
                keep = remapped >= 0
                row_data = row_data[keep]
                row_indices = remapped[keep]
            data_parts.append(row_data)
            index_parts.append(row_indices.astype(np.int32, copy=False))
            indptr.append(indptr[-1] + len(row_data))
    if data_parts:
        data = np.concatenate(data_parts).astype(np.float32, copy=False)
        indices = np.concatenate(index_parts).astype(np.int32, copy=False)
    else:
        data = np.asarray([], dtype=np.float32)
        indices = np.asarray([], dtype=np.int32)
    return sparse.csr_matrix(
        (data, indices, np.asarray(indptr, dtype=np.int32)),
        shape=(len(row_positions), n_features),
        dtype=np.float32,
    )


def _subset_matrix(adata, cells: list[str], features: np.ndarray | slice):
    row_positions = _cell_positions(adata, cells)
    if getattr(adata, "isbacked", False):
        return _subset_backed_csr_matrix(adata, row_positions, features)
    matrix = adata[row_positions, features].X
    if sparse.issparse(matrix):
        return matrix.astype(np.float32)
    return np.asarray(matrix, dtype=np.float32)


def _common_cells(rna, atac) -> list[str]:
    rna_cells = [str(item) for item in rna.obs_names]
    atac_cells = [str(item) for item in atac.obs_names]
    atac_set = set(atac_cells)
    return [cell for cell in rna_cells if cell in atac_set]


def _select_cells(cells: list[str], sample_cells: int, seed: int) -> list[str]:
    if sample_cells <= 0 or sample_cells >= len(cells):
        return cells
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(cells), size=sample_cells, replace=False))
    return [cells[index] for index in indices]


def _log1p_nonnegative(matrix):
    if sparse.issparse(matrix):
        values = matrix.copy().astype(np.float32)
        values.data = np.log1p(np.maximum(values.data, 0.0))
        return values
    values = np.asarray(matrix, dtype=np.float32)
    return np.log1p(np.maximum(values, 0.0))


def _embedding(matrix, dim: int, method: str) -> np.ndarray:
    values = _log1p_nonnegative(matrix)
    values = normalize(values, norm="l1", axis=1)
    if method == "svd" or sparse.issparse(values):
        model = TruncatedSVD(n_components=dim, random_state=0)
        embedded = model.fit_transform(values)
    else:
        model = PCA(n_components=dim, random_state=0)
        embedded = model.fit_transform(values)
    return StandardScaler().fit_transform(embedded).astype(np.float32)


def _soft_assignments(embedding: np.ndarray, components: int, temperature: float, seed: int) -> np.ndarray:
    if components <= 0:
        raise ValueError("components must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    kmeans = KMeans(n_clusters=components, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(embedding)
    centers = kmeans.cluster_centers_
    distances = np.sum((embedding[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    logits = -(distances - distances.min(axis=1, keepdims=True)) / temperature
    weights = np.exp(logits)
    assignments = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    return assignments.astype(np.float32)


def _cell_type(rna, cells: list[str]) -> np.ndarray | None:
    for key in ("cell_type", "celltype", "celltypes", "annotation", "seurat_annotations"):
        if key in rna.obs:
            return np.asarray(rna.obs.loc[cells, key].astype(str).values)
    return None


def main() -> None:
    args = parse_args()
    rna = _load_anndata(args.rna_h5ad)
    atac = _load_anndata(args.atac_h5ad)
    common = _common_cells(rna, atac)
    if not common:
        raise SystemExit("No shared cell IDs found between RNA and ATAC h5ad files.")
    cells = _select_cells(common, args.sample_cells, args.seed)
    rna_features = _select_features(rna, args.top_rna_features, mode="rna")
    atac_features = _select_features(atac, args.top_atac_features, mode="atac")
    rna_matrix = _subset_matrix(rna, cells, rna_features)
    atac_matrix = _subset_matrix(atac, cells, atac_features)
    rna_embedding = _embedding(rna_matrix, args.embedding_dim, method="svd")
    atac_embedding = _embedding(atac_matrix, args.embedding_dim, method="svd")
    rna_assignments = _soft_assignments(
        rna_embedding,
        args.components,
        args.temperature,
        args.seed,
    )
    atac_assignments = _soft_assignments(
        atac_embedding,
        args.components,
        args.temperature,
        args.seed,
    )
    payload: dict[str, Any] = {
        "rna_embedding": rna_embedding,
        "atac_embedding": atac_embedding,
        "rna_assignments": rna_assignments,
        "atac_assignments": atac_assignments,
        "cell_ids": np.asarray(cells),
        "top_rna_features": np.asarray([_feature_count(rna_features, rna.n_vars)], dtype=np.int32),
        "top_atac_features": np.asarray([_feature_count(atac_features, atac.n_vars)], dtype=np.int32),
    }
    cell_type = _cell_type(rna, cells)
    if cell_type is not None:
        payload["cell_type"] = cell_type
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print(f"saved {args.output}")
    print(f"cells={len(cells)} embedding_dim={args.embedding_dim} components={args.components}")
    print(
        "features="
        f"rna:{_feature_count(rna_features, rna.n_vars)}/{rna.n_vars} "
        f"atac:{_feature_count(atac_features, atac.n_vars)}/{atac.n_vars}"
    )
    print(f"cell_type={'yes' if cell_type is not None else 'no'}")


if __name__ == "__main__":
    main()
