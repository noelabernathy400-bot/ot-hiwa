"""Prepare graph-informed PBMC RNA-ATAC gene-activity inputs for CC-HiWA.

This is the first GI-CC-HiWA adapter. It builds a simple peak-gene graph from
genomic coordinates, converts ATAC peaks into gene activity, and outputs the
same NPZ interface consumed by run_cc_hiwa_pbmc.py.

The script uses no cell type labels for preprocessing, prototype learning, or
alignment. Labels are copied only for optional final evaluation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from scipy.linalg import orthogonal_procrustes
from sklearn.cluster import KMeans

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from prepare_pbmc_multiome import (  # noqa: E402
    _cell_type,
    _common_cells,
    _embedding,
    _load_anndata,
    _select_cells,
    _subset_matrix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rna-h5ad", required=True, type=Path)
    parser.add_argument("--atac-h5ad", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample-cells", type=int, default=1000)
    parser.add_argument("--top-genes", type=int, default=2000)
    parser.add_argument("--window-bp", type=int, default=50_000)
    parser.add_argument("--embedding-dim", type=int, default=20)
    parser.add_argument("--components", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prototype-mode", choices=["independent", "shared_procrustes"], default="shared_procrustes")
    return parser.parse_args()


def _select_rna_genes(rna, top_genes: int) -> np.ndarray:
    var = rna.var
    required = {"gene_name", "chrom", "chromStart", "chromEnd"}
    missing = required.difference(var.columns)
    if missing:
        raise ValueError(f"RNA var missing required columns: {sorted(missing)}")
    valid = (
        var["gene_name"].notna()
        & var["chrom"].notna()
        & var["chromStart"].notna()
        & var["chromEnd"].notna()
    )
    if "highly_variable_rank" in var:
        ranks = np.asarray(var["highly_variable_rank"], dtype=float)
        finite = np.isfinite(ranks) & np.asarray(valid, dtype=bool)
        if np.any(finite):
            candidates = np.where(finite)[0]
            ordered = candidates[np.argsort(ranks[finite], kind="stable")]
        else:
            ordered = np.where(np.asarray(valid, dtype=bool))[0]
    elif "n_counts" in var:
        counts = np.asarray(var["n_counts"], dtype=float)
        candidates = np.where(np.asarray(valid, dtype=bool))[0]
        ordered = candidates[np.argsort(-counts[candidates], kind="stable")]
    else:
        ordered = np.where(np.asarray(valid, dtype=bool))[0]
    selected = []
    seen_names: set[str] = set()
    for index in ordered:
        name = str(var.iloc[index]["gene_name"])
        if name in seen_names:
            continue
        selected.append(int(index))
        seen_names.add(name)
        if len(selected) >= top_genes:
            break
    if not selected:
        raise ValueError("no valid RNA genes selected")
    return np.asarray(selected, dtype=np.int64)


def _build_peak_gene_graph(atac, rna, gene_indices: np.ndarray, window_bp: int) -> tuple[sparse.csr_matrix, np.ndarray]:
    peak_var = atac.var
    gene_var = rna.var.iloc[gene_indices]
    required_peak = {"chrom", "chromStart", "chromEnd"}
    missing = required_peak.difference(peak_var.columns)
    if missing:
        raise ValueError(f"ATAC var missing required columns: {sorted(missing)}")

    edge_peak_indices: list[np.ndarray] = []
    edge_gene_indices: list[np.ndarray] = []
    peak_chrom = np.asarray(peak_var["chrom"].astype(str).values)
    peak_start = np.asarray(peak_var["chromStart"], dtype=np.int64)
    peak_end = np.asarray(peak_var["chromEnd"], dtype=np.int64)

    for chrom in sorted(set(gene_var["chrom"].astype(str).values)):
        peak_mask = peak_chrom == chrom
        chrom_peak_indices = np.where(peak_mask)[0]
        if chrom_peak_indices.size == 0:
            continue
        order = np.argsort(peak_start[chrom_peak_indices], kind="stable")
        chrom_peak_indices = chrom_peak_indices[order]
        starts = peak_start[chrom_peak_indices]
        ends = peak_end[chrom_peak_indices]
        gene_rows = np.where(gene_var["chrom"].astype(str).values == chrom)[0]
        for local_gene in gene_rows:
            gene = gene_var.iloc[local_gene]
            region_start = int(gene["chromStart"]) - window_bp
            region_end = int(gene["chromEnd"]) + window_bp
            right = np.searchsorted(starts, region_end, side="right")
            candidates = chrom_peak_indices[:right]
            if candidates.size == 0:
                continue
            keep = peak_end[candidates] >= region_start
            linked = candidates[keep]
            if linked.size:
                edge_peak_indices.append(linked.astype(np.int64, copy=False))
                edge_gene_indices.append(np.full(linked.size, local_gene, dtype=np.int64))

    if not edge_peak_indices:
        raise ValueError("peak-gene graph has no edges; increase --window-bp or check coordinates")
    peak_edges = np.concatenate(edge_peak_indices)
    gene_edges = np.concatenate(edge_gene_indices)
    selected_peaks, remapped_peak_edges = np.unique(peak_edges, return_inverse=True)
    adjacency = sparse.csr_matrix(
        (
            np.ones_like(remapped_peak_edges, dtype=np.float32),
            (remapped_peak_edges, gene_edges),
        ),
        shape=(selected_peaks.size, gene_indices.size),
        dtype=np.float32,
    )
    column_sums = np.asarray(adjacency.sum(axis=0)).ravel()
    inv_column_sums = np.divide(
        1.0,
        column_sums,
        out=np.zeros_like(column_sums, dtype=np.float32),
        where=column_sums > 0,
    )
    adjacency = adjacency @ sparse.diags(inv_column_sums.astype(np.float32))
    return adjacency.tocsr(), selected_peaks.astype(np.int64)


def _soft_assignments_from_centers(embedding: np.ndarray, centers: np.ndarray, temperature: float) -> np.ndarray:
    distances = np.sum((embedding[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    logits = -(distances - distances.min(axis=1, keepdims=True)) / temperature
    weights = np.exp(logits)
    return (weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)).astype(np.float32)


def _shared_procrustes_assignments(
    rna_embedding: np.ndarray,
    atac_embedding: np.ndarray,
    components: int,
    temperature: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    centered_rna = rna_embedding - rna_embedding.mean(axis=0, keepdims=True)
    centered_atac = atac_embedding - atac_embedding.mean(axis=0, keepdims=True)
    rotation, _ = orthogonal_procrustes(centered_rna, centered_atac)
    aligned_rna = centered_rna @ rotation
    model = KMeans(n_clusters=components, random_state=seed, n_init=10)
    model.fit(np.vstack([aligned_rna, centered_atac]))
    centers = model.cluster_centers_
    return (
        _soft_assignments_from_centers(aligned_rna, centers, temperature),
        _soft_assignments_from_centers(centered_atac, centers, temperature),
    )


def _independent_assignments(embedding: np.ndarray, components: int, temperature: float, seed: int) -> np.ndarray:
    model = KMeans(n_clusters=components, random_state=seed, n_init=10)
    model.fit(embedding)
    return _soft_assignments_from_centers(embedding, model.cluster_centers_, temperature)


def main() -> None:
    args = parse_args()
    rna = _load_anndata(args.rna_h5ad)
    atac = _load_anndata(args.atac_h5ad)
    common = _common_cells(rna, atac)
    if not common:
        raise SystemExit("No shared cell IDs found between RNA and ATAC h5ad files.")
    cells = _select_cells(common, args.sample_cells, args.seed)
    gene_indices = _select_rna_genes(rna, args.top_genes)
    adjacency, peak_indices = _build_peak_gene_graph(atac, rna, gene_indices, args.window_bp)
    rna_gene_matrix = _subset_matrix(rna, cells, gene_indices)
    atac_peak_matrix = _subset_matrix(atac, cells, peak_indices)
    atac_gene_activity = atac_peak_matrix @ adjacency

    rna_embedding = _embedding(rna_gene_matrix, args.embedding_dim, method="svd")
    atac_embedding = _embedding(atac_gene_activity, args.embedding_dim, method="svd")
    if args.prototype_mode == "shared_procrustes":
        rna_assignments, atac_assignments = _shared_procrustes_assignments(
            rna_embedding,
            atac_embedding,
            args.components,
            args.temperature,
            args.seed,
        )
    else:
        rna_assignments = _independent_assignments(rna_embedding, args.components, args.temperature, args.seed)
        atac_assignments = _independent_assignments(atac_embedding, args.components, args.temperature, args.seed)

    gene_names = np.asarray(rna.var.iloc[gene_indices]["gene_name"].astype(str).values)
    payload: dict[str, Any] = {
        "rna_embedding": rna_embedding,
        "atac_embedding": atac_embedding,
        "rna_assignments": rna_assignments,
        "atac_assignments": atac_assignments,
        "cell_ids": np.asarray(cells),
        "shared_gene_names": gene_names,
        "selected_peak_count": np.asarray([len(peak_indices)], dtype=np.int32),
        "peak_gene_edge_count": np.asarray([adjacency.nnz], dtype=np.int32),
        "window_bp": np.asarray([args.window_bp], dtype=np.int32),
        "prototype_mode": np.asarray([args.prototype_mode]),
    }
    cell_type = _cell_type(rna, cells)
    if cell_type is not None:
        payload["cell_type"] = cell_type

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print(f"saved {args.output}")
    print(
        f"cells={len(cells)} genes={len(gene_indices)} peaks={len(peak_indices)} "
        f"edges={adjacency.nnz} window_bp={args.window_bp}"
    )
    print(f"embedding_dim={args.embedding_dim} components={args.components} prototype_mode={args.prototype_mode}")
    print(f"cell_type={'yes' if cell_type is not None else 'no'}")


if __name__ == "__main__":
    main()
