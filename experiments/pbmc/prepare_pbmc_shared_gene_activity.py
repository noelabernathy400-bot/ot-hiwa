"""Create a common, label-free RNA/gene-activity embedding for PBMC Multiome."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from prepare_pbmc_multiome import _cell_type, _common_cells, _select_cells, _subset_matrix  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rna-h5ad", required=True, type=Path)
    parser.add_argument("--atac-h5ad", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample-cells", type=int, default=96)
    parser.add_argument("--top-genes", type=int, default=2000)
    parser.add_argument("--gene-window", type=int, default=2000)
    parser.add_argument("--embedding-dim", type=int, default=20)
    parser.add_argument("--seed", type=int, default=120)
    return parser.parse_args()


def _gene_indices(rna, count: int) -> np.ndarray:
    ranks = np.asarray(rna.var["highly_variable_rank"], dtype=float)
    valid = np.flatnonzero(np.isfinite(ranks))
    return valid[np.argsort(ranks[valid], kind="stable")[:count]]


def _peak_gene_map(rna_var, atac_var, genes: np.ndarray, window: int) -> sparse.csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    selected = rna_var.iloc[genes]
    shared_chromosomes = sorted(set(selected["chrom"].dropna()) & set(atac_var["chrom"].dropna()))
    for chromosome in shared_chromosomes:
        gene_pos = np.flatnonzero(selected["chrom"].to_numpy() == chromosome)
        peak_pos = np.flatnonzero(atac_var["chrom"].to_numpy() == chromosome)
        starts = selected.iloc[gene_pos]["chromStart"].to_numpy(dtype=int) - window
        ends = selected.iloc[gene_pos]["chromEnd"].to_numpy(dtype=int) + window
        for peak in peak_pos:
            start = int(atac_var.iloc[peak]["chromStart"])
            end = int(atac_var.iloc[peak]["chromEnd"])
            hit = gene_pos[(starts <= end) & (ends >= start)]
            rows.extend([int(peak)] * len(hit))
            cols.extend(hit.tolist())
    if not rows:
        raise ValueError("peak-to-gene mapping is empty")
    return sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(atac_var.shape[0], len(genes)),
    )


def _log_l1(matrix) -> np.ndarray:
    values = matrix.astype(np.float64).copy()
    if sparse.issparse(values):
        values.data = np.log1p(np.maximum(values.data, 0))
        return normalize(values, norm="l1", axis=1).toarray()
    return normalize(np.log1p(np.maximum(values, 0)), norm="l1", axis=1)


def main() -> None:
    args = parse_args()
    rna = ad.read_h5ad(args.rna_h5ad, backed="r")
    atac = ad.read_h5ad(args.atac_h5ad, backed="r")
    cells = _select_cells(_common_cells(rna, atac), args.sample_cells, args.seed)
    genes = _gene_indices(rna, args.top_genes)
    mapping = _peak_gene_map(rna.var, atac.var, genes, args.gene_window)
    rna_gene = _log_l1(_subset_matrix(rna, cells, genes))
    atac_gene = _log_l1(_subset_matrix(atac, cells, slice(None)) @ mapping)

    common_matrix = np.vstack([rna_gene, atac_gene])
    n_components = min(args.embedding_dim, common_matrix.shape[0] - 1, common_matrix.shape[1])
    pca = PCA(n_components=n_components, random_state=0).fit(common_matrix)
    embedding = StandardScaler().fit_transform(pca.transform(common_matrix)).astype(np.float32)
    split = len(cells)
    payload = {
        "rna_embedding": embedding[:split],
        "atac_embedding": embedding[split:],
        "cell_ids": np.asarray(cells),
        "gene_indices": genes,
        "peak_gene_edges": np.asarray([mapping.nnz], dtype=np.int64),
        "gene_window": np.asarray([args.gene_window], dtype=np.int32),
    }
    labels = _cell_type(rna, cells)
    if labels is not None:
        payload["cell_type"] = labels
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print(f"saved {args.output}; cells={len(cells)} genes={len(genes)} peak_gene_edges={mapping.nnz}")


if __name__ == "__main__":
    main()
