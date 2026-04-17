from typing import Optional, Tuple, List, Dict, Union, Any
from numpy.typing import ArrayLike
import os
from pathlib import Path
import logging

import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import stats
from scipy.special import digamma, polygamma
from joblib import Parallel, delayed
from statsmodels.stats.multitest import multipletests

from .diff_expression import calc_de_score

logger = logging.getLogger(__name__)

"""
Implements functions for calculating differential expression
through moderated t-statistics as defined in 
Smyth, 2004 and Phipson et al., 2016

This module greatly simplifies the full process of
- Create dummy coding design with each cluster as an experimental condition (no intercept)
- fit linear model for each gene, calculated coefficients, residuals, degrees of freedom, etc
- moderate gene expression residual variances using empirical bayes
- create a contrast and update fit for cluster pair of interest
- perform t-test on each contrast fit
by recognizing 
- coefficients are means of each cluster,
- variances and degrees of freedom can be calculated directly from mean and mean squared values

"""

def trigamma_inverse(x, tol=1e-08, iter_limit=50):
    """Newton's method to solve trigamma inverse"""
    y = 0.5 + 1 / x
    for i in range(iter_limit):
        tri = polygamma(1, y)
        diff = tri * (1 - tri / x) / polygamma(2, y)
        y += diff
        if np.max(-diff / y) < tol:
            break
    else:
        warnings.warn(
            "trigamma_inverse iteration limit ({iter_limit}) exceeded"
        )
    return y


def fit_f_dist(x: ArrayLike, df1: ArrayLike):
    """
    Method of moments to fit f-distribution
    
    Parameters
    ----------
    x: samples
    df1: degrees of freedom
    
    Returns
    -------
    df2, scale parameters for f-distribution
    """
    z = np.log(x)
    e = z - digamma(df1 / 2) + np.log(df1 / 2)

    e_mean = np.mean(e)
    e_var = np.var(e, ddof=1)

    e_var -= np.mean(polygamma(1, df1 / 2))

    if e_var > 0:
        df2 = 2 * trigamma_inverse(e_var)
        scale = np.exp(e_mean + digamma(df2 / 2) - np.log(df2 / 2))
    else:
        df2 = np.inf
        scale = np.exp(e_mean)

    return df2, scale


def moderate_variances(
        variances: pd.DataFrame,
        df: int,
    ):
    """
    Moderated variances 
    
    - Assume each gene's variance is sampled from a 
      scaled inverse chi-square prior distribution
      with degrees of freedom d0 and location s_0^2 (sigma_0 squared)
    - Fit fDist to get prior variance
    - Get posterior variance from sample variance and prior variance 
    
    
    Parameters
    ----------
    variances: sample variances (index = gene)
    df: degrees of freedom
    
    Returns
    -------
    pd.DataFrame moderated (posterior) variances
    float prior variance
    float prior degree of freedom
    """

    var = np.squeeze(variances.to_numpy())
    idxs_zero = np.where(var == 0)[0]
    if idxs_zero.size > 0:
        var[idxs_zero] += np.finfo(var.dtype).eps

    df_prior, var_prior = fit_f_dist(var, df)
    var_post = (df_prior * var_prior + df * var) / (df + df_prior)

    var_post = pd.DataFrame(var_post, index=variances.index)

    return var_post, var_prior, df_prior


def get_linear_fit_vals(cl_vars: pd.DataFrame, cl_size: Dict[Any, int]):
    """
    Directly computes sigma squared, degrees of freedom, and stdev_unscaled
    for a linear fit of clusters from cluster variances and cluster size
    """
    cl_size_v = np.asarray([cl_size[clust] for clust in cl_vars.index])
    df = cl_size_v.sum() - len(cl_size_v)
    sigma_sq = cl_vars.T @ cl_size_v / df

    stdev_unscaled = pd.DataFrame(1 / np.sqrt(cl_size_v), index=cl_vars.index)
    return sigma_sq.to_frame(), df, stdev_unscaled


def compute_de_pair_ebayes(
        cluster_a: Any,
        cluster_b: Any,
        cluster_idx: Dict[Any, int],
        cl_means_np: np.ndarray,
        cl_present_np: np.ndarray,
        sigma_sqrt: np.ndarray,
        stdev_unscaled_np: np.ndarray,
        df_total: float,
    q1_thresh: Optional[float],
    q2_thresh: Optional[float],
    qdiff_thresh: Optional[float],
    padj_thresh: Optional[float],
    lfc_thresh: Optional[float],
    present_gt_q1: Optional[np.ndarray],
    present_lt_q2: Optional[np.ndarray],
    present_min_cells: Optional[np.ndarray],
    ) -> Dict[str, Any]:
    """Compute DE statistics for a single cluster pair using array operations."""
    idx_a = cluster_idx[cluster_a]
    idx_b = cluster_idx[cluster_b]

    mean_a = cl_means_np[idx_a]
    mean_b = cl_means_np[idx_b]
    means_diff = mean_a - mean_b
    stdev_unscaled_comb = np.hypot(stdev_unscaled_np[idx_a], stdev_unscaled_np[idx_b])

    t_vals = means_diff / sigma_sqrt / stdev_unscaled_comb
    p_vals = 2 * stats.t.sf(np.abs(t_vals), df_total)
    _, p_adj, _, _ = multipletests(
        p_vals,
        method='holm',
        is_sorted=False,
    )

    q1 = cl_present_np[idx_a]
    q2 = cl_present_np[idx_b]
    up_mask = means_diff > 0
    down_mask = means_diff < 0

    if padj_thresh is not None:
        sig_mask = p_adj < padj_thresh
        up_mask &= sig_mask
        down_mask &= sig_mask
    if lfc_thresh is not None:
        abs_lfc = np.abs(means_diff)
        lfc_mask = abs_lfc > lfc_thresh
        up_mask &= lfc_mask
        down_mask &= lfc_mask
    if q1_thresh is not None and present_gt_q1 is not None:
        up_mask &= present_gt_q1[idx_a]
        down_mask &= present_gt_q1[idx_b]
    if present_min_cells is not None:
        up_mask &= present_min_cells[idx_a]
        down_mask &= present_min_cells[idx_b]
    if q2_thresh is not None and present_lt_q2 is not None:
        up_mask &= present_lt_q2[idx_b]
        down_mask &= present_lt_q2[idx_a]
    if qdiff_thresh is not None:
        qmax = np.maximum(q1, q2)
        qdiff = np.divide(np.abs(q1 - q2), qmax, out=np.zeros_like(q1), where=qmax != 0)
        abs_qdiff = np.abs(qdiff)
        qdiff_mask = abs_qdiff > qdiff_thresh
        up_mask &= qdiff_mask
        down_mask &= qdiff_mask

    up_idx = np.flatnonzero(up_mask)
    down_idx = np.flatnonzero(down_mask)
    up_score = calc_de_score(p_adj[up_idx])
    down_score = calc_de_score(p_adj[down_idx])

    return {
        'score': up_score + down_score,
        'up_score': up_score,
        'down_score': down_score,
        'up_idx': up_idx,
        'down_idx': down_idx,
    }


def process_de_pair_chunk_indexed_with_context(
        indexed_pair_chunk: Tuple[int, List[Tuple[Any, Any]]],
        cluster_idx: Dict[Any, int],
        cl_means_np: np.ndarray,
        cl_present_np: np.ndarray,
        sigma_sqrt: np.ndarray,
        stdev_unscaled_np: np.ndarray,
        df_total: float,
    q1_thresh: Optional[float],
    q2_thresh: Optional[float],
    qdiff_thresh: Optional[float],
    padj_thresh: Optional[float],
    lfc_thresh: Optional[float],
    present_gt_q1: Optional[np.ndarray],
    present_lt_q2: Optional[np.ndarray],
    present_min_cells: Optional[np.ndarray],
    ) -> Tuple[int, Dict[Tuple[Any, Any], Dict[str, Any]]]:
    """Joblib worker entrypoint with explicit context args."""
    chunk_idx, pair_chunk = indexed_pair_chunk

    de_pairs_chunk = {}
    for cluster_a, cluster_b in pair_chunk:
        de_pairs_chunk[(cluster_a, cluster_b)] = compute_de_pair_ebayes(
            cluster_a=cluster_a,
            cluster_b=cluster_b,
            cluster_idx=cluster_idx,
            cl_means_np=cl_means_np,
            cl_present_np=cl_present_np,
            sigma_sqrt=sigma_sqrt,
            stdev_unscaled_np=stdev_unscaled_np,
            df_total=df_total,
            q1_thresh=q1_thresh,
            q2_thresh=q2_thresh,
            qdiff_thresh=qdiff_thresh,
            padj_thresh=padj_thresh,
            lfc_thresh=lfc_thresh,
            present_gt_q1=present_gt_q1,
            present_lt_q2=present_lt_q2,
            present_min_cells=present_min_cells,
        )

    return chunk_idx, de_pairs_chunk


def materialize_de_pairs_chunk(
        de_pairs_chunk: Dict[Tuple[Any, Any], Dict[str, Any]],
        gene_names: np.ndarray,
    ) -> Dict[Tuple[Any, Any], Dict[str, Any]]:
    """Convert worker index arrays to final serializable DE gene lists."""
    de_pairs_materialized: Dict[Tuple[Any, Any], Dict[str, Any]] = {}
    for pair, stats in de_pairs_chunk.items():
        up_idx = stats['up_idx']
        down_idx = stats['down_idx']

        up_genes = gene_names[up_idx].tolist()
        down_genes = gene_names[down_idx].tolist()

        de_pairs_materialized[pair] = {
            'score': stats['score'],
            'up_score': stats['up_score'],
            'down_score': stats['down_score'],
            'up_genes': up_genes,
            'down_genes': down_genes,
            'up_num': len(up_genes),
            'down_num': len(down_genes),
            'num': len(up_genes) + len(down_genes),
        }

    return de_pairs_materialized


def chunk_pairs(
        pairs: List[Tuple[Any, Any]],
        n_workers: int,
    ) -> List[List[Tuple[Any, Any]]]:
    """Split pairs into balanced worker batches (roughly one batch per worker)."""
    n_chunks = max(1, min(n_workers, len(pairs)))
    base_chunk_size, remainder = divmod(len(pairs), n_chunks)

    chunks: List[List[Tuple[Any, Any]]] = []
    start_idx = 0
    for chunk_idx in range(n_chunks):
        this_chunk_size = base_chunk_size + (1 if chunk_idx < remainder else 0)
        end_idx = start_idx + this_chunk_size
        chunks.append(pairs[start_idx:end_idx])
        start_idx = end_idx

    return chunks


def de_pair_chunk_to_frame(
        pair_chunk: List[Tuple[Any, Any]],
        de_pairs_chunk: Dict[Tuple[Any, Any], Dict[str, Any]],
    ) -> pd.DataFrame:
    """Convert a chunk result into a frame while preserving the requested pair order."""
    records = []
    for cluster_a, cluster_b in pair_chunk:
        row = de_pairs_chunk[(cluster_a, cluster_b)].copy()
        row['cluster_a'] = cluster_a
        row['cluster_b'] = cluster_b
        records.append(row)
    return pd.DataFrame.from_records(records)


def make_de_pairs_table_schema() -> pa.Schema:
    """Schema used for streamed parquet output."""
    return pa.schema([
        pa.field('cluster_a', pa.string()),
        pa.field('cluster_b', pa.string()),
        pa.field('score', pa.float64()),
        pa.field('up_score', pa.float64()),
        pa.field('down_score', pa.float64()),
        pa.field('up_genes', pa.list_(pa.string())),
        pa.field('down_genes', pa.list_(pa.string())),
        pa.field('up_num', pa.int64()),
        pa.field('down_num', pa.int64()),
        pa.field('num', pa.int64()),
    ])


def frame_to_de_pairs_table(frame: pd.DataFrame) -> pa.Table:
    """Build a pyarrow table for a DE result chunk."""
    return pa.Table.from_arrays(
        [
            pa.array(frame['cluster_a'].astype(str).tolist(), type=pa.string()),
            pa.array(frame['cluster_b'].astype(str).tolist(), type=pa.string()),
            pa.array(frame['score'].tolist(), type=pa.float64()),
            pa.array(frame['up_score'].tolist(), type=pa.float64()),
            pa.array(frame['down_score'].tolist(), type=pa.float64()),
            pa.array(frame['up_genes'].tolist(), type=pa.list_(pa.string())),
            pa.array(frame['down_genes'].tolist(), type=pa.list_(pa.string())),
            pa.array(frame['up_num'].tolist(), type=pa.int64()),
            pa.array(frame['down_num'].tolist(), type=pa.int64()),
            pa.array(frame['num'].tolist(), type=pa.int64()),
        ],
        schema=make_de_pairs_table_schema(),
    )


def de_pairs_ebayes(
        pairs: List[Tuple[Any, Any]],
        cl_means: pd.DataFrame,
        cl_vars: pd.DataFrame,
        cl_present: pd.DataFrame,
        cl_size: Dict[Any, int],
        de_thresholds: Dict[str, Any],
        n_cores: Optional[int] = 1,
        parquet_path: Optional[Union[str, os.PathLike]] = None,
    ):
    """
    Computes moderated t-statistics for pairs of cluster

    Steps:
        Get sigma squared and degrees of freedom as if a linear model was fit for each gene
        Moderate all gene variances (see moderate variances for details)
        Compute cluster pair t-test p-val for all genes using moderated variances
        Adjust cluster pair pvals
        Filter and compute descore for pair
    
    Parameters
    ----------
    pairs: list of pairs of cluster names
    cl_means: dataframe with index = cluster name, columns = genes,
              values = per cluster mean of gene expression (E[X])
    cl_vars: dataframe with index = cluster name, columns = genes,
                 values = per cluster variance of gene expression
    cl_size: dict of cluster name: number of observations in cluster
    de_thresholds: thresholds for filter de
    n_cores: number of processes to use for pairwise DE computation.
        If None, uses all available CPU cores.
    parquet_path: optional output parquet path. When provided, results are
        streamed to disk with pyarrow instead of being accumulated in memory.

    Returns
    -------
    DataFrame of DE results, or parquet path when parquet output is requested.
    """
    if len(pairs) == 0:
        return pd.DataFrame()

    sigma_sq, df, stdev_unscaled = get_linear_fit_vals(cl_vars, cl_size)
    sigma_sq_post, var_prior, df_prior = moderate_variances(sigma_sq, df)

    if n_cores is None:
        n_workers = os.cpu_count() or 1
    else:
        n_workers = int(n_cores)
    if n_workers < 1:
        raise ValueError('n_cores must be >= 1 or None')
    n_workers = min(n_workers, len(pairs))

    df_total = df + df_prior
    df_pooled = np.sum(df)
    df_total = min(df_total, df_pooled)

    cluster_labels = cl_means.index.to_list()
    cluster_idx = {cluster: idx for idx, cluster in enumerate(cluster_labels)}
    cluster_sizes_v = np.asarray([cl_size[cluster] for cluster in cluster_labels], dtype=np.float64)
    cl_means_np = np.asarray(cl_means.to_numpy(), dtype=np.float64)
    cl_present_np = np.asarray(
        cl_present.reindex(index=cluster_labels, columns=cl_means.columns).to_numpy(),
        dtype=np.float64,
    )
    gene_names = cl_means.columns.to_numpy()
    sigma_sqrt = np.sqrt(
        np.asarray(np.squeeze(sigma_sq_post.reindex(cl_means.columns).to_numpy()), dtype=np.float64)
    )
    stdev_unscaled_np = np.asarray(
        np.squeeze(stdev_unscaled.reindex(cluster_labels).to_numpy()),
        dtype=np.float64,
    )

    q1_thresh = de_thresholds.get('q1_thresh')
    q2_thresh = de_thresholds.get('q2_thresh')
    cluster_size_thresh = de_thresholds.get('cluster_size_thresh')
    qdiff_thresh = de_thresholds.get('qdiff_thresh')
    padj_thresh = de_thresholds.get('padj_thresh')
    lfc_thresh = de_thresholds.get('lfc_thresh')

    present_gt_q1 = (cl_present_np > q1_thresh) if q1_thresh is not None else None
    present_lt_q2 = (cl_present_np < q2_thresh) if q2_thresh is not None else None
    present_min_cells = (
        (cl_present_np * cluster_sizes_v[:, None]) >= cluster_size_thresh
        if cluster_size_thresh is not None else None
    )

    pair_chunks = chunk_pairs(pairs, n_workers)
    total_chunks = len(pair_chunks)
    completed_chunks = 0
    if parquet_path is not None:
        parquet_path = Path(parquet_path)
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        if parquet_path.exists():
            parquet_path.unlink()
        parquet_writer = pq.ParquetWriter(parquet_path, make_de_pairs_table_schema())
    else:
        parquet_writer = None
        de_pairs = {}

    if n_workers == 1:
        for pair_chunk in pair_chunks:
            de_pairs_chunk = process_de_pair_chunk_indexed_with_context(
                (completed_chunks, pair_chunk),
                cluster_idx,
                cl_means_np,
                cl_present_np,
                sigma_sqrt,
                stdev_unscaled_np,
                df_total,
                q1_thresh,
                q2_thresh,
                qdiff_thresh,
                padj_thresh,
                lfc_thresh,
                present_gt_q1,
                present_lt_q2,
                present_min_cells,
            )[1]
            de_pairs_chunk = materialize_de_pairs_chunk(de_pairs_chunk, gene_names)
            if parquet_writer is None:
                de_pairs.update(de_pairs_chunk)
            else:
                parquet_writer.write_table(
                    frame_to_de_pairs_table(de_pair_chunk_to_frame(pair_chunk, de_pairs_chunk))
                )
            completed_chunks += 1
            logger.info('Worker finished DE chunk %d/%d', completed_chunks, total_chunks)
    else:
        indexed_chunks = list(enumerate(pair_chunks))
        chunk_results = Parallel(
            n_jobs=n_workers,
            backend='loky',
            prefer='processes',
            batch_size=1,
            max_nbytes='10M',
            return_as='generator_unordered',
        )(
            delayed(process_de_pair_chunk_indexed_with_context)(
                indexed_pair_chunk,
                cluster_idx,
                cl_means_np,
                cl_present_np,
                sigma_sqrt,
                stdev_unscaled_np,
                df_total,
                q1_thresh,
                q2_thresh,
                qdiff_thresh,
                padj_thresh,
                lfc_thresh,
                present_gt_q1,
                present_lt_q2,
                present_min_cells,
            )
            for indexed_pair_chunk in indexed_chunks
        )
        for chunk_idx, de_pairs_chunk in chunk_results:
            pair_chunk = pair_chunks[chunk_idx]
            de_pairs_chunk = materialize_de_pairs_chunk(de_pairs_chunk, gene_names)
            if parquet_writer is None:
                de_pairs.update(de_pairs_chunk)
            else:
                parquet_writer.write_table(
                    frame_to_de_pairs_table(de_pair_chunk_to_frame(pair_chunk, de_pairs_chunk))
                )
            completed_chunks += 1
            logger.info('Worker finished DE chunk %d/%d', completed_chunks, total_chunks)

    if parquet_writer is not None:
        parquet_writer.close()
        return parquet_path
    else:
        return pd.DataFrame(de_pairs).T.reindex(pd.MultiIndex.from_tuples(pairs))
