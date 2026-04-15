from typing import Optional, Tuple, List, Dict, Union, Any
from numpy.core.fromnumeric import var
from numpy.typing import ArrayLike
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
import os
from pathlib import Path

import warnings
import logging
import time

import numpy as np
import pandas as pd
import scanpy as sc
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import stats
from scipy.special import digamma, polygamma
from statsmodels.stats.multitest import multipletests

from .diff_expression import get_qdiff, filter_gene_stats, calc_de_score

logger = logging.getLogger(__name__)

_DE_PAIR_CONTEXT = None

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
        logger.debug(f'offsetting zero variances from zero')
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
        cl_means: pd.DataFrame,
        cl_present: pd.DataFrame,
        cl_size: Dict[Any, int],
        de_thresholds: Dict[str, Any],
        sigma_sq_post: pd.DataFrame,
        stdev_unscaled: pd.DataFrame,
        df_total: float,
    ) -> Dict[str, Any]:
    """Compute DE statistics for a single cluster pair."""
    means_diff = cl_means.loc[cluster_a] - cl_means.loc[cluster_b]
    means_diff = means_diff.to_frame()
    stdev_unscaled_comb = np.sqrt(np.sum(stdev_unscaled.loc[[cluster_a, cluster_b]] ** 2))[0]

    t_vals = means_diff / np.sqrt(sigma_sq_post) / stdev_unscaled_comb

    p_vals = 2 * stats.t.sf(np.abs(t_vals[0]), df_total)
    _, p_adj, _, _ = multipletests(
        p_vals,
        alpha=de_thresholds['padj_thresh'],
        method='holm',
    )

    de_pair_stats = pd.DataFrame(index=cl_means.columns)
    de_pair_stats['p_value'] = p_vals
    de_pair_stats['p_adj'] = p_adj
    de_pair_stats['lfc'] = means_diff
    de_pair_stats["meanA"] = cl_means.loc[cluster_a]
    de_pair_stats["meanB"] = cl_means.loc[cluster_b]
    de_pair_stats["q1"] = cl_present.loc[cluster_a]
    de_pair_stats["q2"] = cl_present.loc[cluster_b]
    de_pair_stats["qdiff"] = get_qdiff(cl_present.loc[cluster_a], cl_present.loc[cluster_b])

    de_pair_up = filter_gene_stats(
        de_stats=de_pair_stats,
        gene_type='up-regulated',
        cl1_size=cl_size[cluster_a],
        cl2_size=cl_size[cluster_b],
        **de_thresholds
    )
    up_score = calc_de_score(de_pair_up['p_adj'].values)

    de_pair_down = filter_gene_stats(
        de_stats=de_pair_stats,
        gene_type='down-regulated',
        cl1_size=cl_size[cluster_a],
        cl2_size=cl_size[cluster_b],
        **de_thresholds
    )
    down_score = calc_de_score(de_pair_down['p_adj'].values)

    return {
        'score': up_score + down_score,
        'up_score': up_score,
        'down_score': down_score,
        'up_genes': de_pair_up.index.to_list(),
        'down_genes': de_pair_down.index.to_list(),
        'up_num': len(de_pair_up.index),
        'down_num': len(de_pair_down.index),
        'num': len(de_pair_up.index) + len(de_pair_down.index)
    }


def init_de_pair_worker(
        cl_means: pd.DataFrame,
        cl_present: pd.DataFrame,
        cl_size: Dict[Any, int],
        de_thresholds: Dict[str, Any],
        sigma_sq_post: pd.DataFrame,
        stdev_unscaled: pd.DataFrame,
        df_total: float,
    ):
    """Initialize global context for DE worker processes."""
    global _DE_PAIR_CONTEXT
    _DE_PAIR_CONTEXT = {
        'cl_means': cl_means,
        'cl_present': cl_present,
        'cl_size': cl_size,
        'de_thresholds': de_thresholds,
        'sigma_sq_post': sigma_sq_post,
        'stdev_unscaled': stdev_unscaled,
        'df_total': df_total,
    }


def process_de_pair_chunk(pair_chunk: List[Tuple[Any, Any]]) -> Dict[Tuple[Any, Any], Dict[str, Any]]:
    """Compute DE stats for a chunk of cluster pairs in one worker."""
    context = _DE_PAIR_CONTEXT
    de_pairs_chunk = {}
    for cluster_a, cluster_b in pair_chunk:
        de_pairs_chunk[(cluster_a, cluster_b)] = compute_de_pair_ebayes(
            cluster_a=cluster_a,
            cluster_b=cluster_b,
            cl_means=context['cl_means'],
            cl_present=context['cl_present'],
            cl_size=context['cl_size'],
            de_thresholds=context['de_thresholds'],
            sigma_sq_post=context['sigma_sq_post'],
            stdev_unscaled=context['stdev_unscaled'],
            df_total=context['df_total'],
        )
    return de_pairs_chunk


def chunk_pairs(
        pairs: List[Tuple[Any, Any]],
        n_workers: int,
    ) -> List[List[Tuple[Any, Any]]]:
    """Split pairs into moderately sized chunks for serial or parallel execution."""
    chunk_size = max(1, math.ceil(len(pairs) / max(1, n_workers * 4)))
    return [pairs[idx:idx + chunk_size] for idx in range(0, len(pairs), chunk_size)]


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

    logger.info('Fitting Variances')
    sigma_sq, df, stdev_unscaled = get_linear_fit_vals(cl_vars, cl_size)
    logger.info('Moderating Variances')
    sigma_sq_post, var_prior, df_prior = moderate_variances(sigma_sq, df)

    logger.info(f'Comparing {len(pairs)} pairs')
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

    pair_chunks = chunk_pairs(pairs, n_workers)
    total_chunks = len(pair_chunks)
    total_pairs = len(pairs)
    completed_chunks = 0
    completed_pairs = 0
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
            de_pairs_chunk = {}
            for cluster_a, cluster_b in pair_chunk:
                de_pairs_chunk[(cluster_a, cluster_b)] = compute_de_pair_ebayes(
                    cluster_a=cluster_a,
                    cluster_b=cluster_b,
                    cl_means=cl_means,
                    cl_present=cl_present,
                    cl_size=cl_size,
                    de_thresholds=de_thresholds,
                    sigma_sq_post=sigma_sq_post,
                    stdev_unscaled=stdev_unscaled,
                    df_total=df_total,
                )
            if parquet_writer is None:
                de_pairs.update(de_pairs_chunk)
            else:
                parquet_writer.write_table(
                    frame_to_de_pairs_table(de_pair_chunk_to_frame(pair_chunk, de_pairs_chunk))
                )
            completed_chunks += 1
            completed_pairs += len(pair_chunk)
            logger.info(
                'Completed DE chunk %d/%d (%d/%d pairs, %.1f%%)',
                completed_chunks,
                total_chunks,
                completed_pairs,
                total_pairs,
                100.0 * completed_pairs / total_pairs,
            )
    else:
        logger.info(f'Using {n_workers} workers across {len(pair_chunks)} chunks')
        with ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=init_de_pair_worker,
                initargs=(
                    cl_means,
                    cl_present,
                    cl_size,
                    de_thresholds,
                    sigma_sq_post,
                    stdev_unscaled,
                    df_total,
                ),
            ) as executor:
            futures = {
                executor.submit(process_de_pair_chunk, chunk): chunk for chunk in pair_chunks
            }
            for future in as_completed(futures):
                pair_chunk = futures[future]
                de_pairs_chunk = future.result()
                if parquet_writer is None:
                    de_pairs.update(de_pairs_chunk)
                else:
                    parquet_writer.write_table(
                        frame_to_de_pairs_table(de_pair_chunk_to_frame(pair_chunk, de_pairs_chunk))
                    )
                completed_chunks += 1
                completed_pairs += len(pair_chunk)
                logger.info(
                    'Completed DE chunk %d/%d (%d/%d pairs, %.1f%%)',
                    completed_chunks,
                    total_chunks,
                    completed_pairs,
                    total_pairs,
                    100.0 * completed_pairs / total_pairs,
                )

    if parquet_writer is not None:
        parquet_writer.close()
        return parquet_path

    de_pairs_df = pd.DataFrame(de_pairs).T
    de_pairs_df = de_pairs_df.reindex(pd.MultiIndex.from_tuples(pairs))
    return de_pairs_df
