"""
Central modality and integration method profiles for the omics integration pipeline.

This module is purely configuration (no I/O, no LLMs). It describes:

- MODALITY_PROFILES: modality_subtype → preprocessing / view expectations
- INTEGRATION_METHOD_REQUIREMENTS: integration method → input view requirements
"""

from __future__ import annotations

from typing import Dict, List, Any


MODALITY_PROFILES: Dict[str, Dict[str, Any]] = {
    # Classic single-cell RNA-seq expression matrix
    "RNA": {
        "id": "RNA",
        "family": "expression_like",
        # Which conceptual data “views” the pipeline expects to be able to build
        "default_views": {
            "raw_counts_needed": True,
            "log_norm_needed": True,
            "scaled_needed": True,
            "pca_needed": True,
        },
        # Preprocessing recipe defaults (can be overridden per dataset if needed)
        "preprocessing": {
            "hvg_min_mean": 0.0125,
            "hvg_max_mean": 3.0,
            "hvg_min_disp": 0.5,
            "normalization_type": "library_size_log1p",  # counts -> CPM-like -> log1p
            "pca_n_components": 50,
            "pca_distance_metric": "euclidean",
        },
        # Methods that are considered a natural fit for this profile
        "supported_methods": ["scvi", "harmony", "mnn", "cca", "liger"],
        "notes": "Standard scRNA-seq; current default behavior should remain backward compatible.",
    },
    # ATAC gene-activity matrix (peak-to-gene scores; expression-like)
    "ATAC_gene_activity": {
        "id": "ATAC_gene_activity",
        "family": "expression_like",
        "default_views": {
            "raw_counts_needed": True,
            "log_norm_needed": True,
            "scaled_needed": True,
            "pca_needed": True,
        },
        "preprocessing": {
            "hvg_min_mean": 0.0125,
            "hvg_max_mean": 3.0,
            "hvg_min_disp": 0.5,
            "normalization_type": "library_size_log1p",
            "pca_n_components": 50,
            "pca_distance_metric": "euclidean",
        },
        "supported_methods": ["scvi", "harmony", "mnn", "cca", "liger"],
        "notes": "ATAC gene-activity matrices are treated like expression data.",
    },
    # Raw peak-count matrix (sparse genomic peaks: chr:start-end)
    "ATAC_peak": {
        "id": "ATAC_peak",
        "family": "atac_peak",
        "default_views": {
            "raw_counts_needed": True,
            # For classic peak-based ATAC, we'd often work with TF-IDF+LSI instead of log-norm+PCA.
            "log_norm_needed": False,
            "scaled_needed": False,
            "pca_needed": True,  # conceptually LSI-like components; still treated as a "pca" view.
        },
        "preprocessing": {
            "normalization_type": "tfidf_lsi",
            "lsi_n_components": 50,
            "pca_distance_metric": "cosine",
        },
        "supported_methods": ["harmony", "mnn"],
        "notes": "Future: implement TF-IDF + LSI pipeline; currently not fully implemented.",
    },
    # ADT / protein counts (CITE-seq)
    "ADT": {
        "id": "ADT",
        "family": "adt",
        "default_views": {
            "raw_counts_needed": True,
            "log_norm_needed": True,  # e.g. CLR or arcsinh
            "scaled_needed": True,
            "pca_needed": True,
        },
        "preprocessing": {
            "normalization_type": "clr",  # or "arcsinh"
            "pca_n_components": 30,
            "pca_distance_metric": "euclidean",
        },
        "supported_methods": ["harmony", "mnn", "cca", "liger"],
        "notes": "ADT is treated as low-dimensional expression-like data; no HVG step by default.",
    },
}


INTEGRATION_METHOD_REQUIREMENTS: Dict[str, Dict[str, Any]] = {
    "scvi": {
        "name": "scvi",
        "requires_raw_counts": True,
        "input_view": "raw_counts",
        "supports": ["RNA", "ATAC_gene_activity"],
        "notes": "Expects count-like input; will typically perform its own internal normalization.",
    },
    "harmony": {
        "name": "harmony",
        "requires_raw_counts": False,
        "input_view": "pca",
        "supports": ["RNA", "ATAC_gene_activity", "ATAC_peak", "ADT"],
        "notes": "Operates on a low-dimensional PCA/LSI space.",
    },
    "mnn": {
        "name": "mnn",
        "requires_raw_counts": False,
        "input_view": "pca",
        "supports": ["RNA", "ATAC_gene_activity", "ATAC_peak", "ADT"],
        "notes": "Prefers a reasonably scaled PCA/LSI representation.",
    },
    "cca": {
        "name": "cca",
        "requires_raw_counts": False,
        "input_view": "scaled",  # or PCA of scaled expression
        "supports": ["RNA", "ATAC_gene_activity", "ADT"],
        "notes": "Typically used on log-normalized, scaled expression-like data.",
    },
    "liger": {
        "name": "liger",
        "requires_raw_counts": False,
        "input_view": "log_norm",  # LIGER does its own scaling / factorization
        "supports": ["RNA", "ATAC_gene_activity", "ADT"],
        "notes": "Takes log-normalized expression-like data; performs its own factorization.",
    },
}


def resolve_modality_profile(modality_subtype: str) -> Dict[str, Any]:
    """
    Return the modality profile for a given subtype, falling back to RNA if unknown.
    """
    key = (modality_subtype or "RNA").strip()
    if key not in MODALITY_PROFILES:
        key = "RNA"
    return MODALITY_PROFILES[key]


