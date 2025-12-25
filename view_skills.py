from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import scanpy as sc
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD

from modality_profiles import resolve_modality_profile
from tools import DetailedProfiler


ViewPayload = Dict[str, Any]
ViewMeta = Dict[str, Any]


class ViewRegistry:
    """Deterministic, cacheable view builders."""

    @staticmethod
    def _ensure_state_containers(state: Dict[str, Any]) -> None:
        state.setdefault("views", {})
        state.setdefault("view_meta", {})
        state.setdefault("view_build_log", [])

    @classmethod
    def _record_log(
        cls,
        state: Dict[str, Any],
        modality: str,
        view_name: str,
        scope: str,
        status: str,
        reason: Optional[str],
        cache_hit: bool,
        duration_s: float,
    ) -> None:
        state["view_build_log"].append(
            {
                "modality": modality,
                "view": view_name,
                "scope": scope,
                "status": status,
                "reason": reason,
                "cache_hit": cache_hit,
                "duration_s": duration_s,
            }
        )

    @classmethod
    def get_view(
        cls,
        state: Dict[str, Any],
        modality: str,
        view_name: str,
        adata_hvg: Optional[sc.AnnData],
        adata_raw: Optional[sc.AnnData],
        scope: str,
    ) -> Tuple[Optional[ViewPayload], Optional[ViewMeta], Optional[str]]:
        cls._ensure_state_containers(state)
        cached = (
            state["views"]
            .get(scope, {})
            .get(modality, {})
            .get(view_name)
        )
        if cached is not None:
            meta = (
                state["view_meta"]
                .get(scope, {})
                .get(modality, {})
                .get(view_name)
            )
            cls._record_log(state, modality, view_name, scope, "cached", None, True, 0.0)
            return cached, meta, None

        start = time.perf_counter()
        payload = None
        meta = None
        error = None

        builder_map = {
            "raw_counts": cls._build_raw_counts_view,
            "log_norm": cls._build_log_norm_view,
            "scaled": cls._build_scaled_view,
            "pca": cls._build_pca_view,
        }
        builder = builder_map.get(view_name)
        if not builder:
            error = f"Unknown view '{view_name}'"
        else:
            payload, meta, error = builder(modality, adata_hvg, adata_raw)

        duration = time.perf_counter() - start

        if payload is not None and meta is not None and error is None:
            state.setdefault("views", {}).setdefault(scope, {}).setdefault(modality, {})[view_name] = payload
            state.setdefault("view_meta", {}).setdefault(scope, {}).setdefault(modality, {})[view_name] = meta
            cls._record_log(state, modality, view_name, scope, "built", None, False, duration)
            return payload, meta, None

        cls._record_log(state, modality, view_name, scope, "failed", error, False, duration)
        return None, meta, error

    @staticmethod
    def _counts_fingerprint(matrix: Any, shape: Tuple[int, int]) -> Dict[str, Any]:
        stats = DetailedProfiler._matrix_stats(matrix, shape)
        return {
            "shape": shape,
            "integer_fraction": stats.get("integer_fraction"),
            "neg_rate": stats.get("neg_rate"),
            "q99": stats.get("q99"),
            "view_guess": stats.get("view_guess"),
        }

    @classmethod
    def _build_raw_counts_view(
        cls,
        modality: str,
        adata_hvg: Optional[sc.AnnData],
        adata_raw: Optional[sc.AnnData],
    ) -> Tuple[Optional[ViewPayload], Optional[ViewMeta], Optional[str]]:
        source = adata_raw or adata_hvg
        if source is None:
            return None, None, "No AnnData available for raw counts view"

        layer_name = None
        matrix = source.X
        source_label = "X"
        if getattr(source, "raw", None) is not None:
            try:
                raw_adata = source.raw.to_adata()
                source = raw_adata
                matrix = source.X
                source_label = "raw.X"
            except Exception:
                pass
        if getattr(source, "layers", None) is not None and "counts" in source.layers:
            layer_name = "counts"
            matrix = source.layers[layer_name]
            source_label = "layers:counts"

        shape = (source.n_obs, source.n_vars)
        fingerprint = cls._counts_fingerprint(matrix, shape)

        if layer_name is not None:
            counts_adata = source.copy()
            counts_adata.X = matrix.copy()
        else:
            counts_adata = source

        meta: ViewMeta = {
            "view": "raw_counts",
            "source": source_label,
            "n_obs": source.n_obs,
            "n_vars": source.n_vars,
            "fingerprint": fingerprint,
        }
        payload: ViewPayload = {
            "adata": counts_adata,
            "layer": layer_name,
            "obsm_key": None,
        }
        return payload, meta, None

    @classmethod
    def _build_log_norm_view(
        cls,
        modality: str,
        adata_hvg: Optional[sc.AnnData],
        adata_raw: Optional[sc.AnnData],
    ) -> Tuple[Optional[ViewPayload], Optional[ViewMeta], Optional[str]]:
        profile = resolve_modality_profile(modality)
        normalization_type = (profile.get("preprocessing") or {}).get("normalization_type", "library_size_log1p")

        if normalization_type == "tfidf_lsi":
            return None, None, "log_norm view not supported for tfidf_lsi modalities"

        source = adata_raw or adata_hvg
        if source is None:
            return None, None, "No AnnData available for log_norm view"

        matrix_stats = DetailedProfiler._matrix_stats(source.X, (source.n_obs, source.n_vars))
        if matrix_stats.get("view_guess") == "log1p":
            adata_log = source.copy()
            if getattr(adata_log, "layers", None) is not None:
                adata_log.layers["log_norm"] = adata_log.X.copy()
            meta = {
                "view": "log_norm",
                "source": "X",
                "normalization_type": normalization_type,
                "input_guess": matrix_stats.get("view_guess"),
                "n_obs": adata_log.n_obs,
                "n_vars": adata_log.n_vars,
            }
            return {"adata": adata_log, "layer": "log_norm", "obsm_key": None}, meta, None

        adata_log = source.copy()
        if normalization_type == "library_size_log1p":
            sc.pp.normalize_total(adata_log, target_sum=1e4)
            sc.pp.log1p(adata_log)
        elif normalization_type == "clr":
            matrix = adata_log.X
            if sp.issparse(matrix):
                matrix = matrix.toarray()
            matrix = np.log1p(matrix)
            matrix -= matrix.mean(axis=1, keepdims=True)
            adata_log.X = matrix
        else:
            sc.pp.normalize_total(adata_log, target_sum=1e4)
            sc.pp.log1p(adata_log)
        if getattr(adata_log, "layers", None) is not None:
            adata_log.layers["log_norm"] = adata_log.X.copy()

        meta = {
            "view": "log_norm",
            "source": "normalized_log1p",
            "normalization_type": normalization_type,
            "input_guess": matrix_stats.get("view_guess"),
            "n_obs": adata_log.n_obs,
            "n_vars": adata_log.n_vars,
        }
        payload = {"adata": adata_log, "layer": "log_norm", "obsm_key": None}
        return payload, meta, None

    @classmethod
    def _build_scaled_view(
        cls,
        modality: str,
        adata_hvg: Optional[sc.AnnData],
        adata_raw: Optional[sc.AnnData],
    ) -> Tuple[Optional[ViewPayload], Optional[ViewMeta], Optional[str]]:
        base_payload, base_meta, error = cls._build_log_norm_view(modality, adata_hvg, adata_raw)
        if base_payload is None or error:
            return None, base_meta, error or "Unable to build log_norm view for scaling"

        adata_scaled = base_payload["adata"].copy()
        sc.pp.scale(adata_scaled, max_value=10)
        if getattr(adata_scaled, "layers", None) is not None:
            adata_scaled.layers["scaled"] = adata_scaled.X.copy()

        meta = {
            "view": "scaled",
            "source": "log_norm",
            "input_meta": base_meta,
            "n_obs": adata_scaled.n_obs,
            "n_vars": adata_scaled.n_vars,
        }
        payload = {"adata": adata_scaled, "layer": "scaled", "obsm_key": None}
        return payload, meta, None

    @classmethod
    def _build_lsi_view(
        cls,
        modality: str,
        adata_hvg: Optional[sc.AnnData],
        adata_raw: Optional[sc.AnnData],
    ) -> Tuple[Optional[ViewPayload], Optional[ViewMeta], Optional[str]]:
        profile = resolve_modality_profile(modality)
        preprocessing = profile.get("preprocessing") or {}
        n_comps = int(preprocessing.get("lsi_n_components", 50))

        base_payload, base_meta, error = cls._build_raw_counts_view(modality, adata_hvg, adata_raw)
        if base_payload is None or error:
            return None, base_meta, error or "Unable to build raw counts for TF-IDF"

        counts_adata = base_payload["adata"].copy()
        counts_matrix = counts_adata.X
        if sp.issparse(counts_matrix):
            counts_matrix = counts_matrix.tocsr()
            cell_sum = np.asarray(counts_matrix.sum(axis=1)).ravel()
            cell_sum[cell_sum == 0] = 1.0
            tf = counts_matrix.multiply(1.0 / cell_sum[:, None])
            feat_counts = np.asarray((counts_matrix > 0).sum(axis=0)).ravel()
            idf = np.log1p(counts_matrix.shape[0] / (1.0 + feat_counts))
            tfidf = tf.multiply(idf)
        else:
            cell_sum = counts_matrix.sum(axis=1, keepdims=True)
            cell_sum[cell_sum == 0] = 1.0
            tf = counts_matrix / cell_sum
            feat_counts = (counts_matrix > 0).sum(axis=0)
            idf = np.log1p(counts_matrix.shape[0] / (1.0 + feat_counts))
            tfidf = tf * idf

        svd = TruncatedSVD(n_components=n_comps, random_state=0)
        lsi = svd.fit_transform(tfidf)
        counts_adata.obsm["X_pca"] = lsi

        meta = {
            "view": "pca",
            "source": "tfidf_lsi",
            "n_components": n_comps,
            "input_meta": base_meta,
            "n_obs": counts_adata.n_obs,
            "n_vars": counts_adata.n_vars,
        }
        payload = {"adata": counts_adata, "layer": None, "obsm_key": "X_pca"}
        return payload, meta, None

    @classmethod
    def _build_pca_view(
        cls,
        modality: str,
        adata_hvg: Optional[sc.AnnData],
        adata_raw: Optional[sc.AnnData],
    ) -> Tuple[Optional[ViewPayload], Optional[ViewMeta], Optional[str]]:
        profile = resolve_modality_profile(modality)
        family = profile.get("family")
        preprocessing = profile.get("preprocessing") or {}

        if family == "atac_peak" and preprocessing.get("normalization_type") == "tfidf_lsi":
            return cls._build_lsi_view(modality, adata_hvg, adata_raw)

        base_payload, base_meta, error = cls._build_scaled_view(modality, adata_hvg, adata_raw)
        if base_payload is None or error:
            return None, base_meta, error or "Unable to build scaled view for PCA"

        adata_pca = base_payload["adata"].copy()
        n_comps = int(preprocessing.get("pca_n_components", 50))
        if "X_pca" not in adata_pca.obsm or adata_pca.obsm["X_pca"].shape[1] < n_comps:
            sc.tl.pca(adata_pca, n_comps=n_comps, svd_solver="arpack")

        meta = {
            "view": "pca",
            "source": "scaled",
            "n_components": n_comps,
            "input_meta": base_meta,
            "n_obs": adata_pca.n_obs,
            "n_vars": adata_pca.n_vars,
        }
        payload = {"adata": adata_pca, "layer": None, "obsm_key": "X_pca"}
        return payload, meta, None
