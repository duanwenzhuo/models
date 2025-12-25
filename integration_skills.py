from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import scanpy as sc

from modality_profiles import INTEGRATION_METHOD_REQUIREMENTS
from view_skills import ViewRegistry


@dataclass(frozen=True)
class IntegrationSkill:
    name: str
    requirements: Dict[str, Any]

    def can_run(
        self,
        state: Dict[str, Any],
        modality: str,
        adata_hvg: Optional[sc.AnnData],
        adata_raw: Optional[sc.AnnData],
        scope: str,
    ) -> Tuple[bool, Optional[str]]:
        supports = self.requirements.get("supports") or []
        if modality not in supports:
            return False, f"Modality '{modality}' not supported by {self.name}"

        input_view = self.requirements.get("input_view")
        payload, meta, error = ViewRegistry.get_view(
            state=state,
            modality=modality,
            view_name=input_view,
            adata_hvg=adata_hvg,
            adata_raw=adata_raw,
            scope=scope,
        )
        if payload is None or error:
            return False, error or f"Unable to build view '{input_view}'"

        if self.requirements.get("requires_raw_counts"):
            fingerprint = (meta or {}).get("fingerprint") or {}
            integer_fraction = fingerprint.get("integer_fraction")
            neg_rate = fingerprint.get("neg_rate")
            view_guess = fingerprint.get("view_guess")
            source_label = (meta or {}).get("source")
            if adata_raw is None:
                return False, "Raw counts AnnData is missing"
            if integer_fraction is None or neg_rate is None:
                return False, "Unable to validate raw counts fingerprint"
            if integer_fraction <= 0.85 or neg_rate >= 0.01:
                return (
                    False,
                    (
                        "Raw counts fingerprint does not meet scVI requirements "
                        f"(source={source_label}, view_guess={view_guess}, "
                        f"integer_fraction={integer_fraction}, neg_rate={neg_rate})"
                    ),
                )

        return True, None

    def prepare_inputs(
        self,
        state: Dict[str, Any],
        modality: str,
        adata_hvg: Optional[sc.AnnData],
        adata_raw: Optional[sc.AnnData],
        scope: str,
        base_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[sc.AnnData], Optional[sc.AnnData], Dict[str, Any], Optional[str]]:
        input_view = self.requirements.get("input_view")
        payload, meta, error = ViewRegistry.get_view(
            state=state,
            modality=modality,
            view_name=input_view,
            adata_hvg=adata_hvg,
            adata_raw=adata_raw,
            scope=scope,
        )
        if payload is None or error:
            return None, None, base_params or {}, error

        prepared_params = dict(base_params or {})
        if input_view == "raw_counts" and payload.get("layer"):
            prepared_params.setdefault("layer", payload.get("layer"))

        if input_view in {"log_norm", "scaled", "pca"}:
            return payload["adata"], adata_raw, prepared_params, None

        if input_view == "raw_counts":
            return adata_hvg, payload["adata"], prepared_params, None

        return payload["adata"], adata_raw, prepared_params, None


INTEGRATION_SKILLS: Dict[str, IntegrationSkill] = {
    name: IntegrationSkill(name=name, requirements=reqs)
    for name, reqs in INTEGRATION_METHOD_REQUIREMENTS.items()
}
