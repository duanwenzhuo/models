# agents.py
import json
import logging
import os
import re
from collections import defaultdict
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
import scanpy as sc
from anndata import AnnData
from sklearn.metrics import silhouette_score
import scipy.sparse as sp

# from langchain_openai import ChatOpenAI
# from langchain.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import JsonOutputParser

import config
from tools import (
    OmicsTools,
    INTEGRATION_METHODS,
    calculate_graph_connectivity,
    METRIC_SPECS,
    compute_metric_raw,
)
from openai import OpenAI

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def build_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY") or config.OPENAI_API_KEY
    base_url = os.getenv("OPENAI_BASE_URL") or config.OPENAI_BASE_URL

    if not api_key:
        logger.error(
            "OpenAI API key is missing. Set OPENAI_API_KEY in the environment or config.OPENAI_API_KEY."
        )
        raise ValueError("OpenAI API key is required to initialize the client.")

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    return OpenAI(**client_kwargs)


client = build_openai_client()

# Integration method normalization helpers
ALLOWED_INTEGRATION_METHODS = {"scvi", "harmony", "mnn", "cca", "liger"}
METHOD_ALIAS_MAP = {
    "seurat": "cca",
    "seuratcca": "cca",
    "seurat_cca": "cca",
    "seurat-cca": "cca",
    "seurat cca": "cca",
}


def canonicalize_method_name(method_name: Any) -> Optional[str]:
    """Return the canonical integration method name or None if unsupported."""
    if not isinstance(method_name, str):
        return None
    normalized = method_name.strip().lower()
    normalized = METHOD_ALIAS_MAP.get(normalized, normalized)
    if normalized not in ALLOWED_INTEGRATION_METHODS:
        return None
    return normalized


def sanitize_filename(filename: str) -> str:
    """Sanitize filenames to avoid illegal chars and overlong paths."""
    cleaned = re.sub(r'[\\/*?:"<>|]', "_", filename)
    return cleaned[:255]


# ==========================================
# Task parsing helpers
# ==========================================
TASK_MODALITY_KEYWORDS = {
    "scrna": "RNA",
    "rna-seq": "RNA",
    "sc-rna": "RNA",
    "rna": "RNA",
    "atac": "ATAC_gene_activity",
    "atac-seq": "ATAC_gene_activity",
    "multiome": "ATAC_gene_activity",
    "spatial": "spatial",
    "visium": "spatial",
    "slide-seq": "spatial",
    "adt": "ADT",
}

PREPROCESS_KEYWORDS = {
    "harmony": "harmony",
    "mnn": "mnn",
    "seurat": "seurat",
    "cca": "cca",
    "liger": "liger",
    "scvi": "scvi",
}


class TaskIntentParser:
    """Lightweight intent parser to detect modality, preprocessing and subsets."""

    def __init__(self, intent: str):
        self.intent = intent or ""
        self.intent_lower = self.intent.lower()

    def detect_modalities(self) -> List[str]:
        matches: List[str] = []
        for key, modality in TASK_MODALITY_KEYWORDS.items():
            if key in self.intent_lower:
                matches.append(modality)
        if not matches:
            matches.append("RNA")
        return sorted(set(matches))

    def infer_preprocessing(self, modality: str) -> Dict[str, Any]:
        need_preprocess = True
        if any(k in self.intent_lower for k in ["raw already", "preprocessed", "normalized"]):
            need_preprocess = False

        suggested = None
        for key, name in PREPROCESS_KEYWORDS.items():
            if key in self.intent_lower:
                suggested = name
                break

        if suggested is None:
            if "batch" in self.intent_lower or "batch effect" in self.intent_lower:
                suggested = "harmony"
            elif "integration" in self.intent_lower:
                suggested = "scvi" if modality.upper() == "RNA" else "harmony"

        return {"preprocess": need_preprocess, "preferred_method": suggested}

    def extract_subset_preferences(self) -> Dict[str, List[str]]:
        subset: Dict[str, List[str]] = {"genes": [], "celltypes": [], "batches": []}

        gene_match = re.search(r"genes?[:=]\s*([\w,; ]+)", self.intent_lower)
        if gene_match:
            genes = [g.strip().upper() for g in re.split(r"[,;]", gene_match.group(1)) if g.strip()]
            subset["genes"] = genes

        celltype_match = re.search(r"cell(types?)?[:=]\s*([\w,; ]+)", self.intent_lower)
        if celltype_match:
            celltypes = [c.strip() for c in re.split(r"[,;]", celltype_match.group(2)) if c.strip()]
            subset["celltypes"] = celltypes

        batch_match = re.search(r"batches?[:=]\s*([\w,; ]+)", self.intent_lower)
        if batch_match:
            batches = [b.strip() for b in re.split(r"[,;]", batch_match.group(1)) if b.strip()]
            subset["batches"] = batches

        return subset


# ==========================================
# Base Agent
# ==========================================
class BaseAgent:
    def __init__(self, name: str):
        self.name = name

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        接收 state，执行逻辑，返回更新后的 state
        """
        raise NotImplementedError("Subclasses must implement run()")

# ==========================================
# Inspection Agent
# ==========================================
class InspectionAgent(BaseAgent):
    """
    读取 data_path 指向的 h5ad 文件，生成 JSON 格式的结构化检查信息，
    并通过 LLM 确定模态、批次键等。
    """

    def __init__(self):
        super().__init__("Inspection")

    def _display_first_rows(self, adata: AnnData) -> Dict[str, Any]:
        """
        Return a lightweight preview of obs/var (first 5 rows) for LLM context.
        """
        preview = {"obs_head": "", "var_head": ""}
        
        # Handle obs preview (first 5 rows)
        try:
            if isinstance(adata.obs, pd.DataFrame):
                obs_preview = adata.obs.head(5)
                preview["obs_head"] = obs_preview.to_dict(orient="list")
            elif isinstance(adata.obs, np.ndarray):
                preview["obs_head"] = adata.obs[:5].tolist()
            elif sp.issparse(adata.obs):
                preview["obs_head"] = adata.obs[:5].todense().tolist()
            else:
                preview["obs_head"] = "Unknown data type in adata.obs."
        except Exception as e:
            preview["obs_head"] = f"Error: {str(e)}"
        
        # Handle var preview (first 5 rows)
        try:
            if isinstance(adata.var, pd.DataFrame):
                var_preview = adata.var.head(5)
                preview["var_head"] = var_preview.to_dict(orient="list")
            elif isinstance(adata.var, np.ndarray):
                preview["var_head"] = adata.var[:5].tolist()
            elif sp.issparse(adata.var):
                preview["var_head"] = adata.var[:5].todense().tolist()
            else:
                preview["var_head"] = "Unknown data type in adata.var."
        except Exception as e:
            preview["var_head"] = f"Error: {str(e)}"
        
        return preview


    def _infer_modality_from_llm(self, adata: AnnData, user_intent: str) -> str:
        """
        使用 LLM 根据提供的数据和用户意图推断模态。
        """
        obs_columns = adata.obs.columns.tolist()
        var_columns = adata.var.columns.tolist()
        n_cells = adata.n_obs
        n_features = adata.n_vars

        first_rows_info = self._display_first_rows(adata)

        # 额外的诊断信息，帮助 LLM 判断数据的结构
        logger.info(f"[Inspection] obs columns: {obs_columns}")
        logger.info(f"[Inspection] var columns: {var_columns}")
        
        # 添加更多关于数据内容的信息，比如是否有 celltype 和 gene 信息
        is_rna = "celltype" in obs_columns and "gene" in var_columns
        is_atac = "peak" in var_columns and "celltype" in obs_columns

        system_prompt = (
            "You are a bioinformatics assistant. Based on the data provided, determine the modality of the dataset.\n"
            "The dataset contains the following metadata:\n"
            f"  Number of cells: {n_cells}\n"
            f"  Number of features: {n_features}\n"
            f"  Observed columns: {obs_columns}\n"
            f"  Variable columns: {var_columns}\n\n"
            f"  Obs preview (first 5 rows): {first_rows_info.get('obs_head')}\n"
            f"  Var preview (first 5 rows): {first_rows_info.get('var_head')}\n\n"
            "The modalities could be RNA, ATAC, ADT, or others. "
            "Return ONLY JSON like {\"modality\": \"RNA\"} with modality as RNA/ATAC/ADT/unknown.\n"
            
            # 结合 RNA 和 ATAC 的标识
            "Note: If the data contains gene expressions and cell types in `obs` and `var`, it is likely RNA.\n"
            "Note: If the data contains peak identifiers and cell types in `obs` and `var`, it is likely ATAC.\n"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_intent},
        ]

        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.0,
            )
            content = resp.choices[0].message.content.strip()
            cleaned = content.replace("```json", "").replace("```", "").strip()
            m = re.search(r"\{.*\}", cleaned, flags=re.S)
            payload = m.group(0) if m else cleaned
            modality_info = json.loads(payload) if payload else {}
            modality_val = modality_info.get("modality") if isinstance(modality_info, dict) else None
            logger.info(f"[Inspection] LLM returned modality guess: {modality_val}")
            return modality_val or "unknown"
        except Exception as e:
            logger.error(f"[Inspection] LLM failed to infer modality: {e}")
            return "unknown"
    def _infer_batch_key_from_llm(self, adata: AnnData, user_intent: str) -> Optional[str]:
        """
        使用 LLM 根据提供的数据和用户意图推断批次列。
        """
        obs_columns = adata.obs.columns.tolist()
        system_prompt = (
            "You are a bioinformatics assistant. Based on the data provided, determine the best batch key column.\n"
            f"The dataset observed columns: {obs_columns}\n"
            "The batch key column could be 'batch', 'batchname', or any other column that represents batch. "
            "Return ONLY JSON like {\"batch_key\": \"batchname\"} or {\"batch_key\": null}."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_intent},
        ]
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.0,
            )
            content = resp.choices[0].message.content.strip()
            cleaned = content.replace("```json", "").replace("```", "").strip()
            m = re.search(r"\{.*\}", cleaned, flags=re.S)
            payload = m.group(0) if m else cleaned
            batch_info = json.loads(payload) if payload else {}
            batch_val = batch_info.get("batch_key") if isinstance(batch_info, dict) else None
            logger.info(f"[Inspection] LLM returned batch key: {batch_val}")
            if isinstance(batch_val, str):
                return batch_val
            return None
        except Exception as e:
            logger.error(f"[Inspection] LLM failed to infer batch key: {e}")
            return None

    def inspect_adata(self, adata: AnnData, user_intent: str = "Please identify the modality and batch column") -> dict:
        """
        LLM-based inspection of the dataset to determine its properties and modality.
        """
        info: dict[str, Any] = {}

        # 基础统计
        info["n_cells"] = adata.n_obs
        info["n_features"] = adata.n_vars
        obs_cols = adata.obs.columns.tolist()
        info["obs_columns"] = obs_cols
        info["var_columns"] = adata.var.columns.tolist()

        batch_candidates = [
            c for c in obs_cols if isinstance(c, str) and "batch" in c.lower()
        ]
        info["batch_candidates"] = batch_candidates
        info["has_batch"] = len(batch_candidates) > 0

        # LLM 推断模态和批次键
        modality_guess = self._infer_modality_from_llm(adata, user_intent)
        batch_key_effective = self._infer_batch_key_from_llm(adata, user_intent)
        if batch_key_effective is None and len(batch_candidates) == 1:
            batch_key_effective = batch_candidates[0]

        # 继续收集其他元信息
        X = adata.X
        if X is None:
            info["matrix_valid"] = False
            info["nonzero_elements"] = 0
            is_integer_like = False
            sample_vals = np.array([])
        else:
            info["matrix_valid"] = True
            if sp.issparse(X):
                info["nonzero_elements"] = int(X.nnz)
                sample_vals = X.data
            else:
                info["nonzero_elements"] = int(np.count_nonzero(X))
                sample_vals = X.ravel()

            if sample_vals.size > 0:
                subset = sample_vals[: min(sample_vals.size, 5000)]
                is_integer_like = bool(np.allclose(subset, np.round(subset)))
            else:
                is_integer_like = False

        info["has_pca"] = "X_pca" in adata.obsm
        info["has_umap"] = "X_umap" in adata.obsm
        info["has_neighbors"] = ("neighbors" in adata.uns) or ("neighbors" in adata.obsp)
        info["has_raw"] = adata.raw is not None
        info["is_integer_like"] = is_integer_like

        has_log1p = False
        if X is not None and info["matrix_valid"]:
            if not is_integer_like:
                has_log1p = True
        info["has_log1p"] = has_log1p

        ready_for_integration = info["has_pca"] and info["has_neighbors"]
        info["preprocessing"] = {
            "has_raw": info["has_raw"],
            "has_pca": info["has_pca"],
            "has_umap": info["has_umap"],
            "has_neighbors": info["has_neighbors"],
            "has_log1p": info["has_log1p"],
            "is_integer_like": info["is_integer_like"],
            "ready_for_integration": ready_for_integration,
        }

        info["modality"] = modality_guess or "unknown"
        info["modality_guess"] = modality_guess or "unknown"
        info["batch_key_effective"] = batch_key_effective

        return info

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        data_path = state.get("data_path")
        if not data_path:
            msg = "[Inspection] data_path is missing in state."
            logger.error(msg)
            state["error"] = msg
            return state

        if not os.path.exists(data_path):
            msg = f"[Inspection] Data file not found: {data_path}"
            logger.error(msg)
            state["error"] = msg
            return state

        try:
            logger.info(f"[Inspection] Reading data from: {data_path}")
            adata = sc.read(data_path)

            user_intent = state.get("user_intent", "") or "Please identify the modality and batch column"
            info = self.inspect_adata(adata, user_intent=user_intent)
            modality_name = info.get("modality") or "unknown"

            obs_columns = info.get("obs_columns", [])
            batch_candidates = info.get("batch_candidates") or []
            user_intent = state.get("user_intent", "")

            if not batch_candidates:
                msg = (
                    f"Inspection found no batch-like columns for modality '{modality_name}'. "
                    f"obs_columns={obs_columns}"
                )
                logger.error(f"[Inspection] {msg}")
                raise ValueError(msg)

            batch_key_effective = info.get("batch_key_effective")
            if batch_key_effective is None and len(batch_candidates) == 1:
                batch_key_effective = batch_candidates[0]
                info["batch_key_effective"] = batch_key_effective
                logger.info(
                    f"[Inspection] Single batch candidate detected; using '{batch_key_effective}'."
                )

            if batch_key_effective is None and len(batch_candidates) > 1:
                try:
                    system_prompt = (
                        "You are a bioinformatics assistant helping to choose the batch key column "
                        "for single-cell integration.\n"
                        "You are given:\n"
                        "  - a free-text user request (may mention words like 'tech', "
                        "    'technology', 'experiment', 'sample', 'batch', etc.),\n"
                        "  - a list of all obs columns, and\n"
                        "  - a list of batch-like candidate columns (column names that contain 'batch').\n\n"
                        "Your job is to choose exactly ONE column name from the candidate list to use as the batch key.\n"
                        "If none of the candidates make sense, return null.\n\n"
                        "Return ONLY a JSON object of the form:\n"
                        "{\n"
                        '  \"batch_key\": \"<one_of_the_candidate_columns_or_null>\",\n'
                        '  \"reason\": \"<short explanation>\"\n'
                        "}\n"
                    )

                    user_message = (
                        f"User intent: {user_intent}\n"
                        f"All obs columns: {obs_columns}\n"
                        f"Batch-like candidates: {batch_candidates}\n\n"
                        "If the user mentions e.g. 'tech' or 'technology', map that to the most appropriate column "
                        "among the candidates. If unsure, choose the column that most clearly represents batch."
                    )

                    resp = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message},
                        ],
                        temperature=0.0,
                    )
                    content = resp.choices[0].message.content
                    cleaned = content.replace("```json", "").replace("```", "").strip()
                    parsed = json.loads(cleaned)
                    candidate = parsed.get("batch_key")

                    if isinstance(candidate, str) and candidate in obs_columns:
                        batch_key_effective = candidate
                        info["batch_key_effective"] = candidate
                        logger.info(
                            f"[Inspection] LLM chose batch_key_effective='{candidate}' "
                            f"from candidates={batch_candidates}"
                        )
                    else:
                        logger.error(
                            f"[Inspection] LLM did not return a valid batch_key from candidates. "
                            f"parsed={parsed}"
                        )

                except Exception as e:
                    logger.error(f"[Inspection] LLM-based batch_key selection failed: {e}")

            if info.get("batch_key_effective") is None:
                msg = (
                    f"Inspection did not determine batch_key_effective for modality '{modality_name}'. "
                    f"obs_columns={obs_columns}, candidates={batch_candidates}"
                )
                logger.error(f"[Inspection] {msg}")
                raise ValueError(msg)

            state["inspection"] = {modality_name: info}
            logger.info(f"[Inspection] Done. Summary: {info}")
            return state

        except Exception as e:
            logger.error(f"[Inspection] Error while inspecting data: {e}")
            state["error"] = f"Inspection failed: {e}"
            return state

# ==========================================
# 1. Planner Agent (负责规划)
# ==========================================
class PlannerAgent(BaseAgent):
    def __init__(self):
        super().__init__("Planner")

    def _validate_and_normalize_plan(self, raw_plan: dict, inspection: Dict[str, Any]) -> dict:
        valid_plan = {}

        if not isinstance(raw_plan, dict):
            raise ValueError("Planner returned non-dict plan.")

        # ??????methods ?? {method_name: params} ??????? list/str
        for modality, subplan in raw_plan.items():
            if not isinstance(subplan, dict):
                raise ValueError(f"Invalid subplan for modality {modality}")

            insp = inspection.get(modality)
            if insp is None and inspection:
                insp = next(iter(inspection.values()))
            if insp is None:
                raise ValueError(f"No inspection info available to determine batch_key for modality '{modality}'.")

            batch_key_effective = insp.get("batch_key_effective")
            if batch_key_effective is None:
                raise ValueError(f"Inspection did not determine batch_key_effective for modality '{modality}'.")

            preprocess = bool(subplan.get("preprocess", True))

            methods_field = subplan.get("methods", {})
            method_params_field = subplan.get("method_params", {})

            methods_dict: dict[str, dict] = {}

            if isinstance(methods_field, dict):
                for m, params in methods_field.items():
                    m_norm = canonicalize_method_name(m)
                    if m_norm:
                        methods_dict[m_norm] = params or {}
            elif isinstance(methods_field, list):
                for m in methods_field:
                    m_norm = canonicalize_method_name(m)
                    if m_norm:
                        methods_dict[m_norm] = {}
            elif isinstance(methods_field, str):
                m_norm = canonicalize_method_name(methods_field)
                if m_norm:
                    methods_dict[m_norm] = {}

            if isinstance(method_params_field, dict):
                for m, params in method_params_field.items():
                    m_norm = canonicalize_method_name(m)
                    if m_norm:
                        existing = methods_dict.get(m_norm, {})
                        merged = {**(existing or {}), **(params or {})}
                        methods_dict[m_norm] = merged

            if not methods_dict:
                raise ValueError(f"No valid methods specified for modality {modality}")

            valid_plan[modality] = {
                "preprocess": preprocess,
                "batch_key": batch_key_effective,
                "methods": methods_dict,
            }
        if not valid_plan:
            raise ValueError("Planner produced empty plan.")
        return valid_plan

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        user_intent = state.get("user_intent", "")
        inspection = state.get("inspection", {})
        logger.info(f"🤖 [Planner] Analyzing request: {user_intent}")

        parser = TaskIntentParser(user_intent)
        detected_modalities = parser.detect_modalities()
        subset_prefs = parser.extract_subset_preferences()
        state["task_modalities"] = detected_modalities
        state["subset_config"] = subset_prefs
        preprocess_recos: Dict[str, Any] = {}

        # prompt
        system_prompt = (
            "You are a bioinformatics pipeline planner.\n"
            "Output a concise JSON plan only. Example: {\"RNA\": {\"preprocess\": true, \"methods\": [\"scvi\",\"harmony\"]}}\n"
            "Include one entry per modality you see, with keys preprocess (true/false) and methods (list of allowed names).\n"
            "Do not add explanations."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_intent}
        ]
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.0
        )
        try:
            content = resp.choices[0].message.content if resp and resp.choices else ""
            cleaned = content.replace("```json", "").replace("```", "").strip()
            m = re.search(r"\{.*\}", cleaned, flags=re.S)
            raw_plan = json.loads(m.group(0) if m else cleaned) if cleaned else {}
        except Exception:
            raw_plan = {}

        ALL = ["scvi", "harmony", "mnn", "cca", "liger"]
        modalities_for_plan = list(inspection.keys()) or detected_modalities or ["RNA"]

        raw_plan = {}
        for modality in modalities_for_plan:
            reco = parser.infer_preprocessing(modality)
            preprocess_recos[modality] = reco
            methods = ALL.copy()
            preferred = reco.get("preferred_method")
            if preferred and preferred in ALL:
                methods = [preferred] + [m for m in methods if m != preferred]
            raw_plan[modality] = {
                "preprocess": bool(reco.get("preprocess", True)),
                "methods": methods,
                "modality": modality,
            }

        state["preprocess_recommendations"] = preprocess_recos
        plan = self._validate_and_normalize_plan(raw_plan, inspection)

        inspection_info = state.get("inspection") or {}
        for modality, subplan in plan.items():
            ins = inspection_info.get(modality) or next(iter(inspection_info.values()), {})
            effective_batch = ins.get("batch_key_effective")
            if not effective_batch:
                raise ValueError(
                    f"Planner: inspection did not provide batch_key_effective for modality '{modality}'. "
                    f"inspection={ins}"
                )
            subplan["batch_key"] = effective_batch

            # Assign a sanitized integration filename to avoid path issues
            try:
                raw_filename = f"integrated_{modality}_{effective_batch}.h5ad"
                subplan["sanitized_filename"] = sanitize_filename(raw_filename)
            except Exception:
                subplan["sanitized_filename"] = None

        state["plan"] = plan
        logger.info(f"📋 [Planner] Plan created: {plan}")
        return state



# ==========================================
# 1.5 Tuning Agent (hyperparameter presets)
# ==========================================
class TuningAgent(BaseAgent):
    def __init__(self):
        super().__init__("Tuning")

    @staticmethod
    def _get_size_bucket(n_cells: Optional[int]) -> str:
        if n_cells is None:
            return "small"
        if n_cells <= 50000:
            return "small"
        if n_cells <= 200000:
            return "medium"
        return "large"

    @staticmethod
    def _safe_copy(d: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        return dict(d) if isinstance(d, dict) else {}

    def _select_scvi_params(
        self,
        modality_label: str,
        n_cells: Optional[int],
        base_tier: str,
        inspection_info: Dict[str, Any],
    ) -> tuple[str, Dict[str, Any]]:
        presets = config.INTEGRATION_METHOD_PRESETS.get("scvi", {})
        tier_key = base_tier if base_tier in presets else "standard"
        tier_tree = presets.get(tier_key, {})

        modality_key = "ATAC" if str(modality_label).upper() == "ATAC" else "RNA"
        modality_tree = tier_tree.get(modality_key) or tier_tree.get("RNA") or {}
        size_bucket = self._get_size_bucket(n_cells)
        size_params = self._safe_copy(modality_tree.get(size_bucket))

        params = dict(size_params)
        params["is_count_data"] = bool(inspection_info.get("is_integer_like", True))
        if inspection_info.get("has_log1p") and not inspection_info.get("is_integer_like", True):
            params["is_count_data"] = False
        # Enable early stopping in a configurable way
        params["early_stopping"] = True
        params["early_stopping_patience"] = 20
        return tier_key, params

    def _select_harmony_params(
        self,
        n_cells: Optional[int],
        n_batches: int,
        base_tier: str,
    ) -> tuple[str, Dict[str, Any]]:
        presets = config.INTEGRATION_METHOD_PRESETS.get("harmony", {})
        tier_key = base_tier if base_tier in presets else "standard"
        if n_batches >= 6 and "high" in presets:
            tier_key = "high"
        tier_params = self._safe_copy(presets.get(tier_key))
        params = dict(tier_params)
        if params.get("nclust") and n_cells:
            if tier_key == "standard":
                params["nclust"] = min(max(params["nclust"], int(0.01 * n_cells)), 100)
            elif tier_key == "high":
                params["nclust"] = min(max(params["nclust"], int(0.02 * n_cells)), 150)
        return tier_key, params

    def _select_mnn_params(
        self,
        n_batches: int,
        base_tier: str,
    ) -> tuple[str, Dict[str, Any]]:
        presets = config.INTEGRATION_METHOD_PRESETS.get("mnn", {})
        tier_key = base_tier if base_tier in presets else "standard"
        if n_batches >= 6 and "high" in presets:
            tier_key = "high"
        params = self._safe_copy(presets.get(tier_key))
        return tier_key, params

    def _select_cca_params(self, base_tier: str) -> tuple[str, Dict[str, Any]]:
        presets = config.INTEGRATION_METHOD_PRESETS.get("cca", {})
        tier_key = base_tier if base_tier in presets else "standard"
        params = self._safe_copy(presets.get(tier_key))
        return tier_key, params

    def _select_liger_params(self, modality_label: str, base_tier: str) -> tuple[str, Dict[str, Any]]:
        presets = config.INTEGRATION_METHOD_PRESETS.get("liger", {})
        tier_key = base_tier if base_tier in presets else "standard"
        modality_key = "ATAC" if str(modality_label).upper() == "ATAC" else "RNA"
        tier_tree = presets.get(tier_key) or {}
        params = self._safe_copy(tier_tree.get(modality_key))
        return tier_key, params

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        inspection = state.get("inspection") or {}
        plan = state.get("plan") or {}
        run_all_methods_flag = bool(state.get("run_all_methods", False))

        if not inspection or not plan:
            logger.info("[Tuning] Missing inspection or plan; skipping tuning.")
            return state

        method_candidates: List[str] = []
        for modality, subplan in plan.items():
            methods_cfg = (subplan.get("methods") or {}).keys()
            for m in methods_cfg:
                canonical = canonicalize_method_name(m)
                if canonical:
                    method_candidates.append(canonical)
            if run_all_methods_flag and not subplan.get("methods"):
                method_candidates.extend(INTEGRATION_METHODS.keys())

        if run_all_methods_flag and not method_candidates:
            method_candidates = list(INTEGRATION_METHODS.keys())

        method_candidates = sorted(set(method_candidates))
        state["method_candidates"] = method_candidates

        chosen_params: Dict[str, List[str]] = {}
        param_presets: Dict[str, Dict[str, Dict[str, Any]]] = {}
        compute_budget = dict(getattr(config, "COMPUTE_BUDGET", {}) or {})
        debug_light = bool(state.get("quick_tuning"))
        default_tier = state.get("preset_tier_default") or ("light" if debug_light else "standard")

        for modality, subplan in plan.items():
            insp = inspection.get(modality) or next(iter(inspection.values()), {})
            n_cells = insp.get("n_cells")
            n_batches = len(insp.get("batch_candidates") or []) or 1
            modality_label = insp.get("modality") or modality

            for method in (subplan.get("methods") or {}):
                if method not in method_candidates or method in chosen_params:
                    continue

                method_presets = config.INTEGRATION_METHOD_PRESETS.get(method, {}) or {}
                tiers = [t for t in ["light", "standard", "high"] if t in method_presets]
                tiers = (tiers or [default_tier])[:3]

                preset_dict: Dict[str, Dict[str, Any]] = {}
                for t in tiers:
                    params: Dict[str, Any] = {}
                    if method == "scvi":
                        _, params = self._select_scvi_params(modality_label, n_cells, t, insp)
                    elif method == "harmony":
                        _, params = self._select_harmony_params(n_cells, n_batches, t)
                    elif method == "mnn":
                        _, params = self._select_mnn_params(n_batches, t)
                    elif method == "cca":
                        _, params = self._select_cca_params(t)
                    elif method == "liger":
                        _, params = self._select_liger_params(modality_label, t)
                    preset_dict[t] = params

                chosen_params[method] = tiers
                param_presets[method] = preset_dict

        # TODO: multi-preset benchmarking and populating param_results with metrics
        state["chosen_params"] = chosen_params
        state["param_presets"] = param_presets
        state["param_results"] = {}
        state["compute_budget"] = compute_budget

        if state.get("search_params", True):
            return state

        # Merge presets into plan
        for modality, subplan in plan.items():
            methods_cfg = subplan.get("methods") or {}
            merged_methods: Dict[str, Dict[str, Any]] = {}
            for method, base_params in methods_cfg.items():
                preset_name = chosen_params.get(method)
                if isinstance(preset_name, list):
                    preset_key = preset_name[0] if preset_name else None
                else:
                    preset_key = preset_name

                if preset_key:
                    preset_values = param_presets.get(method, {}).get(preset_key, {}) or {}
                    merged_methods[method] = {**(base_params or {}), **preset_values}
                else:
                    merged_methods[method] = base_params or {}
            subplan["methods"] = merged_methods
            plan[modality] = subplan

        state["plan"] = plan
        logger.info(f"[Tuning] Applied presets. chosen={chosen_params}")
        return state
# ==========================================
# 2. Preprocess Agent (负责数据准备)
# ==========================================
class PreprocessAgent(BaseAgent):
    def __init__(self):
        super().__init__("Preprocessor")

    def _apply_subsets(self, adata_hvg: Any, adata_raw: Any, subset_cfg: Dict[str, Any], modality: str):
        if not subset_cfg:
            return adata_hvg, adata_raw

        result_hvg = adata_hvg
        result_raw = adata_raw

        genes = subset_cfg.get("genes") or []
        if genes:
            upper_map = {g.upper(): g for g in result_hvg.var_names}
            keep_genes = [upper_map[g] for g in genes if g in upper_map]
            if keep_genes:
                result_hvg = result_hvg[:, keep_genes].copy()
                if result_raw is not None:
                    result_raw = result_raw[:, keep_genes].copy()
                logger.info(f"[Preprocessor] {modality}: subset to {len(keep_genes)} genes from intent")

        celltypes = subset_cfg.get("celltypes") or []
        if celltypes:
            candidate_cols = [c for c in result_hvg.obs.columns if "cell" in c.lower() or "type" in c.lower()]
            matched_col = candidate_cols[0] if candidate_cols else None
            if matched_col:
                mask = result_hvg.obs[matched_col].astype(str).isin(celltypes)
                if mask.sum() > 0:
                    result_hvg = result_hvg[mask].copy()
                    if result_raw is not None:
                        result_raw = result_raw[result_hvg.obs_names].copy()
                    logger.info(
                        f"[Preprocessor] {modality}: subset to {mask.sum()} cells matching {celltypes} via {matched_col}"
                    )

        batches = subset_cfg.get("batches") or []
        if batches and "batch" in result_hvg.obs:
            mask = result_hvg.obs["batch"].astype(str).isin(batches)
            if mask.sum() > 0:
                result_hvg = result_hvg[mask].copy()
                if result_raw is not None:
                    result_raw = result_raw[result_hvg.obs_names].copy()
                logger.info(f"[Preprocessor] {modality}: subset to batches {batches}")

        return result_hvg, result_raw

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        plan = state.get("plan", {})
        data_path = state.get("data_path")
        benchmark_fraction = state.get("benchmark_fraction")
        subset_cfg = state.get("subset_config") or {}

        # Ensure containers exist and are dicts
        if not isinstance(state.get("data_hvg"), dict):
            state["data_hvg"] = {}
        if not isinstance(state.get("data_raw"), dict):
            state["data_raw"] = {}
        if not isinstance(state.get("data_hvg_full"), dict):
            state["data_hvg_full"] = {}
        if not isinstance(state.get("data_raw_full"), dict):
            state["data_raw_full"] = {}

        for modality, subplan in plan.items():
            batch_key = subplan.get("batch_key", "batch")
            preprocess_flag = subplan.get("preprocess", True)
            adata_existing = state["data_hvg"].get(modality)
            modality_param = subplan.get("modality", modality)
            reco = (state.get("preprocess_recommendations") or {}).get(modality, {})

            # Store a sanitized preprocessing output filename to avoid path issues
            try:
                raw_preproc_name = f"{modality}_preprocessed_{batch_key}.h5ad"
                subplan["sanitized_preprocessed_filename"] = sanitize_filename(raw_preproc_name)
            except Exception:
                subplan["sanitized_preprocessed_filename"] = None

            if not preprocess_flag and adata_existing is not None:
                state["data_hvg_full"].setdefault(modality, adata_existing)
                state["data_raw_full"].setdefault(modality, state["data_raw"].get(modality))
                logger.info(f"✅ [Preprocessor] Skipped {modality}")
                continue

            logger.info(f"🧪 [Preprocessor] Processing {modality} data...")
            if reco.get("preferred_method"):
                logger.info(
                    f"[Preprocessor] Intent suggested preprocessing helper: {reco.get('preferred_method')} (need_preprocess={preprocess_flag})"
                )
            try:
                adata_hvg, adata_raw = OmicsTools.load_and_preprocess(
                    data_path,
                    original_batch_key=batch_key,
                    modality=modality_param
                )
                adata_hvg, adata_raw = self._apply_subsets(adata_hvg, adata_raw, subset_cfg.get(modality, subset_cfg), modality)
                state["data_hvg_full"][modality] = adata_hvg.copy()
                state["data_raw_full"][modality] = adata_raw.copy() if adata_raw is not None else None
                # Subsample for benchmark mode
                if benchmark_fraction is not None:
                    try:
                        frac = float(benchmark_fraction)
                    except Exception:
                        raise ValueError(f"Invalid benchmark_fraction={benchmark_fraction}")
                    if not (0.0 < frac < 1.0):
                        raise ValueError(f"benchmark_fraction must be in (0, 1), got {frac}")
                    if adata_hvg.n_obs <= 10:
                        raise ValueError(f"[Preprocessor] Too few cells ({adata_hvg.n_obs}) for benchmark subsampling.")
                    else:
                        n_cells = adata_hvg.n_obs
                        n_keep = max(int(n_cells * frac), 10)
                        idx = np.random.choice(n_cells, size=n_keep, replace=False)
                        adata_hvg = adata_hvg[idx].copy()
                        if adata_raw is not None:
                            adata_raw = adata_raw[adata_hvg.obs_names].copy()
                        logger.info(f"[Preprocessor] Subsampled {modality}: {n_cells} -> {n_keep} cells for benchmark.")
                # HVG requirement
                if adata_hvg.shape[1] < 200:
                    raise ValueError(f"[Preprocessor] {modality} has too few HVGs ({adata_hvg.shape[1]}).")

                state["data_hvg"][modality] = adata_hvg
                state["data_raw"][modality] = adata_raw
                logger.info(f"✅ [Preprocessor] {modality} done. Cells: {adata_hvg.n_obs}, Genes: {adata_hvg.n_vars}")

            except Exception as e:
                logger.error(f"🛑 [Preprocessor] Error processing {modality}: {e}")
                state["error"] = str(e)
                raise

        return state

# ==========================================
# 3. Integration Agent (负责调用整合算法)
# ==========================================
class IntegrationAgent(BaseAgent):
    def __init__(self):
        super().__init__("Integrator")

    @staticmethod
    def _extract_methods_config(methods_cfg: Any) -> Dict[str, Dict[str, Any]]:
        normalized: Dict[str, Dict[str, Any]] = {}

        if isinstance(methods_cfg, dict):
            iterable = methods_cfg.items()
        elif isinstance(methods_cfg, list):
            iterable = [(m, {}) for m in methods_cfg]
        elif isinstance(methods_cfg, str):
            iterable = [(methods_cfg, {})]
        else:
            iterable = []

        for method_name, params in iterable:
            canonical = canonicalize_method_name(method_name)
            if not canonical:
                continue
            merged = {**(normalized.get(canonical, {}) or {}), **(params or {})}
            normalized[canonical] = merged

        return normalized

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        plan = state.get("plan", {})
        run_all_methods_flag = bool(state.get("run_all_methods", False))
        top1_only = bool(state.get("top1_only", True))

        state.setdefault("results", {})
        state.setdefault("param_search_results", [])
        state.setdefault("best_params", {})
        state.setdefault("best_param_tiers", {})
        state.setdefault("best_rank", {})
        state.setdefault("method_errors", {})
        state.setdefault("final_selection", {})

        adata_hvg_all = state.get("data_hvg") or {}
        adata_raw_all = state.get("data_raw") or {}
        adata_hvg_full_all = state.get("data_hvg_full") or {}
        adata_raw_full_all = state.get("data_raw_full") or {}

        if not adata_hvg_all:
            raise RuntimeError("No input data found for integration.")

        for modality, subplan in plan.items():
            batch_key = "batch"  # downstream integration统一使用标准化后的 batch 列
            adata_hvg_bm = adata_hvg_all.get(modality)
            adata_raw_bm = adata_raw_all.get(modality)
            adata_hvg_full = adata_hvg_full_all.get(modality, adata_hvg_bm)
            adata_raw_full = adata_raw_full_all.get(modality, adata_raw_bm)

            if adata_hvg_bm is None:
                raise RuntimeError(f"No input data found for modality '{modality}'")

            state["results"].setdefault(modality, {})
            state["method_errors"].setdefault(modality, {})
            state["best_params"][modality] = {}
            state["best_param_tiers"][modality] = {}
            state["best_rank"][modality] = []

            if subplan.get("methods"):
                methods_cfg = self._extract_methods_config(subplan["methods"])
            elif run_all_methods_flag:
                methods_cfg = {name: {} for name in INTEGRATION_METHODS.keys()}
                logger.info(f"[Integrator] run_all_methods=True, using all methods for {modality}: {list(methods_cfg.keys())}")
            else:
                methods_cfg = self._extract_methods_config(subplan.get("methods", {}))

            if not methods_cfg:
                methods_cfg = {name: {} for name in INTEGRATION_METHODS.keys()}

            for method_name, method_params in methods_cfg.items():
                runner = INTEGRATION_METHODS.get(method_name)
                if runner is None:
                    state["method_errors"][modality][method_name] = "Unknown integration method"
                    continue

                presets = state.get("param_presets", {}).get(method_name, {}) or {}
                candidates = list(presets.items())[:3]
                if not candidates:
                    candidates = [("single", method_params or {})]

                best_score = None
                best_params_for_method: Dict[str, Any] | None = None
                best_tier_name: Optional[str] = None

                for tier, params in candidates:
                    record = {
                        "modality": modality,
                        "method": method_name,
                        "tier": tier,
                        "gc": None,
                        "basw": None,
                        "score": None,
                        "error": None,
                    }
                    try:
                        result_adata = runner(
                            adata_hvg_bm,
                            adata_raw_bm,
                            batch_key=batch_key,
                            **(params or {})
                        )
                    except Exception as e:
                        record["error"] = str(e)
                        state["param_search_results"].append(record)
                        continue

                    if result_adata is None:
                        record["error"] = "Returned None"
                        state["param_search_results"].append(record)
                        continue

                    embedding_key = f"X_{method_name}"
                    if embedding_key not in result_adata.obsm:
                        record["error"] = f"Missing embedding {embedding_key}"
                        state["param_search_results"].append(record)
                        continue

                    emb = result_adata.obsm[embedding_key]
                    gc_value = None
                    try:
                        gc_value = calculate_graph_connectivity(emb, n_neighbors=15, n_cells=emb.shape[0])
                    except Exception:
                        gc_value = None

                    basw = 0.0
                    try:
                        if "batch" in result_adata.obs and result_adata.obs["batch"].nunique() >= 2:
                            basw = float(silhouette_score(emb, result_adata.obs["batch"]))
                    except Exception:
                        basw = 0.0

                    score = (gc_value if gc_value is not None else 0.0) - basw

                    record.update({"gc": gc_value, "basw": basw, "score": score})
                    state["param_search_results"].append(record)

                    if (best_score is None) or (score > best_score):
                        best_score = score
                        best_params_for_method = params
                        best_tier_name = tier

                if best_params_for_method is not None:
                    state["best_params"][modality][method_name] = best_params_for_method
                    state["best_param_tiers"][modality][method_name] = best_tier_name
                    state["best_rank"][modality].append((method_name, best_score if best_score is not None else 0.0))
                else:
                    state["method_errors"][modality][method_name] = "No valid params in benchmark"

            ranked = sorted(state["best_rank"].get(modality, []), key=lambda x: x[1], reverse=True)
            if not ranked:
                continue

            top_methods = ranked[:1] if top1_only else ranked
            for top_method, top_score in top_methods:
                runner = INTEGRATION_METHODS.get(top_method)
                if runner is None:
                    state["method_errors"][modality][top_method] = "Unknown integration method"
                    continue
                params = state["best_params"][modality].get(top_method, {})
                try:
                    result_adata = runner(
                        adata_hvg_full,
                        adata_raw_full,
                        batch_key=batch_key,
                        **(params or {})
                    )
                except Exception as e:
                    state["method_errors"][modality][top_method] = str(e)
                    continue

                if result_adata is None:
                    state["method_errors"][modality][top_method] = "Returned None on full data"
                    continue

                embedding_key = f"X_{top_method}"
                if embedding_key not in result_adata.obsm:
                    state["method_errors"][modality][top_method] = f"Missing embedding {embedding_key} on full data"
                    continue

                state["results"][modality][top_method] = result_adata
                state["final_selection"][modality] = {"method": top_method, "params": params, "score": top_score}

                if top1_only:
                    break

        return state

# ==========================================
# 4. Evaluation Agent
# ==========================================
class EvaluationAgent(BaseAgent):
    def __init__(self):
        super().__init__("Evaluator")

    def _get_celltype_key(
        self,
        modality: str,
        adata_integrated: sc.AnnData,
        inspection: Dict[str, Any],
    ) -> Optional[str]:
        if "celltype" in adata_integrated.obs:
            return "celltype"
        if "final_cell_label" in adata_integrated.obs:
            return "final_cell_label"
        insp = inspection.get(modality) or {}
        cols = insp.get("celltype_columns") or []
        return cols[0] if cols else None

    def _get_cluster_key(self, method_name: str, adata_integrated: sc.AnnData) -> Optional[str]:
        if method_name == "mnn" and "leiden_mnn" in adata_integrated.obs:
            return "leiden_mnn"
        if "leiden" in adata_integrated.obs:
            return "leiden"
        if method_name == "cca" and "cca_cluster" in adata_integrated.obs:
            return "cca_cluster"
        if method_name == "liger" and "liger_cluster" in adata_integrated.obs:
            return "liger_cluster"
        return None

    def _task_minmax(self, metric_id: str, values_by_method: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
        standardized: Dict[str, Optional[float]] = {}
        finite_vals = [v for v in values_by_method.values() if v is not None]
        if not finite_vals:
            return {m: None for m in values_by_method}
        min_v = min(finite_vals)
        max_v = max(finite_vals)
        direction = (METRIC_SPECS.get(metric_id, {}) or {}).get("direction", "higher_better")

        for method, raw_value in values_by_method.items():
            if raw_value is None:
                standardized[method] = None
                continue
            if max_v == min_v:
                standardized[method] = 0.5
                continue
            if direction == "higher_better":
                norm = (raw_value - min_v) / (max_v - min_v)
            elif direction == "lower_better":
                norm = (max_v - raw_value) / (max_v - min_v)
            else:  # zero_best
                norm = 1.0 - (abs(raw_value) / max(abs(min_v), abs(max_v)))
            standardized[method] = float(np.clip(norm, 0.0, 1.0))
        return standardized

    def _aggregate_group(
        self,
        metric_ids: List[str],
        normalized_scores: Dict[str, Optional[float]],
    ) -> tuple[float, float]:
        if not metric_ids:
            return 0.0, 0.0
        total = 0.0
        weight_sum = 0.0
        available = 0
        for mid in metric_ids:
            val = normalized_scores.get(mid)
            if val is None:
                continue
            weight = float(METRIC_SPECS.get(mid, {}).get("weight", 1.0))
            total += weight * val
            weight_sum += weight
            available += 1
        coverage = available / len(metric_ids)
        if available == 0 or weight_sum == 0.0:
            return 0.0, coverage
        base = total / weight_sum
        return float(base * np.sqrt(coverage)), coverage

    def _derive_group_weights(
        self, group_to_metrics: Dict[str, List[str]], overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        weights: Dict[str, float] = {}
        if overrides:
            for key, val in overrides.items():
                try:
                    w = float(val)
                except (TypeError, ValueError):
                    continue
                if w <= 0:
                    continue
                weights[key.lower()] = w
        if not weights:
            for group_key, metric_ids in group_to_metrics.items():
                weights[group_key] = sum(
                    float(METRIC_SPECS.get(mid, {}).get("weight", 1.0)) for mid in metric_ids
                )
        positive_total = sum(w for w in weights.values() if w > 0)
        if positive_total <= 0:
            if not group_to_metrics:
                return {}
            uniform = 1.0 / len(group_to_metrics)
            return {g: uniform for g in group_to_metrics.keys()}
        return {g: w / positive_total for g, w in weights.items() if w > 0}

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        results = state.get("results") or {}
        inspection = state.get("inspection") or {}
        adata_full = state.get("data_hvg_full") or {}

        raw_scores: Dict[str, Dict[str, Dict[str, Optional[float]]]] = defaultdict(lambda: defaultdict(dict))
        normalized_scores: Dict[str, Dict[str, Dict[str, Optional[float]]]] = defaultdict(lambda: defaultdict(dict))
        group_scores: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        final_scores: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        # Build group mapping from METRIC_SPECS
        group_to_metrics: Dict[str, List[str]] = defaultdict(list)
        for mid, spec in METRIC_SPECS.items():
            group = (spec.get("group") or "").lower()
            if group:
                group_to_metrics[group].append(mid)

        group_weight_overrides = getattr(config, "EVALUATION_GROUP_WEIGHTS", None)
        group_weights = self._derive_group_weights(group_to_metrics, group_weight_overrides)

        # Stage 1: collect raw scores
        for modality, methods in results.items():
            for method_name, adata in (methods or {}).items():
                embedding_key = f"X_{method_name}"
                if embedding_key not in adata.obsm:
                    logger.warning(f"[Evaluation] Missing embedding {embedding_key} for {modality}-{method_name}, skip.")
                    continue

                celltype_key = self._get_celltype_key(modality, adata, inspection)
                cluster_key = self._get_cluster_key(method_name, adata)
                for metric_id in METRIC_SPECS.keys():
                    raw_value = compute_metric_raw(
                        metric_id=metric_id,
                        adata_integrated=adata,
                        adata_unintegrated=adata_full.get(modality),
                        embedding_key=embedding_key,
                        modality=modality,
                        batch_key="batch",
                        celltype_key=celltype_key,
                        cluster_key=cluster_key,
                    )
                    raw_scores[modality][method_name][metric_id] = raw_value

        # Stage 2: normalize per metric within each modality
        for modality, method_metrics in raw_scores.items():
            for metric_id in METRIC_SPECS.keys():
                values_by_method = {m: metrics.get(metric_id) for m, metrics in method_metrics.items()}
                standardized = self._task_minmax(metric_id, values_by_method)
                for m, val in standardized.items():
                    normalized_scores[modality][m][metric_id] = val

        # Stage 3: group aggregation with coverage penalty
        for modality, methods in normalized_scores.items():
            for method_name, metrics in methods.items():
                group_scores[modality][method_name] = {}
                for group_key, metric_list in group_to_metrics.items():
                    agg, _coverage = self._aggregate_group(metric_list, metrics)
                    group_scores[modality][method_name][group_key] = {
                        "score": agg,
                        "coverage": _coverage,
                    }

        # Stage 4: final score per modality
        for modality, methods in group_scores.items():
            for method_name, groups in methods.items():
                contributions: Dict[str, Any] = {}
                weighted_sum = 0.0
                effective_weight_sum = 0.0
                for group_key, group_info in groups.items():
                    base_weight = group_weights.get(group_key, 0.0)
                    group_score = float(group_info.get("score", 0.0))
                    coverage = float(group_info.get("coverage", 0.0))
                    effective_weight = base_weight * coverage
                    contribution = group_score * effective_weight
                    weighted_sum += contribution
                    effective_weight_sum += effective_weight
                    contributions[group_key] = {
                        "score": group_score,
                        "coverage": coverage,
                        "weight": base_weight,
                        "effective_weight": effective_weight,
                        "contribution": contribution,
                    }
                score = weighted_sum / effective_weight_sum if effective_weight_sum > 0 else 0.0
                final_scores[modality].append(
                    {
                        "method": method_name,
                        "score": score,
                        "groups": contributions,
                    }
                )
            final_scores[modality] = sorted(final_scores[modality], key=lambda x: x["score"], reverse=True)

        state["evaluation"] = {
            "raw": raw_scores,
            "normalized": normalized_scores,
            "group_scores": group_scores,
            "final_scores": final_scores,
            "group_weights": group_weights,
        }
        return state

# ==========================================
# 5. Reporter Agent
# ==========================================
class ReporterAgent(BaseAgent):
    def __init__(self):
        super().__init__("Reporter")

    @staticmethod
    def _print_plan(plan: Dict[str, Any]):
        print("\n--- Plan Overview ---")
        for modality, subplan in (plan or {}).items():
            methods_list = list(subplan.get("methods", {}).keys())
            methods_fmt = []
            for m in methods_list:
                if m == "cca":
                    methods_fmt.append("cca (Seurat-CCA integration via Seurat package)")
                else:
                    methods_fmt.append(m)
            print(f"[{modality}] preprocess={subplan.get('preprocess', True)}, batch_key={subplan.get('batch_key')}, methods={methods_fmt}")

    @staticmethod
    def _print_task_profile(state: Dict[str, Any]):
        modalities = state.get("task_modalities") or list((state.get("inspection") or {}).keys())
        subset_cfg = state.get("subset_config") or {}
        print("\n--- Task Understanding ---")
        print(f"Detected modalities: {modalities}")
        if subset_cfg:
            print(f"Subset preferences: {subset_cfg}")

    @staticmethod
    def _print_results(results: Dict[str, Any]):
        print("\n--- Results Overview ---")
        if not results:
            print("No integration results available.")
            return
        for modality, methods in results.items():
            if not methods:
                print(f"[{modality}] no results.")
                continue
            for method, adata in methods.items():
                emb_keys = list(adata.obsm.keys())
                print(f"[{modality}] {method}: cells={adata.n_obs}, features={adata.n_vars}, embeddings={emb_keys}")

    @staticmethod
    def _print_final_selection(final_selection: Dict[str, Any]):
        if not final_selection:
            return
        print("\n--- Final Selection ---")
        for modality, info in final_selection.items():
            method = info.get("method")
            score = info.get("score")
            print(f"[{modality}] method={method}, score={score}, params={info.get('params')}")

    @staticmethod
    def _print_method_comparison(evaluation: Optional[Dict[str, Any]]):
        if not isinstance(evaluation, dict):
            return
        final_scores = evaluation.get("final_scores") or {}
        if not final_scores:
            return
        print("\n--- Method Comparison ---")
        for modality, entries in final_scores.items():
            if not entries:
                continue
            best = entries[0]
            print(
                f"[{modality}] Top method: {best.get('method')} (score={best.get('score'):.4f}). "
                f"Other tested: {[e.get('method') for e in entries]}"
            )

    @staticmethod
    def _print_param_choices(state: Dict[str, Any]):
        tiers = state.get("best_param_tiers") or {}
        if not tiers:
            return
        print("\n--- Best Parameter Tiers ---")
        for modality, method_map in tiers.items():
            for method, tier in (method_map or {}).items():
                print(f"[{modality}] {method}: {tier}")

    @staticmethod
    def _print_evaluation(eval_obj: Optional[Any]):
        print("\n--- Evaluation ---")
        if eval_obj is None:
            print("No evaluation metrics were computed.")
            return
        if isinstance(eval_obj, pd.DataFrame):
            if eval_obj.empty:
                print("No evaluation metrics were computed.")
                return
            for metric, subdf in eval_obj.groupby("metric"):
                best_val = subdf["value"].max()
                print(f"\nMetric: {metric}")
                for _, row in subdf.iterrows():
                    star = " *" if row["value"] == best_val else ""
                    print(f"  {row['modality']} - {row['method']}: {row['value']:.4f}{star}")
            return
        if isinstance(eval_obj, dict):
            final_scores = eval_obj.get("final_scores") or {}
            if not final_scores:
                print("Evaluation structure present but empty.")
                return
            for modality, entries in final_scores.items():
                print(f"\n[{modality}] Final Scores:")
                for rec in entries:
                    print(f"  {rec.get('method')}: {rec.get('score'):.4f}")
            return
        print("Unknown evaluation format.")

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print("\n===== OMICS INTEGRATION REPORT =====")
        if state.get("error"):
            print(f"ERROR: {state.get('error')}")
        self._print_task_profile(state)
        self._print_plan(state.get("plan") or {})
        self._print_results(state.get("results") or {})
        self._print_final_selection(state.get("final_selection") or {})
        self._print_evaluation(state.get("evaluation"))
        self._print_method_comparison(state.get("evaluation"))
        self._print_param_choices(state)

        chosen_params = state.get("chosen_params") or {}
        param_presets = state.get("param_presets") or {}
        plan = state.get("plan") or {}
        report_entries: Dict[str, Any] = {}
        for modality in plan.keys():
            report_entries[modality] = {
                "dataset_overview": {},
                "benchmark_summary": {},
                "decision": {},
                "full_integration": {},
                "summary_text": "",
                "hyperparams": {
                    "chosen_params": chosen_params,
                    "param_presets": param_presets,
                },
            }
        if report_entries:
            state["report"] = report_entries

        if chosen_params:
            summary_line = ", ".join(f"{m}->{p}" for m, p in chosen_params.items())
            print(f"Hyperparameter presets used: {summary_line}")
        else:
            print("Hyperparameter presets used: default settings")
        return state
