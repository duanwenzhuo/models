# agents.py
import json
import logging
import os
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
from tools import OmicsTools, INTEGRATION_METHODS, calculate_graph_connectivity
from openai import OpenAI
client = OpenAI(
    api_key="sk-L1TUuj5pxxyBbg1aNiM2t5fQGdbhAOmJtgVufoXiek3KZLnJ",  
    base_url="https://api.lmtgpt.top/v1"  
)
# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    并确定最终使用的 batch_key_effective。
    """

    def __init__(self):
        super().__init__("Inspection")

    @staticmethod
    def inspect_adata(adata: AnnData, modality: str = "RNA") -> dict:
        info: dict[str, Any] = {}
        info["n_cells"] = adata.n_obs
        info["n_features"] = adata.n_vars

        obs_cols = adata.obs.columns.tolist()
        info["obs_columns"] = obs_cols
        info["var_columns"] = adata.var.columns.tolist()

        batch_candidates = [
            c for c in obs_cols
            if isinstance(c, str) and "batch" in c.lower()
        ]
        info["batch_candidates"] = batch_candidates
        info["has_batch"] = len(batch_candidates) > 0
        info["batch_key_effective"] = None

        celltype_cols = [
            c for c in obs_cols
            if isinstance(c, str) and ("celltype" in c.lower() or "cell_type" in c.lower())
        ]
        info["celltype_available"] = len(celltype_cols) > 0
        info["celltype_columns"] = celltype_cols

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

        info["modality"] = modality
        info["modality_guess"] = "rna" if is_integer_like else "unknown"

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

            modality_name = "RNA"
            info = self.inspect_adata(adata, modality=modality_name)

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
        # self.llm = ChatOpenAI(
        #     model=config.LLM_MODEL,
        #     api_key=config.OPENAI_API_KEY,
        #     base_url=config.OPENAI_BASE_URL,
        #     temperature=0
        # )
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

    # def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
    #     user_intent = state.get("user_intent", "")
    #     logger.info(f"🤖 [Planner] Analyzing request: {user_intent}")
    #     inspection = state.get("inspection", {})
    #     # Prompt: 要求 LLM 返回特定格式的 JSON
    #     system_prompt = (
    #         "You are a bioinformatics pipeline planner. "
    #         "Based on the user request, decide if preprocessing is needed "
    #         "and list which integration methods to run (choose from: scvi, harmony, mnn, seurat, liger, cca). "
    #         "Identify the batch key if mentioned, otherwise default to 'batch'.\n\n"
    #         "Output ONLY a JSON object in this structure:\n"
    #         "{{\n"
    #         "  \"RNA\": {{\"preprocess\": true, \"methods\": [\"scvi\", \"harmony\"], \"batch_key\": \"tech\"}},\n"
    #         "  \"ATAC\": {{\"preprocess\": true, \"methods\": [\"mnn\"], \"batch_key\": null}}\n"
    #         "}}"
    #     )



    #     prompt = ChatPromptTemplate.from_messages([
    #         ("system", system_prompt),
    #         ("human", "{input}")
    #     ])

    #     chain = prompt | self.llm | JsonOutputParser()
    #     default_plan = {
    #         "RNA": {
    #             "preprocess": True,
    #             "methods": ["scvi", "harmony"],
    #             "batch_key": "batch",
    #             "method_params": {}
    #         }
    #     }

    #     try:
    #         response = chain.invoke({"input": user_intent})
    #         # 清理可能存在的 Markdown 标记
    #         cleaned_json = response.replace("```json", "").replace("```", "").strip()
    #         raw_plan = json.loads(cleaned_json)
            
    #         # 使用内部 Checker 进行验证和规范化
    #         plan = self._validate_and_normalize_plan(raw_plan)

    #     except Exception as e:
    #         logger.error(f"⚠️ Plan parsing or validation failed: {e}. Using default plan.")
    #         plan = default_plan

    #     state["plan"] = plan
    #     logger.info(f"📋 [Planner] Plan created: {plan}")
    #     return state
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        user_intent = state.get("user_intent", "")
        inspection = state.get("inspection", {})
        logger.info(f"🤖 [Planner] Analyzing request: {user_intent}")

        # prompt
        system_prompt = (
            "You are a bioinformatics pipeline planner.\n"
            "Allowed integration methods (canonical names) are: scvi, harmony, mnn, cca, liger.\n"
            "The name 'cca' means Seurat-CCA integration via the Seurat package.\n"
            "\n"
            "You receive inspection info for each modality, which already decided the effective batch key.\n"
            "DO NOT invent or choose batch columns yourself. You only decide:\n"
            "  - whether preprocessing is needed (preprocess: true/false)\n"
            "  - which integration methods to run (subset of: scvi, harmony, mnn, cca, liger)\n"
            "\n"
            "For each modality in the inspection, output a JSON object with only 'preprocess' and 'methods'.\n"
            "Do NOT include 'batch_key' in the JSON.\n"
            f"Data inspection info: {inspection}\n\n"
            "Output ONLY a JSON object like:\n"
            '{ \"RNA\": {\"preprocess\": true, \"methods\": [\"scvi\", \"harmony\", \"cca\"]} }'
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
        content = resp.choices[0].message.content
        cleaned_json = content.replace("```json", "").replace("```", "").strip()
        raw_plan = json.loads(cleaned_json)
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

        chosen_params: Dict[str, str] = {}
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

                tier_key = default_tier
                params: Dict[str, Any] = {}

                if method == "scvi":
                    tier_key, params = self._select_scvi_params(modality_label, n_cells, default_tier, insp)
                elif method == "harmony":
                    tier_key, params = self._select_harmony_params(n_cells, n_batches, default_tier)
                elif method == "mnn":
                    tier_key, params = self._select_mnn_params(n_batches, default_tier)
                elif method == "cca":
                    tier_key, params = self._select_cca_params(default_tier)
                elif method == "liger":
                    tier_key, params = self._select_liger_params(modality_label, default_tier)

                chosen_params[method] = tier_key
                param_presets[method] = {tier_key: params}

        # TODO: multi-preset benchmarking and populating param_results with metrics
        state["chosen_params"] = chosen_params
        state["param_presets"] = param_presets
        state["param_results"] = {}
        state["compute_budget"] = compute_budget

        # Merge presets into plan
        for modality, subplan in plan.items():
            methods_cfg = subplan.get("methods") or {}
            merged_methods: Dict[str, Dict[str, Any]] = {}
            for method, base_params in methods_cfg.items():
                preset_name = chosen_params.get(method)
                if preset_name:
                    preset_values = param_presets.get(method, {}).get(preset_name, {}) or {}
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

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        plan = state.get("plan", {})
        data_path = state.get("data_path")
        benchmark_fraction = state.get("benchmark_fraction")

        # Ensure containers exist and are dicts
        if not isinstance(state.get("data_hvg"), dict):
            state["data_hvg"] = {}
        if not isinstance(state.get("data_raw"), dict):
            state["data_raw"] = {}

        for modality, subplan in plan.items():
            batch_key = subplan.get("batch_key", "batch")
            preprocess_flag = subplan.get("preprocess", True)
            adata_existing = state["data_hvg"].get(modality)
            modality_param = subplan.get("modality", modality)

            if not preprocess_flag and adata_existing is not None:
                logger.info(f"✅ [Preprocessor] Skipped {modality}")
                continue

            logger.info(f"🧪 [Preprocessor] Processing {modality} data...")
            try:
                adata_hvg, adata_raw = OmicsTools.load_and_preprocess(
                    data_path,
                    original_batch_key=batch_key,
                    modality=modality_param
                )
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

        if "results" not in state:
            state["results"] = {}

        adata_hvg_all = state.get("data_hvg") or {}
        adata_raw_all = state.get("data_raw") or {}

        if not adata_hvg_all:
            raise RuntimeError("No input data found for integration.")

        for modality, subplan in plan.items():
            batch_key = "batch"  # downstream integration统一使用标准化后的 batch 列
            adata_hvg = adata_hvg_all.get(modality)
            adata_raw = adata_raw_all.get(modality)

            if adata_hvg is None:
                raise RuntimeError(f"No input data found for modality '{modality}'")

            if subplan.get("methods"):
                methods_cfg = self._extract_methods_config(subplan["methods"])
            elif run_all_methods_flag:
                methods_cfg = {name: {} for name in INTEGRATION_METHODS.keys()}
                logger.info(f"[Integrator] run_all_methods=True, using all methods for {modality}: {list(methods_cfg.keys())}")
            else:
                methods_cfg = self._extract_methods_config(subplan.get("methods", {}))

            if not methods_cfg:
                raise RuntimeError(f"No integration methods configured for modality '{modality}'")

            state["results"].setdefault(modality, {})

            for method_name, method_params in methods_cfg.items():
                runner = INTEGRATION_METHODS.get(method_name)
                if runner is None:
                    raise KeyError(f"Unknown integration method '{method_name}' for {modality}")

                try:
                    result_adata = runner(
                        adata_hvg,
                        adata_raw,
                        batch_key=batch_key,
                        **(method_params or {})
                    )

                    if result_adata is None:
                        raise RuntimeError(f"Integration method '{method_name}' returned None for modality '{modality}'")

                    embedding_key = f"X_{method_name}"
                    if embedding_key not in result_adata.obsm:
                        raise KeyError(f"Method '{method_name}' result missing expected embedding '{embedding_key}'")

                    state["results"][modality][method_name] = result_adata
                    logger.info(f"✅ [Integrator] {method_name} finished for {modality}")

                except Exception as e:
                    logger.error(f"🛑 [Integrator] Error running {method_name} on {modality}: {e}")
                    state["error"] = str(e)
                    raise

        return state

# ==========================================
# 4. Evaluation Agent
# ==========================================
class EvaluationAgent(BaseAgent):
    def __init__(self):
        super().__init__("Evaluator")

    def _compute_batch_asw(self, embedding: np.ndarray, labels: pd.Series) -> Optional[float]:
        unique_labels = labels.dropna().unique()
        if len(unique_labels) < 2 or embedding.shape[0] <= len(unique_labels):
            return None
        try:
            return float(silhouette_score(embedding, labels))
        except Exception as e:
            logger.warning(f"Batch ASW failed: {e}")
            return None

    def _compute_celltype_asw(self, embedding: np.ndarray, labels: pd.Series) -> Optional[float]:
        unique_labels = labels.dropna().unique()
        if len(unique_labels) < 2 or embedding.shape[0] <= len(unique_labels):
            return None
        try:
            return float(silhouette_score(embedding, labels))
        except Exception as e:
            logger.warning(f"Celltype ASW failed: {e}")
            return None

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        results = state.get("results") or {}
        records = []

        for modality, methods in results.items():
            if not methods:
                continue
            for method_name, adata in methods.items():
                embedding_key = f"X_{method_name}"
                if embedding_key not in adata.obsm:
                    logger.warning(f"[Evaluation] Missing embedding {embedding_key} for {modality}-{method_name}, skip.")
                    continue

                emb = adata.obsm[embedding_key]
                # Graph connectivity (simple fallback)
                try:
                    gc_value = calculate_graph_connectivity(emb, n_neighbors=15, n_cells=emb.shape[0])
                    records.append({"modality": modality, "method": method_name, "metric": "graph_connectivity", "value": gc_value})
                except Exception as e:
                    logger.warning(f"[Evaluation] Graph connectivity failed for {modality}-{method_name}: {e}")

                # Batch ASW
                if "batch" in adata.obs:
                    batch_val = self._compute_batch_asw(emb, adata.obs["batch"])
                    if batch_val is not None:
                        records.append({"modality": modality, "method": method_name, "metric": "batch_asw", "value": batch_val})

                # Celltype ASW
                if "celltype" in adata.obs:
                    ct_val = self._compute_celltype_asw(emb, adata.obs["celltype"])
                    if ct_val is not None:
                        records.append({"modality": modality, "method": method_name, "metric": "celltype_asw", "value": ct_val})

        eval_df = pd.DataFrame(records, columns=["modality", "method", "metric", "value"])
        state["evaluation"] = eval_df
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
    def _print_evaluation(eval_df: Optional[pd.DataFrame]):
        print("\n--- Evaluation ---")
        if eval_df is None or eval_df.empty:
            print("No evaluation metrics were computed.")
            return
        for metric, subdf in eval_df.groupby("metric"):
            best_val = subdf["value"].max()
            print(f"\nMetric: {metric}")
            for _, row in subdf.iterrows():
                star = " *" if row["value"] == best_val else ""
                print(f"  {row['modality']} - {row['method']}: {row['value']:.4f}{star}")

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print("\n===== OMICS INTEGRATION REPORT =====")
        if state.get("error"):
            print(f"ERROR: {state.get('error')}")
        self._print_plan(state.get("plan") or {})
        self._print_results(state.get("results") or {})
        self._print_evaluation(state.get("evaluation"))

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
