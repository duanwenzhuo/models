# tools.py
# Core libraries
import scanpy as sc
import scanpy.external as sce
import scvi
import numpy as np
import pandas as pd
import os
import gc
import anndata
import scipy.sparse as sp
import re
from typing import Optional, Dict, Any, Union, Callable
import harmonypy as hm
import subprocess
import config
    
# Machine learning & data analysis
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import f1_score
from sklearn.linear_model import LinearRegression

# Scipy for sparse matrix handling
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix
from scipy.stats import chisquare

# Logging
import logging
logger = logging.getLogger(f"{config.LOGGER_NAME}.{__name__}")

CELL_CYCLE_GENES: Dict[str, list[str]] = {
    "s_genes": [
        "MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG", "GINS2",
        "MCM6", "CDCA7", "DTL", "PRIM1", "UHRF1", "HELLS", "RFC2", "RPA2", "NASP",
        "RAD51AP1", "GMNN", "WDR76", "SLBP", "CCNE2", "UBR7", "POLD3", "MSH2",
        "ATAD2", "RAD51", "RRM2", "CDC45", "CDC6", "EXO1", "TIPIN", "DSCC1",
        "BLM", "CASP8AP2", "USP1", "CLSPN", "POLA1", "CHAF1B", "BRIP1",
        "E2F8"
    ],
    "g2m_genes": [
        "HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A", "NDC80",
        "CKS2", "NUF2", "CKS1B", "MKI67", "TMPO", "CENPF", "TACC3", "FAM64A",
        "SMC4", "CCNB2", "CKAP2L", "CKAP2", "AURKB", "BUB1", "KIF11", "ANP32E",
        "TUBB4B", "GTSE1", "KIF20B", "HJURP", "CDCA3", "HN1", "CDC20", "TTK",
        "CDC25C", "KIF2C", "RANGAP1", "NCAPD2", "DLGAP5", "CDCA2", "CDCA8",
        "ECT2", "KIF23", "HMMR", "AURKA", "PSRC1", "ANLN", "LBR", "CKAP5",
        "CENPE", "CTCF", "NEK2", "G2E3", "GAS2L3", "CBX5", "CENPA"
    ],
}


def slugify_dataset_name(path_or_name: str) -> str:
    """
    Normalize a dataset name into a safe slug.
    Uses the parent folder and stem of the provided path when available.
    """
    path_str = path_or_name or ""
    base = os.path.basename(path_str)
    stem, _ext = os.path.splitext(base)
    parent = os.path.basename(os.path.dirname(path_str))
    raw_name = f"{parent}_{stem}" if parent else stem
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", raw_name.lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "dataset"


def build_result_filename(dataset_name: str, modality: str, method: str, benchmark_fraction: Optional[float]) -> str:
    """
    Build a standardized filename for integrated results.
    Example: integrated_dataset_rna_scvi.h5ad or integrated_dataset_atac_scvi_benchmark0p2.h5ad
    """
    dataset_slug = slugify_dataset_name(dataset_name)
    modality_slug = re.sub(r"[^0-9a-zA-Z]+", "_", str(modality).lower()).strip("_") or "modality"
    method_slug = re.sub(r"[^0-9a-zA-Z]+", "_", str(method).lower()).strip("_") or "method"
    filename = f"integrated_{dataset_slug}_{modality_slug}_{method_slug}"
    if benchmark_fraction is not None:
        try:
            frac = float(benchmark_fraction)
            suffix = str(frac).replace(".", "p")
        except Exception:
            suffix = "unknown"
        filename += f"_benchmark{suffix}"
    return f"{filename}.h5ad"

R_LIGER_SEURAT_SCRIPT_TEMPLATE = r"""
suppressPackageStartupMessages({
  library(Seurat)
  library(rliger)
})

DATA_INPUT_DIR <- "__DATA_DIR__"
RESULTS_OUTPUT_DIR <- "__DATA_DIR__"
BASE_NAME <- "__BASE_NAME__"
BATCH_KEY <- "__BATCH_KEY__"
IS_GENE_ACTIVITY <- __IS_GENE_ACTIVITY__
PROJECT_NAME <- "__PROJECT_NAME__"

CCA_FEATURE_NUM <- __CCA_FEATURE_NUM__
CCA_DIMS <- __CCA_DIMS__
CCA_K_ANCHOR <- __CCA_K_ANCHOR__
CCA_K_FILTER <- __CCA_K_FILTER__
LIGER_K <- __LIGER_K__
LIGER_LAMBDA <- __LIGER_LAMBDA__

COUNT_MATRIX_FILE <- file.path(DATA_INPUT_DIR, paste0(BASE_NAME, ".csv"))
METADATA_FILE <- file.path(DATA_INPUT_DIR, paste0(BASE_NAME, "_metadata.csv"))

message(sprintf("项目: %s", PROJECT_NAME))
message(sprintf("批次列: %s", BATCH_KEY))
message(sprintf("数据目录: %s", DATA_INPUT_DIR))
message(sprintf("Counts: %s", COUNT_MATRIX_FILE))
message(sprintf("Metadata: %s", METADATA_FILE))

load_and_split_data <- function() {
  counts_df <- read.csv(COUNT_MATRIX_FILE, row.names = 1, check.names = FALSE, stringsAsFactors = FALSE)
  metadata_df <- read.csv(METADATA_FILE, row.names = 1, check.names = FALSE, stringsAsFactors = FALSE)
  metadata_df$OriginalCellID <- rownames(metadata_df)

  if (!(BATCH_KEY %in% colnames(metadata_df))) {
    stop(sprintf("Metadata 缺少批次列 '%s'.", BATCH_KEY))
  }

  clean_matrix_cols <- trimws(colnames(counts_df))
  colnames(counts_df) <- clean_matrix_cols

  clean_meta_rows <- trimws(rownames(metadata_df))
  rownames(metadata_df) <- clean_meta_rows

  common_cells <- intersect(clean_matrix_cols, clean_meta_rows)
  if (length(common_cells) == 0) {
    stop("Counts 与 metadata 没有共同的细胞 ID.")
  }

  counts_matrix <- as.matrix(counts_df[, common_cells, drop = FALSE])
  storage.mode(counts_matrix) <- "double"
  metadata_df <- metadata_df[common_cells, , drop = FALSE]
  metadata_df[[BATCH_KEY]] <- as.factor(metadata_df[[BATCH_KEY]])

  batches <- unique(metadata_df[[BATCH_KEY]])
  seurat_object_list <- list()
  liger_raw_data_list <- list()

  for (batch_name in batches) {
    batch_cells <- rownames(metadata_df)[metadata_df[[BATCH_KEY]] == batch_name]
    batch_counts <- counts_matrix[, batch_cells, drop = FALSE]
    batch_counts <- as.matrix(batch_counts)
    storage.mode(batch_counts) <- "double"
    seurat_obj <- CreateSeuratObject(counts = batch_counts, project = as.character(batch_name),
                                     meta.data = metadata_df[batch_cells, , drop = FALSE])
    seurat_object_list[[as.character(batch_name)]] <- seurat_obj
    liger_raw_data_list[[as.character(batch_name)]] <- batch_counts
  }

  list(seurat_list = seurat_object_list, liger_list = liger_raw_data_list)
}

run_liger_integration <- function(liger_raw_data_list, output_prefix, is_gene_activity) {
  message("运行 LIGER ...")
  cell_ids <- unlist(lapply(liger_raw_data_list, colnames))
  liger_object <- createLiger(rawData = liger_raw_data_list, removeMissing = TRUE)
  if (nrow(liger_object@cellMeta) != length(cell_ids)) {
    stop(sprintf(
      "Number of cells in liger_object@cellMeta (%d) does not match length of cell_ids (%d).",
      nrow(liger_object@cellMeta), length(cell_ids)
    ))
  }
  liger_object@cellMeta$OriginalCellID <- cell_ids
  liger_object <- normalize(liger_object)
  num_genes_liger <- if (is_gene_activity) min(2000, nrow(liger_raw_data_list[[1]])) else 2000
  liger_object <- selectGenes(liger_object, num.genes = num_genes_liger)
  liger_object <- scaleNotCenter(liger_object)
  liger_object <- optimizeALS(liger_object, k = LIGER_K, lambda = LIGER_LAMBDA)
  tryCatch({ liger_object <- quantileNorm(liger_object) }, error = function(e) message("LIGER quantileNorm failed: ", e$message))
  tryCatch({ liger_object <- runCluster(liger_object, resolution = 0.4) }, error = function(e) message("LIGER clustering failed: ", e$message))
  tryCatch({ liger_object <- runUMAP(liger_object) }, error = function(e) message("LIGER runUMAP failed: ", e$message))

  dim_reds <- liger_object@dimReds
  umap_coords <- NULL
  if (!is.null(dim_reds) && "UMAP" %in% names(dim_reds)) {
    umap_candidate <- dim_reds$UMAP
    if (!is.null(umap_candidate) && ncol(umap_candidate) >= 2) {
      umap_coords <- umap_candidate[, 1:2, drop = FALSE]
    }
  }
  if (is.null(umap_coords)) {
    h_matrix <- t(liger_object@H)
    if (ncol(h_matrix) >= 2) {
      umap_coords <- h_matrix[, 1:2, drop = FALSE]
    } else {
      stop("LIGER integration failed to produce a usable embedding.")
    }
  }
  colnames(umap_coords) <- c("UMAP_1", "UMAP_2")

  cell_meta <- liger_object@cellMeta
  cluster_key <- NA
  if (!is.null(cell_meta)) {
    if ("quantileNorm_cluster" %in% colnames(cell_meta)) {
      cluster_key <- "quantileNorm_cluster"
    } else if ("leiden_cluster" %in% colnames(cell_meta)) {
      cluster_key <- "leiden_cluster"
    }
  }
  if (!is.na(cluster_key)) {
    cluster_vec <- cell_meta[[cluster_key]]
  } else {
    cluster_vec <- rep("cluster_1", nrow(umap_coords))
  }

  if (is.null(cell_meta) || !("OriginalCellID" %in% colnames(cell_meta))) {
    stop("OriginalCellID not found in liger_object@cellMeta after assignment.")
  }
  cell_ids_export <- as.character(cell_meta$OriginalCellID)
  if (length(cell_ids_export) != nrow(umap_coords)) {
    stop("Number of UMAP coordinates does not match length of OriginalCellID.")
  }

  export_df <- data.frame(
    UMAP_1 = umap_coords[, 1],
    UMAP_2 = umap_coords[, 2],
    LIGER_Cluster = cluster_vec,
    row.names = cell_ids_export
  )
  liger_csv <- file.path(RESULTS_OUTPUT_DIR, paste0(output_prefix, "_liger_umap_and_clusters.csv"))
  write.csv(export_df, liger_csv)

  rds_path <- file.path(RESULTS_OUTPUT_DIR, paste0(output_prefix, "_liger_integrated_result.rds"))
  saveRDS(liger_object, rds_path)
  message(sprintf("LIGER 结果导出: %s", liger_csv))
}

run_seurat_integration <- function(seurat_object_list, output_prefix, is_gene_activity) {
  message("运行 Seurat CCA ...")
  feature_num_param <- if (!is.na(CCA_FEATURE_NUM)) CCA_FEATURE_NUM else if (is_gene_activity) min(2000, nrow(seurat_object_list[[1]])) else 2000
  dims_to_use <- if (!is.null(CCA_DIMS)) CCA_DIMS else 1:30
  k_anchor_param <- ifelse(is.na(CCA_K_ANCHOR), 5, CCA_K_ANCHOR)
  k_filter_param <- ifelse(is.na(CCA_K_FILTER), 200, CCA_K_FILTER)

  seurat_list <- lapply(seurat_object_list, function(x) {
    x <- NormalizeData(x, verbose = FALSE)
    x <- FindVariableFeatures(x, selection.method = "vst", nfeatures = feature_num, verbose = FALSE)
    x
  })

  integration.anchors <- FindIntegrationAnchors(object.list = seurat_list, dims = dims_to_use, reduction = "cca", verbose = FALSE, k.anchor = k_anchor_param, k.filter = k_filter_param)
  integrated.seurat <- IntegrateData(anchorset = integration.anchors, dims = dims_to_use, verbose = FALSE)
  DefaultAssay(integrated.seurat) <- "integrated"
  integrated.seurat <- ScaleData(integrated.seurat, verbose = FALSE)
  max_dim <- max(dims_to_use)
  integrated.seurat <- RunPCA(integrated.seurat, npcs = max_dim, verbose = FALSE)
  integrated.seurat <- RunUMAP(integrated.seurat, dims = dims_to_use, verbose = FALSE)
  integrated.seurat <- FindNeighbors(integrated.seurat, dims = dims_to_use, verbose = FALSE)
  integrated.seurat <- FindClusters(integrated.seurat, resolution = 0.5, verbose = FALSE)

  umap_coords <- Embeddings(integrated.seurat, reduction = "umap")
  cluster_vec <- integrated.seurat@meta.data[[paste0("integrated_snn_res.", 0.5)]]

  if ("OriginalCellID" %in% colnames(integrated.seurat@meta.data)) {
    cell_ids <- as.character(integrated.seurat@meta.data$OriginalCellID)
  } else {
    cell_ids <- rownames(umap_coords)
  }

  export_df <- data.frame(
    UMAP_1 = umap_coords[, 1],
    UMAP_2 = umap_coords[, 2],
    Seurat_Cluster = cluster_vec,
    row.names = cell_ids
  )
  seurat_csv <- file.path(RESULTS_OUTPUT_DIR, paste0(output_prefix, "_seurat_umap_and_clusters.csv"))
  write.csv(export_df, seurat_csv)

  rds_path <- file.path(RESULTS_OUTPUT_DIR, paste0(output_prefix, "_seurat_cca_integrated_result.rds"))
  saveRDS(integrated.seurat, rds_path)
  message(sprintf("Seurat CCA 结果导出: %s", seurat_csv))
}

dir.create(RESULTS_OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)
message("开始 R 端整合流程 ...")

message("所有 R 整合流程完成。")
"""


def write_r_liger_seurat_script(
    output_dir: str,
    base_name: str,
    batch_key: str,
    is_gene_activity: bool,
    project_name: str = "Python_R_Integration",
    mode: str = "cca",
    cca_params: Optional[Dict[str, Any]] = None,
    liger_params: Optional[Dict[str, Any]] = None,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    cca_params = cca_params or {}
    liger_params = liger_params or {}

    feature_num = cca_params.get("nfeatures")
    dims = cca_params.get("dims")
    k_anchor = cca_params.get("k_anchor")
    k_filter = cca_params.get("k_filter")
    liger_k = liger_params.get("k")
    liger_lambda = liger_params.get("lambda") if isinstance(liger_params, dict) else None

    def _dims_to_r_expr(dims_val: Any) -> str:
        if isinstance(dims_val, (list, tuple)) and dims_val:
            # If sequential starting at 1, compress to 1:n
            try:
                dims_sorted = sorted(int(x) for x in dims_val)
                if dims_sorted[0] == 1 and dims_sorted == list(range(1, dims_sorted[-1] + 1)):
                    return f"1:{dims_sorted[-1]}"
                return "c(" + ",".join(str(x) for x in dims_sorted) + ")"
            except Exception:
                return "1:30"
        if isinstance(dims_val, int):
            return f"1:{dims_val}"
        return "1:30"

    dims_expr = _dims_to_r_expr(dims)

    script_content = R_LIGER_SEURAT_SCRIPT_TEMPLATE
    script_content = script_content.replace("__DATA_DIR__", output_dir.replace("\\", "/"))
    script_content = script_content.replace("__BASE_NAME__", base_name)
    script_content = script_content.replace("__BATCH_KEY__", batch_key)
    script_content = script_content.replace("__IS_GENE_ACTIVITY__", "TRUE" if is_gene_activity else "FALSE")
    script_content = script_content.replace("__PROJECT_NAME__", project_name)
    script_content = script_content.replace("__CCA_FEATURE_NUM__", str(feature_num) if feature_num is not None else "NA")
    script_content = script_content.replace("__CCA_DIMS__", dims_expr)
    script_content = script_content.replace("__CCA_K_ANCHOR__", str(k_anchor) if k_anchor is not None else "NA")
    script_content = script_content.replace("__CCA_K_FILTER__", str(k_filter) if k_filter is not None else "NA")
    script_content = script_content.replace("__LIGER_K__", str(liger_k) if liger_k is not None else "20")
    script_content = script_content.replace("__LIGER_LAMBDA__", str(liger_lambda) if liger_lambda is not None else "5")
    mode_normalized = (mode or "").lower()
    if mode_normalized == "liger":
        main_calls = (
            "data_lists <- load_and_split_data()\n"
            "run_liger_integration(data_lists$liger_list, BASE_NAME, IS_GENE_ACTIVITY)\n"
        )
    elif mode_normalized == "cca":
        main_calls = (
            "data_lists <- load_and_split_data()\n"
            "run_seurat_integration(data_lists$seurat_list, BASE_NAME, IS_GENE_ACTIVITY)\n"
        )
    else:
        main_calls = (
            "data_lists <- load_and_split_data()\n"
            "run_liger_integration(data_lists$liger_list, BASE_NAME, IS_GENE_ACTIVITY)\n"
            "run_seurat_integration(data_lists$seurat_list, BASE_NAME, IS_GENE_ACTIVITY)\n"
        )
        script_content = script_content.replace("__MAIN_CALLS__", main_calls.rstrip())
    script_path = os.path.join(output_dir, f"{base_name}_liger_seurat_integration.R")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_content)
    return script_path
# ======================================================================
# A. 核心功能类 (Core Functionality Classes - 移除 DataConfig 依赖)
# ======================================================================

# 注：DatasetFeatureChecker 类被移除，因为它不被 agents.py 调用。

class DataPreprocessor:
    """数据预处理类，处理过滤、标准化、PCA 和 UMAP"""
    # 构造函数现在直接接受所需的配置参数，而不是 DataConfig 对象
    def __init__(self, adata: sc.AnnData, is_count_data: bool = True):
        self.adata = adata
        self.is_count_data = is_count_data

    def preprocess(self) -> tuple[sc.AnnData, sc.AnnData]: 
        """执行过滤、HVG、PCA 和 UMAP。返回 (adata_hvg, adata_raw)。"""
        adata = self.adata.copy()
        
        # --- 过滤 (使用默认值) ---
        sc.pp.filter_cells(adata, min_genes=100 if self.is_count_data else 10)
        sc.pp.filter_genes(adata, min_cells=1)
        
        # QC Metrics
        if self.is_count_data:
            adata.var['mt'] = adata.var_names.str.startswith('MT-')
            sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
        else:
            sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)

        raw_adata_storage = adata.copy() # Raw data for scVI
        
        # --- HVG & PCA & UMAP ---
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        adata_hvg = adata[:, adata.var['highly_variable']].copy()
        sc.pp.scale(adata_hvg, max_value=10)
        sc.tl.pca(adata_hvg, svd_solver='arpack')
        sc.pp.neighbors(adata_hvg, n_neighbors=10, n_pcs=40)
        sc.tl.umap(adata_hvg)
        sc.tl.leiden(adata_hvg, resolution=0.5)
        adata_hvg.raw = raw_adata_storage # Store raw counts (required by scVI)
        return adata_hvg, raw_adata_storage

class ScVIIntegration:
    """scVI ????"""
    # ?????????????????
    def __init__(self, adata_hvg: sc.AnnData, adata_raw_full: sc.AnnData, batch_key: str, 
                 is_count_data: bool = True, model_save_path: Optional[str] = None, 
                 n_latent: int = 20, max_epochs: int = 50, batch_size: Optional[int] = None,
                 n_layers: Optional[int] = None, early_stopping: bool = False,
                 early_stopping_patience: int = 20):
        self.adata_hvg = adata_hvg
        self.adata_raw_full = adata_raw_full
        self.batch_key = batch_key
        self.is_count_data = is_count_data
        self.model_save_path = model_save_path
        self.n_latent = n_latent
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience

    def integrate(self) -> sc.AnnData:
        """?? scVI ??"""
        # ?? raw data ? hvg ??? cell ? gene ????
        adata_raw = self.adata_raw_full[self.adata_hvg.obs_names, self.adata_hvg.var_names].copy()
        adata_raw.obs[self.batch_key] = self.adata_hvg.obs[self.batch_key].copy()
        
        layer_name = "counts" if self.is_count_data else "activity"
        adata_raw.layers[layer_name] = adata_raw.X.copy() 

        scvi.model.SCVI.setup_anndata(adata_raw, layer=layer_name, batch_key=self.batch_key)
        
        # ???????
        vae = None
        if self.model_save_path and os.path.exists(self.model_save_path):
            vae = scvi.model.SCVI.load(self.model_save_path, adata=adata_raw)

        if vae is None:
            n_layers = self.n_layers if self.n_layers is not None else 2
            vae = scvi.model.SCVI(adata_raw, n_layers=n_layers, n_latent=self.n_latent, n_hidden=128, gene_likelihood="nb")
            vae.train(
                max_epochs=self.max_epochs,
                batch_size=self.batch_size,
                enable_progress_bar=False,
                early_stopping=self.early_stopping,
                early_stopping_patience=self.early_stopping_patience,
            ) 
            if self.model_save_path:
                vae.save(self.model_save_path, overwrite=True)
        
        latent_representation = vae.get_latent_representation()
        scvi_adata = self.adata_hvg.copy()
        scvi_adata.obsm["X_scvi"] = latent_representation
        sc.pp.neighbors(scvi_adata, use_rep="X_scvi", n_neighbors=15)
        sc.tl.umap(scvi_adata)
        sc.tl.leiden(scvi_adata, resolution=0.5)
        return scvi_adata
class HarmonyIntegration:
    """Harmony ????"""
    # ?????????????????
    def __init__(self, adata: sc.AnnData, batch_key: str = 'batch', n_pcs: Optional[int] = None,
                 theta: Optional[float] = None, lambda_: Optional[float] = None, nclust: Optional[int] = None):
        self.adata = adata
        self.batch_key = batch_key
        self.n_pcs = n_pcs
        self.theta = theta
        self.lambda_ = lambda_
        self.nclust = nclust

    def integrate(self) -> sc.AnnData:
        """?? Harmony ??"""
        adata = self.adata.copy()
        if self.n_pcs:
            if 'X_pca' not in adata.obsm or adata.obsm['X_pca'].shape[1] < self.n_pcs:
                sc.tl.pca(adata, n_comps=self.n_pcs)
            else:
                adata.obsm['X_pca'] = adata.obsm['X_pca'][:, : self.n_pcs]
        data_mat = adata.obsm.get('X_pca')
        if data_mat is None:
            sc.tl.pca(adata)
            data_mat = adata.obsm['X_pca']
        meta_data = adata.obs
        harmony_kwargs = {}
        if self.theta is not None:
            harmony_kwargs["theta"] = self.theta
        if self.lambda_ is not None:
            harmony_kwargs["lambda_"] = self.lambda_
        if self.nclust is not None:
            harmony_kwargs["nclust"] = self.nclust
        ho = hm.run_harmony(data_mat, meta_data, self.batch_key, **harmony_kwargs)
        adata.obsm['X_harmony'] = ho.Z_corr.T
        sc.pp.neighbors(adata, use_rep="X_harmony", n_neighbors=15)
        sc.tl.umap(adata)
        sc.tl.leiden(adata, resolution=0.5)
        return adata
class MNNIntegration:
    """MNN (Mutual Nearest Neighbors) ????"""
    def __init__(self, adata: sc.AnnData, batch_key: str = 'batch', n_pcs: Optional[int] = None,
                 k: Optional[int] = None, cos_norm: Optional[bool] = None):
        self.adata = adata
        self.method_name = "mnn"
        self.batch_key = batch_key
        self.n_pcs = n_pcs
        self.k = k
        self.cos_norm = cos_norm

    def integrate(self) -> anndata.AnnData:
        """
        ?? MNN ????????? X_mnn ???
        """
        logger.info("Starting MNN correction with batch_key=%s", self.batch_key)
        adata = self.adata.copy()
        if self.n_pcs:
            if 'X_pca' not in adata.obsm or adata.obsm['X_pca'].shape[1] < self.n_pcs:
                sc.tl.pca(adata, n_comps=self.n_pcs)
            else:
                adata.obsm['X_pca'] = adata.obsm['X_pca'][:, : self.n_pcs]
        try:
            mnn_kwargs = {
                "batch_key": self.batch_key,
                "use_rep": "X_pca",
                "do_concatenate": False,
            }
            if self.n_pcs is not None:
                mnn_kwargs["n_pcs"] = self.n_pcs
            if self.k is not None:
                mnn_kwargs["k"] = self.k
            if self.cos_norm is not None:
                mnn_kwargs["cos_norm"] = self.cos_norm
            corrected_data_or_tuple = sce.pp.mnn_correct(
                adata,
                **mnn_kwargs,
            )
        except Exception as e:
            raise RuntimeError(f"MNN correction failed in sce.pp.mnn_correct: {e}")

        if isinstance(corrected_data_or_tuple, anndata.AnnData):
            corrected_adata = corrected_data_or_tuple
        elif isinstance(corrected_data_or_tuple, tuple):
            current_obj = corrected_data_or_tuple
            while isinstance(current_obj, tuple) and len(current_obj) > 0:
                current_obj = current_obj[0]
            if not isinstance(current_obj, anndata.AnnData):
                raise TypeError(
                    f"mnn_correct returned a tuple but no AnnData object could be extracted: "
                    f"types={tuple(type(x) for x in corrected_data_or_tuple)}"
                )
            corrected_adata = current_obj
        else:
            raise TypeError(f"Unexpected return type from mnn_correct: {type(corrected_data_or_tuple)}")

        if "X_pca" not in corrected_adata.obsm or corrected_adata.obsm["X_pca"].shape[1] < 2:
            raise RuntimeError("MNN-corrected AnnData has invalid or missing 'X_pca' embedding.")

        corrected_adata.obsm["X_mnn"] = corrected_adata.obsm["X_pca"].copy()

        try:
            sc.pp.neighbors(corrected_adata, use_rep="X_mnn", n_neighbors=15)
            sc.tl.umap(corrected_adata)
            sc.tl.leiden(corrected_adata, resolution=0.5, key_added="leiden_mnn")
        except Exception as e:
            raise RuntimeError(f"MNN postprocessing (neighbors/UMAP/Leiden) failed: {e}")

        logger.info("MNN integration finished; X_mnn and X_umap computed.")
        return corrected_adata
# --- OmicsTools 统一接口 (Wrapper for Agents) ---

class OmicsTools:
    """提供了供 Agent 调用的统一工具接口。"""

    @staticmethod
    def load_and_preprocess(
        data_path: str,
        original_batch_key: str = "batch",
        is_count_data: bool = True,
        modality: str = "RNA",
    ) -> tuple[sc.AnnData, sc.AnnData]:
        """
        加载数据并执行基本预处理 (QC, HVG, PCA, UMAP)。
        original_batch_key 会被映射/复制到标准化的 adata.obs['batch']。
        modality 预留给多模态扩展；当前 ATAC/ADT 暂走 RNA 流程。
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        adata = sc.read(data_path)

        # 批次键标准化：若缺失则视为配置错误
        if original_batch_key in adata.obs:
            adata.obs["batch"] = adata.obs[original_batch_key].astype("category")
        else:
            raise KeyError(f"Batch key '{original_batch_key}' not found in adata.obs")

        # 预处理分支（未来可扩展）
        modality_upper = (modality or "RNA").upper()
        if modality_upper in ["RNA", "ATAC", "ADT"]:
            # TODO: 为 ATAC / ADT 替换为各自专用预处理流程
            preprocessor = DataPreprocessor(adata=adata, is_count_data=is_count_data)
        else:
            preprocessor = DataPreprocessor(adata=adata, is_count_data=is_count_data)

        return preprocessor.preprocess()
        
    @staticmethod
    def run_scvi(adata_hvg: sc.AnnData, adata_raw: sc.AnnData, batch_key: str, **kwargs) -> sc.AnnData:
        """?? scVI ???"""
        n_latent = kwargs.get('n_latent', 20)
        max_epochs = kwargs.get('max_epochs', 50)
        batch_size = kwargs.get('batch_size')
        is_count_data = kwargs.get('is_count_data', True)
        n_layers = kwargs.get('n_layers')
        model_save_path = kwargs.get('model_save_path')
        early_stopping = kwargs.get('early_stopping', False)
        early_stopping_patience = kwargs.get('early_stopping_patience', 20)

        integrator = ScVIIntegration(
            adata_hvg=adata_hvg,
            adata_raw_full=adata_raw,
            batch_key=batch_key,
            is_count_data=is_count_data,
            model_save_path=model_save_path,
            n_latent=n_latent,
            max_epochs=max_epochs,
            batch_size=batch_size,
            n_layers=n_layers,
            early_stopping=early_stopping,
            early_stopping_patience=early_stopping_patience,
        )
        return integrator.integrate()

    @staticmethod
    def run_harmony(adata_hvg: sc.AnnData, batch_key: str, **kwargs) -> sc.AnnData:
        """?? Harmony ???"""
        integrator = HarmonyIntegration(
            adata=adata_hvg,
            batch_key=batch_key,
            n_pcs=kwargs.get('n_pcs'),
            theta=kwargs.get('theta'),
            lambda_=kwargs.get('lambda_'),
            nclust=kwargs.get('nclust'),
        )
        return integrator.integrate()

    @staticmethod
    def run_mnn(adata_hvg: sc.AnnData, batch_key: str, **kwargs) -> sc.AnnData:
        """?? MNN ???"""
        integrator = MNNIntegration(
            adata=adata_hvg,
            batch_key=batch_key,
            n_pcs=kwargs.get('n_pcs'),
            k=kwargs.get('k'),
            cos_norm=kwargs.get('cos_norm'),
        )
        return integrator.integrate()

    @staticmethod
    def export_adata_for_r(
        adata_hvg: sc.AnnData,
        adata_raw: Optional[sc.AnnData],
        batch_key: str,
        base_name: str,
        output_dir: str,
    ) -> tuple[str, str]:
        os.makedirs(output_dir, exist_ok=True)
        adata_source = adata_raw if adata_raw is not None else adata_hvg
        common_cells = adata_hvg.obs_names.intersection(adata_source.obs_names)
        if len(common_cells) == 0:
            raise ValueError("No overlapping cells between HVG and source AnnData for R export.")
        adata_source = adata_source[common_cells].copy()
        adata_hvg_aligned = adata_hvg[common_cells].copy()
        X = adata_source.X
        if sp.issparse(X):
            X = X.toarray()
        counts_df = pd.DataFrame(
            X.T,
            index=adata_source.var_names,
            columns=adata_source.obs_names,
        )
        counts_path = os.path.join(output_dir, f"{base_name}.csv")
        counts_df.to_csv(counts_path)

        meta = adata_hvg_aligned.obs.copy()
        if batch_key not in meta.columns:
            raise KeyError(f"Batch key '{batch_key}' not found in metadata for R export.")
        meta.index.name = "Cell_ID"
        meta_path = os.path.join(output_dir, f"{base_name}_metadata.csv")
        meta.to_csv(meta_path)

        logger.info(f"[R Export] Counts saved to {counts_path}, metadata saved to {meta_path}")
        return counts_path, meta_path

    @staticmethod
    def run_external_r_integration(
        adata_hvg: sc.AnnData,
        adata_raw: Optional[sc.AnnData],
        batch_key: str,
        method_name: str,
        base_name: str = "benchmark_subset",
        output_dir: Optional[str] = None,
        is_gene_activity: bool = False,
        preset_params: Optional[Dict[str, Any]] = None,
    ) -> Optional[sc.AnnData]:
        if output_dir is None:
            output_dir = os.path.join(config.RESULTS_DIR, "r_external")
        os.makedirs(output_dir, exist_ok=True)

        OmicsTools.export_adata_for_r(
            adata_hvg=adata_hvg,
            adata_raw=adata_raw,
            batch_key=batch_key,
            base_name=base_name,
            output_dir=output_dir,
        )

        preset_params = preset_params or {}
        method_lower = method_name.lower()
        script_path = write_r_liger_seurat_script(
            output_dir=output_dir,
            base_name=base_name,
            batch_key=batch_key,
            is_gene_activity=is_gene_activity,
            project_name=f"{method_name}_integration",
            mode=method_lower if method_lower in {"cca", "liger"} else "cca",
            cca_params=preset_params if method_lower == "cca" else None,
            liger_params=preset_params if method_lower == "liger" else None,
        )

        result = subprocess.run(
            [config.RSCRIPT_PATH, script_path],
            cwd=output_dir,
            capture_output=True,
            text=False,
        )
        stdout_bytes = result.stdout or b""
        stderr_bytes = result.stderr or b""
        try:
            stdout_text = stdout_bytes.decode("utf-8", errors="ignore")
        except Exception:
            stdout_text = stdout_bytes.decode(errors="ignore")
        try:
            stderr_text = stderr_bytes.decode("utf-8", errors="ignore")
        except Exception:
            stderr_text = stderr_bytes.decode(errors="ignore")
        if result.returncode != 0:
            if stderr_text:
                logger.error(f"Rscript failed with code {result.returncode}. stderr:\n{stderr_text}")
            else:
                logger.error(f"Rscript failed with code {result.returncode}, stderr is empty.")
            raise RuntimeError(f"External R integration failed for {method_name}.")
        if stdout_text:
            logger.info(f"Rscript stdout:\n{stdout_text}")
        else:
            logger.info("Rscript stdout is empty.")

        if method_lower == "liger":
            csv_path = os.path.join(output_dir, f"{base_name}_liger_umap_and_clusters.csv")
            if not os.path.exists(csv_path):
                logger.error(f"LIGER output not found at {csv_path}")
                return None
            df = pd.read_csv(csv_path, index_col=0)
            for col in ("UMAP_1", "UMAP_2"):
                if col not in df.columns:
                    raise KeyError(f"LIGER output missing column {col}")
            common_cells = adata_hvg.obs_names.intersection(df.index)
            if len(common_cells) == 0:
                raise ValueError("No overlapping cells between HVG data and LIGER output.")
            df = df.loc[common_cells]
            adata_out = adata_hvg[common_cells].copy()
            adata_out.obsm["X_liger"] = df[["UMAP_1", "UMAP_2"]].to_numpy()
            if "LIGER_Cluster" in df.columns:
                adata_out.obs["liger_cluster"] = df["LIGER_Cluster"].astype("category")
            return adata_out

        if method_lower == "cca":
            csv_path = os.path.join(output_dir, f"{base_name}_seurat_umap_and_clusters.csv")
            if not os.path.exists(csv_path):
                logger.error(f"Seurat CCA output not found at {csv_path}")
                return None
            df = pd.read_csv(csv_path, index_col=0)
            for col in ("UMAP_1", "UMAP_2"):
                if col not in df.columns:
                    raise KeyError(f"Seurat CCA output missing column {col}")
            common_cells = adata_hvg.obs_names.intersection(df.index)
            if len(common_cells) == 0:
                raise ValueError("No overlapping cells between HVG data and Seurat CCA output.")
            df = df.loc[common_cells]
            adata_out = adata_hvg[common_cells].copy()
            adata_out.obsm["X_cca"] = df[["UMAP_1", "UMAP_2"]].to_numpy()
            if "Seurat_Cluster" in df.columns:
                adata_out.obs["cca_cluster"] = df["Seurat_Cluster"].astype("category")
            return adata_out

        raise ValueError(f"Unsupported external R method {method_name}")
# --- Integration method registry -------------------------------------------------

def _run_scvi_registered(adata_hvg: sc.AnnData, adata_raw: Optional[sc.AnnData], batch_key: str = "batch", **kwargs) -> Optional[sc.AnnData]:
    if adata_raw is None:
        raise ValueError("scVI integration requires raw AnnData input.")
    return OmicsTools.run_scvi(adata_hvg, adata_raw, batch_key, **kwargs)


def _run_harmony_registered(adata_hvg: sc.AnnData, adata_raw: Optional[sc.AnnData], batch_key: str = "batch", **kwargs) -> Optional[sc.AnnData]:
    return OmicsTools.run_harmony(adata_hvg, batch_key, **kwargs)


def _run_mnn_registered(adata_hvg: sc.AnnData, adata_raw: Optional[sc.AnnData], batch_key: str = "batch", **kwargs) -> Optional[sc.AnnData]:
    return OmicsTools.run_mnn(adata_hvg, batch_key, **kwargs)


def _run_cca_registered(adata_hvg: sc.AnnData, adata_raw: Optional[sc.AnnData], batch_key: str = "batch", **kwargs) -> Optional[sc.AnnData]:
    """Run Seurat CCA integration via external Rscript pipeline."""
    return OmicsTools.run_external_r_integration(
        adata_hvg=adata_hvg,
        adata_raw=adata_raw,
        batch_key=batch_key,
        method_name="cca",
        base_name="benchmark_subset",
        output_dir=os.path.join(config.RESULTS_DIR, "r_external"),
        is_gene_activity=False,
        preset_params=kwargs,
    )


# NOTE: All integration runners must treat their AnnData inputs as method-local, only
# add embeddings relevant to themselves (e.g., "X_scvi"), and return a fresh AnnData
# object that does not contain embeddings from other methods. IntegrationAgent already
# passes copies of the preprocessed AnnData to enforce isolation.
def _run_liger_registered(adata_hvg: sc.AnnData, adata_raw: Optional[sc.AnnData], batch_key: str = "batch", **kwargs) -> Optional[sc.AnnData]:
    """Run LIGER integration via external Rscript pipeline."""
    return OmicsTools.run_external_r_integration(
        adata_hvg=adata_hvg,
        adata_raw=adata_raw,
        batch_key=batch_key,
        method_name="liger",
        base_name="benchmark_subset",
        output_dir=os.path.join(config.RESULTS_DIR, "r_external"),
        is_gene_activity=False,
        preset_params=kwargs,
    )


# IntegrationRunner contract: each runner receives an AnnData copy, must not mutate
# external objects, and should return an AnnData containing only its own embeddings.
IntegrationRunner = Callable[[sc.AnnData, Optional[sc.AnnData], str], Optional[sc.AnnData]]
INTEGRATION_METHODS: Dict[str, IntegrationRunner] = {
    "scvi": _run_scvi_registered,
    "harmony": _run_harmony_registered,
    "mnn": _run_mnn_registered,
    "cca": _run_cca_registered,
    "liger": _run_liger_registered,
}

# ======================================================================
# C. 评估工具 (Evaluation Tools)
# ======================================================================

METRIC_SPECS: Dict[str, Dict[str, Any]] = {
    "graph_connectivity": {
        "id": "graph_connectivity",
        "name": "Graph connectivity",
        "group": "batch",
        "modalities": ["RNA", "ATAC", "ADT", "MULTI"],
        "requires": ["integrated", "embedding"],
        "range": (0.0, 1.0),
        "direction": "higher_better",
        "weight": 1.0,
        "implemented": True,
    },
    "batch_asw": {
        "id": "batch_asw",
        "name": "Batch ASW",
        "group": "batch",
        "modalities": ["RNA", "ATAC", "ADT", "MULTI"],
        "requires": ["integrated", "embedding", "batch"],
        "range": (-1.0, 1.0),
        "direction": "lower_better",
        "weight": 1.0,
        "implemented": True,
    },
    "ilisi": {
        "id": "ilisi",
        "name": "iLISI",
        "group": "batch",
        "modalities": ["RNA", "ATAC", "ADT", "MULTI"],
        "requires": ["integrated", "embedding", "batch"],
        "range": (0.0, 5.0),
        "direction": "higher_better",
        "weight": 1.5,
        "implemented": True,
    },
    "kbet": {
        "id": "kbet",
        "name": "kBET rejection rate",
        "group": "batch",
        "modalities": ["RNA", "ATAC", "ADT", "MULTI"],
        "requires": ["integrated", "embedding", "batch"],
        "range": (0.0, 1.0),
        "direction": "lower_better",
        "weight": 1.5,
        "implemented": True,
    },
    "pcr_batch": {
        "id": "pcr_batch",
        "name": "Principal component regression (batch)",
        "group": "batch",
        "modalities": ["RNA", "ATAC", "ADT", "MULTI"],
        "requires": ["integrated", "embedding", "batch"],
        "range": (0.0, 1.0),
        "direction": "lower_better",
        "weight": 1.0,
        "implemented": True,
    },
    "ct_asw": {
        "id": "ct_asw",
        "name": "Cell type ASW",
        "group": "bio_label",
        "modalities": ["RNA", "ATAC", "ADT", "MULTI"],
        "requires": ["integrated", "embedding", "celltype"],
        "range": (-1.0, 1.0),
        "direction": "higher_better",
        "weight": 1.0,
        "implemented": True,
    },
    "nmi": {
        "id": "nmi",
        "name": "Normalized mutual information",
        "group": "bio_label",
        "modalities": ["RNA", "ATAC", "ADT", "MULTI"],
        "requires": {"integrated", "celltype", "cluster"},
        "range": (0.0, 1.0),
        "direction": "higher_better",
        "weight": 1.0,
        "implemented": True,
    },
    "ari": {
        "id": "ari",
        "name": "Adjusted Rand index",
        "group": "bio_label",
        "modalities": ["RNA", "ATAC", "ADT", "MULTI"],
        "requires": {"integrated", "celltype", "cluster"},
        "range": (-1.0, 1.0),
        "direction": "higher_better",
        "weight": 1.0,
        "implemented": True,
    },
    "isolated_label_f1": {
        "id": "isolated_label_f1",
        "name": "Isolated label F1",
        "group": "bio_label",
        "modalities": ["RNA", "ATAC", "ADT", "MULTI"],
        "requires": {"integrated", "embedding", "celltype", "cluster", "batch"},
        "range": (0.0, 1.0),
        "direction": "higher_better",
        "weight": 1.5,
        "implemented": True,
    },
    "isolated_label_asw": {
        "id": "isolated_label_asw",
        "name": "Isolated label ASW",
        "group": "bio_label",
        "modalities": ["RNA", "ATAC", "ADT", "MULTI"],
        "requires": {"integrated", "embedding", "celltype", "batch"},
        "range": (-1.0, 1.0),
        "direction": "higher_better",
        "weight": 1.0,
        "implemented": True,
    },
    "clisi": {
        "id": "clisi",
        "name": "cell-type LISI (cLISI)",
        "group": "bio_label",
        "modalities": ["RNA", "ATAC", "ADT", "MULTI"],
        "requires": {"integrated", "embedding", "celltype"},
        "range": (0.0, 5.0),
        "direction": "higher_better",
        "weight": 1.5,
        "implemented": True,
    },
    "hvg_overlap": {
        "id": "hvg_overlap",
        "name": "HVG overlap",
        "group": "bio_label_free",
        "modalities": ("RNA",),
        "requires": {"unintegrated", "batch"},
        "range": (0.0, 1.0),
        "direction": "higher_better",
        "weight": 1.0,
        "implemented": True,
    },
    "cell_cycle_conservation": {
        "id": "cell_cycle_conservation",
        "name": "Cell cycle conservation",
        "group": "bio_label_free",
        "modalities": ("RNA",),
        "requires": {"unintegrated", "batch"},
        "range": (0.0, 1.0),
        "direction": "higher_better",
        "weight": 1.0,
        "implemented": True,
    },
    "trajectory_conservation": {
        "id": "trajectory_conservation",
        "name": "Trajectory conservation",
        "group": "bio_label_free",
        "modalities": ["RNA", "ATAC", "MULTI"],
        "requires": ["integrated", "trajectory"],
        "range": (0.0, 1.0),
        "direction": "higher_better",
        "weight": 1.5,
        "implemented": False,
    },
    "atac_peak_gene_consistency": {
        "id": "atac_peak_gene_consistency",
        "name": "ATAC peak-gene consistency",
        "group": "bio_label_free",
        "modalities": ["ATAC"],
        "requires": ["integrated", "atac_peak", "gene_activity"],
        "range": (0.0, 1.0),
        "direction": "higher_better",
        "weight": 1.5,
        "implemented": False,
    },
    "atac_motif_gene_activity_consistency": {
        "id": "atac_motif_gene_activity_consistency",
        "name": "ATAC motif/gene-activity consistency",
        "group": "bio_label_free",
        "modalities": ["ATAC"],
        "requires": ["integrated", "motif", "gene_activity"],
        "range": (0.0, 1.0),
        "direction": "higher_better",
        "weight": 1.5,
        "implemented": False,
    },
    "cross_modal_modality_mixing": {
        "id": "cross_modal_modality_mixing",
        "name": "Cross-modal modality mixing",
        "group": "cross_modal",
        "modalities": ["MULTI"],
        "requires": ["integrated", "embedding", "modality_label"],
        "range": (0.0, 1.0),
        "direction": "higher_better",
        "weight": 1.0,
        "implemented": False,
    },
    "cross_modal_label_transfer": {
        "id": "cross_modal_label_transfer",
        "name": "Cross-modal label transfer",
        "group": "cross_modal",
        "modalities": ["MULTI"],
        "requires": ["integrated", "embedding", "celltype", "modality_label"],
        "range": (0.0, 1.0),
        "direction": "higher_better",
        "weight": 1.5,
        "implemented": False,
    },
    "cross_modal_alignment": {
        "id": "cross_modal_alignment",
        "name": "Cross-modal alignment",
        "group": "cross_modal",
        "modalities": ["MULTI"],
        "requires": ["integrated", "embedding", "modality_label"],
        "range": (0.0, 1.0),
        "direction": "higher_better",
        "weight": 1.5,
        "implemented": False,
    },
}


def _compute_lisi(embedding: np.ndarray, labels: np.ndarray, n_neighbors: int = 30) -> np.ndarray:
    n_cells = embedding.shape[0]
    if n_cells < 2:
        return np.array([])
    k = min(max(n_neighbors, 1), n_cells - 1)
    if k < 1:
        return np.array([])
    nbrs = NearestNeighbors(n_neighbors=k)
    nbrs.fit(embedding)
    indices = nbrs.kneighbors(return_distance=False)
    lisi_values = np.zeros(n_cells, dtype=float)
    for i in range(n_cells):
        neigh_labels = labels[indices[i]]
        unique, counts = np.unique(neigh_labels, return_counts=True)
        probs = counts.astype(float) / counts.sum()
        lisi_values[i] = 1.0 / np.sum(probs ** 2)
    return lisi_values


def _compute_ilisi_batch(embedding: np.ndarray, batch_labels: np.ndarray, n_neighbors: int = 30) -> Optional[float]:
    lisi_values = _compute_lisi(embedding, batch_labels, n_neighbors=n_neighbors)
    if lisi_values.size == 0:
        return None
    return float(np.median(lisi_values))


def _compute_clisi_celltype(embedding: np.ndarray, ct_labels: np.ndarray, n_neighbors: int = 30) -> Optional[float]:
    lisi_values = _compute_lisi(embedding, ct_labels, n_neighbors=n_neighbors)
    if lisi_values.size == 0:
        return None
    median_lisi = float(np.median(lisi_values))
    if median_lisi <= 0:
        return None
    purity = 1.0 / median_lisi
    return float(np.clip(purity, 0.0, 5.0))


def _compute_kbet(
    embedding: np.ndarray,
    batch_labels: np.ndarray,
    n_neighbors: int = 50,
    alpha: float = 0.05,
    max_cells: int = 5000,
) -> Optional[float]:
    n_cells = embedding.shape[0]
    if n_cells < 2:
        return None
    if n_cells > max_cells:
        idx = np.random.choice(n_cells, size=max_cells, replace=False)
        embedding = embedding[idx]
        batch_labels = batch_labels[idx]
        n_cells = embedding.shape[0]
    k = min(max(n_neighbors, 5), n_cells - 1)
    if k < 1:
        return None
    nbrs = NearestNeighbors(n_neighbors=k)
    nbrs.fit(embedding)
    indices = nbrs.kneighbors(return_distance=False)
    batch_series = pd.Series(batch_labels)
    global_freq = batch_series.value_counts(normalize=True)
    if global_freq.empty:
        return None
    tested = 0
    rejected = 0
    for neigh_idx in indices:
        neigh_labels = batch_series.iloc[neigh_idx]
        obs_counts = neigh_labels.value_counts()
        expected = global_freq * len(neigh_idx)
        expected = expected.loc[expected > 0]
        if expected.empty:
            continue
        obs = obs_counts.reindex(expected.index).fillna(0).to_numpy()
        exp = expected.to_numpy()
        if exp.sum() == 0:
            continue
        try:
            _, p_val = chisquare(f_obs=obs, f_exp=exp)
        except Exception:
            continue
        tested += 1
        if p_val < alpha:
            rejected += 1
    if tested == 0:
        return None
    return float(rejected / tested)


def _compute_pcr_batch(embedding: np.ndarray, batch_labels: np.ndarray) -> Optional[float]:
    if embedding.shape[0] < 2:
        return None
    batch_df = pd.get_dummies(pd.Series(batch_labels).astype(str), drop_first=False)
    if batch_df.empty:
        return None
    B = batch_df.to_numpy(dtype=float)
    X = embedding - np.mean(embedding, axis=0, keepdims=True)
    variances = np.var(X, axis=0)
    total_var = variances.sum()
    if total_var <= 0:
        return None
    r2_values = []
    weights = []
    for dim in range(X.shape[1]):
        if variances[dim] <= 1e-12:
            continue
        model = LinearRegression(fit_intercept=True)
        try:
            model.fit(B, X[:, dim])
            r2 = model.score(B, X[:, dim])
        except Exception:
            continue
        r2_values.append(r2)
        weights.append(variances[dim])
    if not r2_values:
        return None
    weighted = np.average(r2_values, weights=weights)
    return float(np.clip(weighted, 0.0, 1.0))


def _find_isolated_celltypes(batch_labels: pd.Series, ct_labels: pd.Series) -> list[str]:
    df = pd.DataFrame({"batch": batch_labels, "celltype": ct_labels})
    counts = df.groupby("celltype")["batch"].nunique()
    return counts[counts == 1].index.tolist()


def _compute_isolated_label_f1(
    ct_labels: pd.Series,
    cluster_labels: pd.Series,
    batch_labels: pd.Series,
) -> Optional[float]:
    isolated = _find_isolated_celltypes(batch_labels, ct_labels)
    if not isolated:
        return None
    mask = ct_labels.isin(isolated)
    if mask.sum() < 5:
        return None
    df = pd.DataFrame({
        "celltype": ct_labels[mask],
        "cluster": cluster_labels[mask],
    }).dropna()
    if df.empty or df["celltype"].nunique() < 2:
        return None
    cluster_majority = df.groupby("cluster")["celltype"].agg(lambda s: s.value_counts().idxmax())
    predicted = df["cluster"].map(cluster_majority).fillna(df["celltype"])
    try:
        score = f1_score(
            df["celltype"],
            predicted,
            labels=list(isolated),
            average="macro",
        )
    except Exception:
        return None
    return float(np.clip(score, 0.0, 1.0))


def _compute_isolated_label_asw(
    embedding: np.ndarray,
    ct_labels: pd.Series,
    batch_labels: pd.Series,
) -> Optional[float]:
    isolated = _find_isolated_celltypes(batch_labels, ct_labels)
    if not isolated:
        return None
    mask = ct_labels.isin(isolated)
    if mask.sum() < 5:
        return None
    subset_labels = ct_labels[mask]
    if subset_labels.nunique() < 2:
        return None
    emb_subset = embedding[mask.to_numpy()]
    if emb_subset.shape[0] <= subset_labels.nunique():
        return None
    try:
        return float(silhouette_score(emb_subset, subset_labels))
    except Exception:
        return None


def _compute_hvg_overlap(adata_unintegrated: anndata.AnnData, batch_key: Optional[str]) -> Optional[float]:
    if adata_unintegrated is None or "highly_variable" not in adata_unintegrated.var:
        return None
    hv_mask = adata_unintegrated.var["highly_variable"].astype(bool)
    global_hvgs = set(adata_unintegrated.var_names[hv_mask])
    if not global_hvgs:
        return None
    if not batch_key or batch_key not in adata_unintegrated.obs:
        return 1.0
    union_hvgs: set[str] = set()
    for _, idx in adata_unintegrated.obs.groupby(batch_key).groups.items():
        subset = adata_unintegrated[idx].copy()
        if subset.n_obs < 5:
            continue
        try:
            n_top = min(2000, subset.n_vars)
            sc.pp.highly_variable_genes(subset, flavor="seurat", n_top_genes=n_top, inplace=True)
            hvgs = set(subset.var_names[subset.var["highly_variable"]])
        except Exception:
            continue
        union_hvgs.update(hvgs)
    if not union_hvgs:
        return None
    overlap = len(global_hvgs & union_hvgs)
    denom = len(global_hvgs | union_hvgs)
    if denom == 0:
        return None
    return float(overlap / denom)


def _compute_cell_cycle_conservation(adata_unintegrated: anndata.AnnData) -> Optional[float]:
    if adata_unintegrated is None or adata_unintegrated.n_obs < 3:
        return None
    if "batch" not in adata_unintegrated.obs:
        return None
    adata_cc = adata_unintegrated.copy()
    try:
        sc.tl.score_genes_cell_cycle(
            adata_cc,
            s_genes=CELL_CYCLE_GENES["s_genes"],
            g2m_genes=CELL_CYCLE_GENES["g2m_genes"],
        )
    except Exception:
        return None
    if "S_score" not in adata_cc.obs or "G2M_score" not in adata_cc.obs:
        return None
    scores = adata_cc.obs["S_score"] - adata_cc.obs["G2M_score"]
    df = pd.DataFrame({"score": scores, "batch": adata_cc.obs["batch"]}).dropna()
    if df.empty or df["batch"].nunique() < 2:
        return None
    overall_mean = df["score"].mean()
    total_var = df["score"].var(ddof=1)
    if not np.isfinite(total_var) or total_var <= 0:
        return None
    counts = df.groupby("batch")["score"].count()
    means = df.groupby("batch")["score"].mean()
    between = ((counts * (means - overall_mean) ** 2).sum()) / max(df.shape[0] - 1, 1)
    ratio = float(np.clip(between / total_var, 0.0, 1.0))
    # High value should indicate consistent cell-cycle patterns across batches
    return float(1.0 - ratio)

def normalize_metric_value(metric_id: str, raw_value: Optional[float]) -> Optional[float]:
    """
    Map a raw metric value into [0, 1] with 1 meaning best. Returns None when normalization fails.
    """
    if raw_value is None:
        return None
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    spec = METRIC_SPECS.get(metric_id)
    if spec is None:
        return None

    if metric_id in {"batch_asw", "ct_asw", "isolated_label_asw"}:
        mapped = (value + 1.0) / 2.0
        mapped = float(np.clip(mapped, 0.0, 1.0))
        if metric_id == "batch_asw":
            return float(np.clip(1.0 - mapped, 0.0, 1.0))
        return mapped

    if metric_id == "ari":
        mapped = (value + 1.0) / 2.0
        return float(np.clip(mapped, 0.0, 1.0))

    metric_range = spec.get("range")
    if not metric_range or len(metric_range) != 2:
        return None
    mn, mx = metric_range
    direction = spec.get("direction")
    if direction not in {"higher_better", "lower_better", "zero_best"}:
        return None
    if direction == "zero_best":
        denom = max(abs(mn), abs(mx), 1e-12)
        norm = 1.0 - min(abs(value) / denom, 1.0)
        return float(np.clip(norm, 0.0, 1.0))
    denom = mx - mn
    if denom <= 0:
        return None
    if direction == "higher_better":
        norm = (value - mn) / denom
    else:
        norm = (mx - value) / denom
    return float(np.clip(norm, 0.0, 1.0))

def _get_obs_series(
    adata: Optional[anndata.AnnData],
    key: Optional[str],
    reference_index: Optional[pd.Index],
) -> Optional[pd.Series]:
    if adata is None or key is None:
        return None
    if key not in adata.obs:
        return None
    series = adata.obs[key]
    if reference_index is not None:
        try:
            series = series.reindex(reference_index)
        except Exception:
            try:
                series = series.loc[reference_index]
            except Exception:
                return None
    return series

def compute_metric_raw(
    metric_id: str,
    adata_integrated: anndata.AnnData,
    adata_unintegrated: Optional[anndata.AnnData],
    embedding_key: str,
    modality: str,
    batch_key: Optional[str],
    celltype_key: Optional[str],
    cluster_key: Optional[str],
) -> Optional[float]:
    """
    Compute the raw metric value for a given integrated AnnData.
    """
    spec = METRIC_SPECS.get(metric_id)
    if spec is None:
        logger.warning(f"[Metrics] Unknown metric_id '{metric_id}'")
        return None
    if not spec.get("implemented", False):
        return None
    requires = set(spec.get("requires") or [])
    emb = None
    if "embedding" in requires:
        if embedding_key not in adata_integrated.obsm:
            logger.warning(f"[Metrics] Embedding '{embedding_key}' missing for metric '{metric_id}'")
            return None
        emb = np.asarray(adata_integrated.obsm[embedding_key])
    elif embedding_key in adata_integrated.obsm:
        emb = np.asarray(adata_integrated.obsm[embedding_key])

    try:
        if metric_id == "graph_connectivity":
            if emb is None:
                return None
            n_cells = emb.shape[0]
            if n_cells < 2:
                return None
            return float(calculate_graph_connectivity(emb, n_neighbors=15, n_cells=n_cells))

        if metric_id == "batch_asw":
            if emb is None:
                return None
            labels = _get_obs_series(adata_integrated, batch_key, adata_integrated.obs_names)
            if labels is None and batch_key:
                labels = _get_obs_series(adata_unintegrated, batch_key, adata_integrated.obs_names)
            if labels is None:
                return None
            labels = labels.astype("object")
            mask = labels.notna()
            if mask.sum() < 2:
                return None
            labels_filtered = labels[mask].astype(str)
            emb_filtered = emb[mask.to_numpy()]
            if len(pd.unique(labels_filtered)) < 2 or emb_filtered.shape[0] <= len(pd.unique(labels_filtered)):
                return None
            return float(silhouette_score(emb_filtered, labels_filtered))

        if metric_id == "ilisi":
            if emb is None or not batch_key:
                return None
            labels = _get_obs_series(adata_integrated, batch_key, adata_integrated.obs_names)
            if labels is None:
                return None
            values = _compute_ilisi_batch(emb, labels.astype(str).to_numpy())
            return values

        if metric_id == "kbet":
            if emb is None or not batch_key:
                return None
            labels = _get_obs_series(adata_integrated, batch_key, adata_integrated.obs_names)
            if labels is None:
                return None
            return _compute_kbet(emb, labels.astype(str).to_numpy())

        if metric_id == "pcr_batch":
            if emb is None or not batch_key:
                return None
            labels = _get_obs_series(adata_integrated, batch_key, adata_integrated.obs_names)
            if labels is None:
                return None
            return _compute_pcr_batch(emb, labels.astype(str).to_numpy())

        if metric_id == "ct_asw":
            if emb is None:
                return None
            labels = _get_obs_series(adata_integrated, celltype_key, adata_integrated.obs_names)
            if labels is None and celltype_key:
                labels = _get_obs_series(adata_unintegrated, celltype_key, adata_integrated.obs_names)
            if labels is None:
                return None
            labels = labels.astype("object")
            mask = labels.notna()
            if mask.sum() < 2:
                return None
            labels_filtered = labels[mask].astype(str)
            emb_filtered = emb[mask.to_numpy()]
            if len(pd.unique(labels_filtered)) < 2 or emb_filtered.shape[0] <= len(pd.unique(labels_filtered)):
                return None
            return float(silhouette_score(emb_filtered, labels_filtered))

        if metric_id in {"nmi", "ari"}:
            ct_series = _get_obs_series(adata_integrated, celltype_key, adata_integrated.obs_names)
            if ct_series is None and celltype_key:
                ct_series = _get_obs_series(adata_unintegrated, celltype_key, adata_integrated.obs_names)
            cluster_series = _get_obs_series(adata_integrated, cluster_key, adata_integrated.obs_names)
            if ct_series is None or cluster_series is None:
                return None
            df = pd.DataFrame({"celltype": ct_series, "cluster": cluster_series}).dropna()
            if df.empty:
                return None
            ct_vals = df["celltype"].astype(str).to_numpy()
            cl_vals = df["cluster"].astype(str).to_numpy()
            if len(np.unique(ct_vals)) < 2 or len(np.unique(cl_vals)) < 2:
                return None
            if metric_id == "nmi":
                return float(normalized_mutual_info_score(cl_vals, ct_vals))
            return float(adjusted_rand_score(cl_vals, ct_vals))

        if metric_id == "isolated_label_f1":
            if not all([batch_key, celltype_key, cluster_key]):
                return None
            batch_series = _get_obs_series(adata_integrated, batch_key, adata_integrated.obs_names)
            ct_series = _get_obs_series(adata_integrated, celltype_key, adata_integrated.obs_names)
            cluster_series = _get_obs_series(adata_integrated, cluster_key, adata_integrated.obs_names)
            if batch_series is None or ct_series is None or cluster_series is None:
                return None
            return _compute_isolated_label_f1(ct_series.astype(str), cluster_series.astype(str), batch_series.astype(str))

        if metric_id == "isolated_label_asw":
            if emb is None or not (batch_key and celltype_key):
                return None
            batch_series = _get_obs_series(adata_integrated, batch_key, adata_integrated.obs_names)
            ct_series = _get_obs_series(adata_integrated, celltype_key, adata_integrated.obs_names)
            if batch_series is None or ct_series is None:
                return None
            return _compute_isolated_label_asw(emb, ct_series.astype(str), batch_series.astype(str))

        if metric_id == "clisi":
            if emb is None or not celltype_key:
                return None
            ct_series = _get_obs_series(adata_integrated, celltype_key, adata_integrated.obs_names)
            if ct_series is None:
                return None
            return _compute_clisi_celltype(emb, ct_series.astype(str).to_numpy())

        if metric_id == "hvg_overlap":
            if adata_unintegrated is None:
                return None
            return _compute_hvg_overlap(adata_unintegrated, batch_key)

        if metric_id == "cell_cycle_conservation":
            return _compute_cell_cycle_conservation(adata_unintegrated)

    except Exception as exc:
        logger.warning(f"[Metrics] Metric '{metric_id}' failed for modality {modality}: {exc}")
        return None

    return None

def _clean_nested_dict(d: Any) -> tuple[Any, bool]:
    # Legacy helper kept for backwards compatibility; currently unused in the main pipeline.
    raise NotImplementedError("clean_nested_dict is not used in the current pipeline")

def _clean_adata_keys_minimal(adata: sc.AnnData) -> bool:
    # Legacy helper kept for backwards compatibility; currently unused in the main pipeline.
    raise NotImplementedError("clean_adata_keys_minimal is not used in the current pipeline")

def calculate_graph_connectivity(X_reduced: np.ndarray, n_neighbors: int, n_cells: int) -> float:
    if X_reduced is None:
        raise ValueError("X_reduced is None")
    if X_reduced.shape[0] != n_cells:
        raise ValueError("X_reduced shape does not match n_cells")
    if n_cells < 2:
        return 0.0
    n_neighbors = min(n_neighbors, n_cells - 1)
    knn = NearestNeighbors(n_neighbors=n_neighbors)
    knn.fit(X_reduced)
    graph = knn.kneighbors_graph(mode="connectivity")
    n_components, labels = connected_components(graph)
    largest = np.max(np.bincount(labels)) if labels.size else 0
    return float(largest) / float(n_cells)

def calculate_clustering_metrics(X_reduced: np.ndarray, celltype_labels: np.ndarray, n_clusters: int) -> dict:
    # Legacy helper kept for backwards compatibility; currently unused in the main pipeline.
    raise NotImplementedError("calculate_clustering_metrics is not used in the current pipeline")


def evaluate_integration_results(
    results: Dict[str, Optional[anndata.AnnData]],
    batch_key_global: str = "batch",
    celltype_key: str = "celltype",
) -> pd.DataFrame:
    # Legacy helper kept for backwards compatibility; not used by the main evaluation pipeline.
    return pd.DataFrame()
