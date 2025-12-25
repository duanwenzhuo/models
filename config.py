import logging
import os
import logging
import os
from typing import Any, Dict

# API configuration
OPENAI_API_KEY = "sk-L1TUuj5pxxyBbg1aNiM2t5fQGdbhAOmJtgVufoXiek3KZLnJ"


OPENAI_BASE_URL = "https://api.openai.com/v1"
LLM_MODEL = "gpt-4o"

# Fail-fast switch: when True, any LLM unavailability should halt the workflow
# during development/testing rather than silently falling back.
REQUIRE_LLM = os.getenv("REQUIRE_LLM", "true").lower() in {"1", "true", "yes"}

RSCRIPT_PATH = r"C:\Program Files\R\R-4.4.1\bin\x64\Rscript.exe"

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")

# Logger
LOGGER_NAME = "integration"

# Integration hyperparameter presets
# NOTE: TuningAgent will select one preset per method per run. Defaults maintain
# prior behavior when presets are not used.
INTEGRATION_METHOD_PRESETS: Dict[str, Dict[str, Any]] = {
    "scvi": {
        "light": {
            "RNA": {
                "small": {"n_latent": 16, "max_epochs": 150, "batch_size": 128},
                "medium": {"n_latent": 16, "max_epochs": 100, "batch_size": 256},
                "large": {"n_latent": 10, "max_epochs": 70, "batch_size": 512},
            },
            "ATAC": {
                "small": {"n_latent": 20, "max_epochs": 150, "batch_size": 128},
                "medium": {"n_latent": 20, "max_epochs": 100, "batch_size": 256},
                "large": {"n_latent": 16, "max_epochs": 70, "batch_size": 512},
            },
        },
        "standard": {
            "RNA": {
                "small": {"n_latent": 20, "max_epochs": 350, "batch_size": 128},
                "medium": {"n_latent": 20, "max_epochs": 225, "batch_size": 256},
                "large": {"n_latent": 16, "max_epochs": 135, "batch_size": 512},
            },
            "ATAC": {
                "small": {"n_latent": 24, "max_epochs": 350, "batch_size": 128},
                "medium": {"n_latent": 24, "max_epochs": 225, "batch_size": 256},
                "large": {"n_latent": 20, "max_epochs": 135, "batch_size": 512},
            },
        },
        "high": {
            "RNA": {
                "small": {"n_latent": 32, "max_epochs": 500, "batch_size": 128},
                "medium": {"n_latent": 32, "max_epochs": 350, "batch_size": 256},
                "large": {"n_latent": 24, "max_epochs": 200, "batch_size": 512},
            },
            "ATAC": {
                "small": {"n_latent": 32, "max_epochs": 500, "batch_size": 128},
                "medium": {"n_latent": 32, "max_epochs": 350, "batch_size": 256},
                "large": {"n_latent": 24, "max_epochs": 200, "batch_size": 512},
            },
        },
    },
    "harmony": {
        "light": {"theta": 0.5, "lambda": 0.1, "nclust": 30},
        "standard": {"theta": 2.0, "lambda": 1.0, "nclust": 50},
        "high": {"theta": 4.0, "lambda": 2.0, "nclust": 75},
    },
    "mnn": {
        "light": {"n_pcs": 30, "k": 20, "cos_norm": True},
        "standard": {"n_pcs": 50, "k": 30, "cos_norm": True},
        "high": {"n_pcs": 75, "k": 50, "cos_norm": True},
    },
    "cca": {
        "light": {"nfeatures": 2000, "dims": list(range(1, 21)), "k_anchor": 5, "k_filter": 200},
        "standard": {"nfeatures": 3000, "dims": list(range(1, 31)), "k_anchor": 10, "k_filter": 250},
        "high": {"nfeatures": 4000, "dims": list(range(1, 41)), "k_anchor": 20, "k_filter": 500},
    },
    "liger": {
        "light": {
            "RNA": {"k": 20, "lambda": 5},
            "ATAC": {"k": 20, "lambda": 10},
        },
        "standard": {
            "RNA": {"k": 30, "lambda": 5},
            "ATAC": {"k": 30, "lambda": 10},
        },
        "high": {
            "RNA": {"k": 40, "lambda": 7},
            "ATAC": {"k": 40, "lambda": 15},
        },
    },
}

COMPUTE_BUDGET: Dict[str, Any] = {
    "max_presets_per_method": 3,
    "max_total_presets": 10,
}
# TODO: enforce compute budget once multi-preset benchmarking is enabled.


def configure_logging(dataset_name: str) -> str:
    """
    Quiet-by-default logging.

    - Console: only ERROR (so normal runs stay silent)
    - File: INFO (full trace for debugging)
    - Captures Python warnings into the log file
    """
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_path = os.path.join(LOGS_DIR, f"{dataset_name}.log")

    # ---- Root logger (captures 3rd-party libs too) ----
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    ))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)
    console_handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))

    root.addHandler(file_handler)
    root.addHandler(console_handler)

    # ---- Our project namespace ----
    proj = logging.getLogger(LOGGER_NAME)
    proj.setLevel(logging.INFO)
    proj.propagate = True
    proj.handlers.clear()

    # ---- Quiet noisy libraries ----
    noisy = [
        "httpx",
        "openai",
        "harmonypy",
        "umap",
        "pynndescent",
        "numba",
        "scanpy",
        "anndata",
        "scvi",
        "matplotlib",
        "langgraph",
    ]
    for name in noisy:
        logging.getLogger(name).setLevel(logging.WARNING)

    # ---- Capture warnings into log file (no console spam) ----
    logging.captureWarnings(True)
    logging.getLogger("py.warnings").setLevel(logging.WARNING)

    def _showwarning(message, category, filename, lineno, file=None, line=None):
        logging.getLogger(f"{LOGGER_NAME}.warnings").warning(
            "%s:%s: %s: %s", filename, lineno, category.__name__, message
        )

    import warnings as _warnings
    _warnings.showwarning = _showwarning
    _warnings.filterwarnings("default")

    return log_path


os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

#
