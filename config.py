import logging
import os
from typing import Any, Dict

# API configuration
OPENAI_API_KEY = 



OPENAI_BASE_URL = "https://api.openai.com/v1"
LLM_MODEL = "gpt-4o"

RSCRIPT_PATH = r"C:\Program Files\R\R-4.4.1\bin\x64\Rscript.exe"

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
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
    Configure the global logger for the current run and return the log-path.
    """
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_path = os.path.join(LOGS_DIR, f"{dataset_name}.log")

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_formatter = logging.Formatter("%(levelname)s | %(message)s")
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return log_path


os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

#