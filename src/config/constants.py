"""
Centralized project constants for the
Early Sepsis Risk Prediction using Electronic Health Records and Explainable AI
project.
"""

# ==========================================================
# Project Metadata
# ==========================================================

PROJECT_NAME = (
    "Early Sepsis Risk Prediction using Electronic Health Records "
    "and Explainable AI"
)

PROJECT_VERSION = "1.0.0"

PROJECT_AUTHOR = "Meignana Vignesh"

PROJECT_DESCRIPTION = (
    "An AI-powered Clinical Decision Support System for early "
    "sepsis risk prediction using Electronic Health Records (EHR) "
    "and Explainable Artificial Intelligence."
)

PYTHON_VERSION = "3.12"

# ==========================================================
# Machine Learning Configuration
# ==========================================================

DEFAULT_MODEL = "xgboost"

SUPPORTED_MODELS = (
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "xgboost",
    "lightgbm",
)

DEFAULT_SEED = 42
DEFAULT_RANDOM_STATE = DEFAULT_SEED

DEFAULT_TEST_SIZE = 0.20
DEFAULT_VALIDATION_SIZE = 0.20

# ==========================================================
# Explainable AI
# ==========================================================

SUPPORTED_EXPLAINABILITY_METHODS = (
    "shap",
)

SHAP_OUTPUT_FOLDER_NAME = "shap"

# ==========================================================
# Database Configuration
# ==========================================================

DEFAULT_DATABASE_NAME = "early_sepsis_db"

# ==========================================================
# Supported File Formats
# ==========================================================

SUPPORTED_FILE_EXTENSIONS = (
    ".psv",
    ".csv",
    ".json",
)

SUPPORTED_DATA_FORMATS = (
    "psv",
    "csv",
    "json",
)

MODEL_FILE_EXTENSION = ".pkl"
REPORT_FILE_EXTENSION = ".html"

DEFAULT_ENCODING = "utf-8"

# ==========================================================
# Logging
# ==========================================================

DEFAULT_LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | %(name)s | "
    "%(filename)s:%(lineno)d | %(message)s"
)

DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ==========================================================
# Folder Names
# ==========================================================

DATA_FOLDER_NAME = "data"

RAW_DATA_FOLDER_NAME = "raw"

INTERIM_DATA_FOLDER_NAME = "interim"

PROCESSED_DATA_FOLDER_NAME = "processed"

EXTERNAL_DATA_FOLDER_NAME = "external"

MODEL_FOLDER_NAME = "models"

REPORT_FOLDER_NAME = "reports"

FIGURES_FOLDER_NAME = "figures"

METRICS_FOLDER_NAME = "metrics"

LOG_FOLDER_NAME = "logs"

ARTIFACTS_FOLDER_NAME = "artifacts"

NOTEBOOKS_FOLDER_NAME = "notebooks"

TESTS_FOLDER_NAME = "tests"