"""
Global configuration module for paths, parameters, and training settings.
"""

from pathlib import Path


class Config:
    """Central configuration registry for the recommendation pipeline."""

    # Project paths
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
    CACHE_DIR = PROJECT_ROOT / 'cache'
    OUTPUT_DIR = PROJECT_ROOT / 'output'
    SUBMISSIONS_DIR = PROJECT_ROOT / 'submissions'

    # Ensure required runtime directories exist
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    # Dataset split and sampling
    SAMPLE_RATIO = 1.0
    SPLIT_DATE = '2020-09-16'
    RANDOM_SEED = 42
    USE_RECENT_MONTHS = 3

    # Retrieval stage candidate parameters
    RECALL_TOP_K = 4000
    HOT_ITEMS_ALL = 40000
    HOT_CANDIDATES = 2500

    REPURCHASE_CANDIDATES = 150
    PRODUCT_CODE_CANDIDATES = 400
    COOCCURRENCE_CANDIDATES = 600
    SEQUENCE_CANDIDATES = 600
    AGE_GROUP_CANDIDATES = 600

    USER_CF_CANDIDATES = 500
    USER_CF_TOP_K = 5000
    USER_CF_SIM_ITEMS = 50

    SALES_CHANNEL_CANDIDATES = 500

    ITEMCF_HOT_ITEMS = 8000
    ITEMCF_CANDIDATES = 500
    ITEMCF_SIM_ITEMS = 30

    GRAPH_CANDIDATES = 500

    # Retrieval channel switches
    ENABLE_GRAPH = True
    ENABLE_ITEMCF = True
    ENABLE_COOCCURRENCE = True
    ENABLE_SEQUENCE = True
    ENABLE_AGE_GROUP = True
    ENABLE_USER_CF = True
    ENABLE_SALES_CHANNEL = True
    ENABLE_POSTAL_CODE = False
    ENABLE_BASKET_TO_BASKET = False

    # Negative sampling parameters
    NEG_POS_RATIO = 4
    HARD_NEG_RATIO = 0.7
    HARD_NEG_TOP_K = 50

    # LightGBM ranker hyperparameters
    LGB_N_ESTIMATORS = 350
    LGB_LEARNING_RATE = 0.04
    LGB_NUM_LEAVES = 63
    LGB_MAX_DEPTH = 6
    LGB_MIN_CHILD_SAMPLES = 50
    LGB_SUBSAMPLE = 0.8
    LGB_COLSAMPLE_BYTREE = 0.8
    LGB_REG_ALPHA = 1.0
    LGB_REG_LAMBDA = 2.0
    LGB_LABEL_GAIN = [0, 1, 3, 7, 15, 31]
    LGB_TRUNCATION_LEVEL = 20

    # LightGBM ensemble parameters
    ENSEMBLE_SEEDS = [42, 142, 242]
    ENSEMBLE_WEIGHT_RANKER = 0.6
    ENSEMBLE_WEIGHT_CLASSIFIER = 0.4

    # CatBoost hyperparameters
    CB_ITERATIONS = 450
    CB_LEARNING_RATE = 0.045
    CB_DEPTH = 6
    CB_THREAD_COUNT = 16

    # Reciprocal Rank Fusion (RRF) parameters
    RRF_K0 = 12.0
    WEIGHT_LGB = 0.50
    WEIGHT_CB = 0.50
    REP_BOOST = 0.35

    # Training and prediction configurations
    RANK_TRAIN_USERS = 150000
    VAL_RATIO = 0.15
    PREDICT_TOP_N = 40
    PREDICT_BATCH_SIZE = 5000

    @classmethod
    def print_config(cls):
        """Prints the runtime configuration parameters to console."""
        print("\n" + "=" * 65)
        print("Pipeline Configuration Summary")
        print("=" * 65)
        print(f"Sample Ratio:               {cls.SAMPLE_RATIO}")
        print(f"Split Cutoff Date:          {cls.SPLIT_DATE}")
        print(f"Active History Window:      {cls.USE_RECENT_MONTHS} months")
        print(f"Ranking Training Users:     {cls.RANK_TRAIN_USERS:,}")
        print(f"Predict Top-N Candidates:   Top-{cls.PREDICT_TOP_N}")
        print(f"RRF Fusion Parameters:      k0={cls.RRF_K0}, LGB={cls.WEIGHT_LGB}, CB={cls.WEIGHT_CB}")
        print("=" * 65 + "\n")