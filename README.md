# H&M Personalized Fashion Recommendations

An end-to-end multi-stage recommendation pipeline built for the Kaggle H&M Personalized Fashion Recommendations competition (31M+ transactions, 1.37M customers, 105K articles).

The system addresses strong seasonal transitions and extreme customer sparsity, achieving an offline validation score of 0.0267 MAP@12 via a heterogeneous ensemble of dual-head LightGBM, CatBoost, and metric-aligned Reciprocal Rank Fusion (RRF).



## Benchmark Results

All evaluations are conducted on a strictly held-out final week (2020-09-16 to 2020-09-22) using transactions from the preceding three months for training. All features follow strict time-lagging to prevent lookahead leakage.

| Model / Strategy | Key Architecture & Components | MAP@12 |
| :--- | :--- | :--- |
| Global Popularity Baseline | Weekly exponential decay transaction volume | 0.0210 |
| LightGBM Single Model | Dual-head (Ranker + Classifier) + Internal RRF + Temporal Priors | 0.0241 |
| CatBoost Single Model | Symmetric Oblivious Trees + Novel Sample Boost + Temporal Priors | 0.0248 |
| Linear Rank Average | 0.50 LightGBM + 0.50 CatBoost linear blending | 0.0254 |
| Final RRF Ensemble | Reciprocal Rank Fusion (k0=12) + Seasonal Trend Gating | 0.0267 |



## System Architecture

The recommendation pipeline comprises four sequential stages:

    [Raw Data: Transactions / Articles / Customers]
                           │
                           ▼
           [Data Preprocessing & Temporal Split]
      (3-month active window, strictly time-lagged validation)
                           │
                           ▼
           [Feature Engineering & Multimodal Extractors]
      (76 tabular features + ResNet visual & TF-IDF text embeddings)
                           │
                           ▼
           [Multi-Channel Retrieval & Seasonal Gating]
      (Repurchase ladder, product variants, co-occurrence, seasonal trends)
                           │
                           ▼
                 [Dual-Model Ranking Layer]
         ┌─────────────────┴─────────────────┐
         ▼                                   ▼
    [LightGBM Dual-Head Engine]     [CatBoost Classifier]
     (LambdaRank + Binary Logloss)   (Symmetric trees + novel-item weighting)
         └─────────────────┬─────────────────┘
                           │
                           ▼
        [Metric-Aligned Reciprocal Rank Fusion (k0=12)]
                           │
                           ▼
             [Top-12 Recommendations Output]



## Methodology & Engineering Design

### 1. Temporal Split & Anti-Leakage Formulation
* Validation Window: The final week (2020-09-16 to 2020-09-22) serves as the ground-truth target.
* Leakage Defense: All training samples are constructed via a rolling two-week lag window. Candidate features for target week W are computed strictly using transaction histories prior to week W.

### 2. Feature Engineering (76-Dimensional Space)
* Temporal Repurchase Dynamics: Half-life decay repurchase scores, repurchase interval gaps, and customer purchase cadence.
* Item Momentum and Seasonality: 1-day, 3-day, 7-day, 14-day, and 30-day transaction volumes; short-to-long sales burst ratios (7d/30d and 3d/7d); days elapsed since first and latest sale.
* Interaction Affinity: User-category and user-department affinity ratios; price z-score deviations against customer historical spend and category benchmarks.
* Multimodal Representations: Cosine similarities computed between user recent interaction profiles and candidate items using 32-dimensional visual embeddings (ResNet-50) and textual metadata embeddings (TF-IDF + TruncatedSVD).

### 3. Multi-Channel Candidate Retrieval
* Temporal Repurchase Ladder: Exponentially discounted repurchase history spanning Weeks 1 through 4 prior to the cutoff date.
* Cross-Color Sibling Mapping: Associated color variants retrieved via shared product_code.
* Seasonal Recency Masking: Prioritizes trending autumn inventory within the most recent 14-day sales window to handle the seasonal transition while pruning dormant catalog items.
* Multi-View Channels: In-basket co-occurrence, sequential transition patterns, demographic age-group popularity, User-User Collaborative Filtering, and Item-based Collaborative Filtering.

### 4. Dual-Model Ranking Architecture
* Dual-Head LightGBM: Combines a pairwise lambdarank head (optimizing top-order rank inversions) with a pointwise binary classification head (calibrating absolute purchase probability) to stabilize ranking variance.
* CatBoost Classifier: Utilizes symmetric oblivious trees with reweighted novel-item positive samples to encourage discovery of unseen catalog items.
* Reciprocal Rank Fusion (RRF): Replaces linear rank averaging with non-linear reciprocal rank fusion:
  Score(d) = sum_{m in {LGB, CB}} [ w_m / (k0 + rank_m(d)) ]
  A grid search established k0 = 12 as the optimal smoothing parameter, aligning with the hyperbolic discount curve (1/k) of the target MAP@12 metric.



## Repository Structure

    hm-recsys/
    ├── config/
    │   ├── __init__.py
    │   └── config.py                 # Central configurations, paths, and hyperparameters
    ├── data/
    │   └── raw/                      # Raw competition datasets
    │       ├── articles.csv
    │       ├── customers.csv
    │       └── transactions_train.csv
    ├── cache/                        # Cached embeddings and intermediate features
    ├── submissions/                  # Generated submission outputs
    ├── src/
    │   ├── __init__.py
    │   ├── pipeline.py               # End-to-end training and evaluation pipeline
    │   ├── eval_sandbox.py           # Offline grid search sandbox
    │   ├── generate_submission.py    # Chunked streaming submission generator (1.37M users)
    │   ├── data/
    │   │   ├── __init__.py
    │   │   └── prepare_data.py       # Data loading and time-based split
    │   ├── evaluate/
    │   │   ├── __init__.py
    │   │   └── metrics.py            # MAP@12 and recall evaluation metrics
    │   ├── features/
    │   │   ├── __init__.py
    │   │   ├── build_features.py     # Tabular feature engineering
    │   │   └── image_text_embedding.py # Multimodal embedding extractors
    │   ├── recall/
    │   │   ├── __init__.py
    │   │   ├── base.py               # Base retrieval interface
    │   │   ├── itemcf_recall.py      # Item-based collaborative filtering
    │   │   ├── hot_recall.py         # Global popularity recall
    │   │   ├── image_recall.py       # Visual similarity recall
    │   │   ├── embedding_recall.py   # Vectorized multimodal recall
    │   │   ├── award_recall.py       # Multi-channel candidate retrieval
    │   │   └── merger.py             # Weighted candidate merger
    │   └── rank/
    │       ├── __init__.py
    │       ├── ranker.py             # Base ranking class with hard negative mining
    │       └── embedding_ranker.py   # Dual-head LightGBM ranking engine
    ├── .gitignore
    ├── requirements.txt
    └── README.md



## Getting Started

### 1. Environment Installation
Clone the repository and install the dependencies:

    git clone https://github.com/your-username/hm-recsys.git
    cd hm-recsys
    pip install -r requirements.txt

### 2. Dataset Setup
Ensure raw competition files are placed in the data/raw/ directory:

    data/raw/
    ├── articles.csv
    ├── customers.csv
    ├── transactions_train.csv
    └── images/                       # Optional: extracted catalog images



## Execution Guide

### 1. Run the Full Training Pipeline
Executes data preprocessing, feature engineering, multimodal extraction, dual-model training, and full validation evaluation:

    python -m src.pipeline

### 2. Fast Offline Experimentation Sandbox
Runs rapid grid search over RRF smoothing parameters (k0) and blending weights using precomputed prediction caches:

    python -m src.eval_sandbox

### 3. Generate Kaggle Submission
Executes batch streaming inference for all 1,371,980 customers and outputs the formatted submission.csv:

    python -m src.generate_submission