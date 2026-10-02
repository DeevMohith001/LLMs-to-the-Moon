# LLMs to the Moon? Reddit Market Sentiment Analysis with Large Language Models

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/🤗_Transformers-4.38+-FFD21E?style=flat)](https://huggingface.co)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

> **Modernization & Reproduction Notice:**  
> This project is a **modernized reproduction** of the paper's methodology — NOT a bit-for-bit reproduction. The original paper used PaLM-540B. This implementation uses a modern configurable LLM as a practical replacement while faithfully preserving the paper's prompting, repeated-generation, aggregation, and distillation methodology.

---

## 1. Project Overview
This repository provides an open-source, reproducible end-to-end financial sentiment analysis system for social media (Reddit). Built from scratch, it implements the semi-supervised knowledge distillation methodology from the research paper:

> **Xiang Deng, Vasilisa Bashlovkina, Feng Han, Simon Baumgartner, Michael Bendersky.**  
> *"What do LLMs Know about Financial Markets? A Case Study on Reddit Market Sentiment Analysis"*  
> In *Companion Proceedings of the ACM Web Conference 2023 (WWW '23 Companion)*, pp. 326–330. [arXiv:2212.11311](https://arxiv.org/abs/2212.11311)

The codebase is strictly organized into two distinct parts:
- **[PAPER REPRODUCTION]:** In-context 6-shot Chain-of-Thought (CoT) prompting, stochastic multi-path generation ($K=8$ paths at $T=0.5$), majority voting, continuous soft agreement scores, consistency filtering ($\ge 5/8$), and student transformer knowledge distillation with Mean Squared Error (MSE) regression loss.
- **[PROJECT EXTENSION]:** Classical ML baselines (TF-IDF + Logistic Regression, SVM, Naive Bayes), lexicon baselines (VADER), modern FinBERT baselines (ProsusAI & HKUST), empirical market econometric return analysis (1d, 3d, 5d forward returns, correlations, event studies), and an interactive 8-page Streamlit research dashboard.

---

## 2. Research Questions
- **RQ1:** How accurately do lexicon rules, classical ML, domain transformers, and LLM prompting classify Reddit market sentiment?
- **RQ2:** Does financial domain pre-training (FinBERT) outperform general foundation LLMs on retail financial text?
- **RQ3:** Does Chain-of-Thought (CoT) financial reasoning improve label quality over zero-shot and standard few-shot prompting?
- **RQ4:** How does multi-path sampling ($K=1$ to $16$) and majority voting stabilize predictions?
- **RQ5:** Can weak soft labels from an LLM teacher effectively train a smaller, low-latency student model via regression loss?
- **RQ6:** Is aggregated retail Reddit sentiment predictive of subsequent equity returns or post-earnings reversals?

---

## 3. Paper Summary
Deng et al. addressed the scarcity of high-quality labeled financial social media posts by introducing a two-stage pipeline:
1. **Teacher LLM Weak Supervision:** Prompted PaLM-540B with 6 balanced demonstrations (2 bullish, 2 neutral, 2 bearish) accompanied by manual Chain-of-Thought reasoning. Repeated stochastic sampling ($K=8, T=0.5$) produced diverse reasoning paths, combined via majority voting for direct evaluation and soft agreement scores for distillation.
2. **Student Knowledge Distillation:** Filtered out inconsistent posts ($<5/8$ agreement), then trained a 102M parameter Charformer student using MSE regression loss on the continuous soft scores, achieving performance competitive with supervised models at a fraction of the inference latency.

---

## 4. Our Methodology & Modernizations
- **Teacher Modernization:** Replaced inaccessible PaLM-540B with a decoupled provider abstraction (`BaseLLMProvider`) supporting OpenAI (`gpt-4o-mini`), Google Gemini (`gemini-1.5-flash`), Anthropic Claude (`claude-3-5-sonnet`), Local Ollama (`llama3`), and a deterministic zero-cost `MockProvider` for continuous offline verification.
- **Student Modernization:** Replaced proprietary Charformer with modern, accessible sub-word transformer encoders (`distilbert-base-uncased`, 66M; `microsoft/deberta-v3-small`, 44M) while preserving the core regression distillation loss.
- **Strict Output Validation:** Pydantic/regex parsing enforcing normalized labels `positive`, `neutral`, and `negative`.
- **Reproducibility Guarantee:** Random seeds fixed to `42`, response caching, and comprehensive unit and end-to-end smoke tests.

---

## 5. System Architecture

```
project_root/
├── configs/
│   ├── data.yaml                    # Ingestion, cleaning & split configurations
│   ├── model.yaml                   # Teacher & student architectures
│   ├── training.yaml                # Distillation hyperparameters
│   ├── experiments.yaml             # Ablation matrices
│   └── prompts/
│       └── sentiment_prompt.yaml    # 6 demonstrations & CoT instructions
│
├── data/
│   ├── raw/                         # Raw Reddit parquet dumps
│   ├── interim/                     # Cleaned intermediate files
│   ├── processed/                   # Train/Val/Test splits & return datasets
│   └── external/                    # FiQA benchmark
│
├── notebooks/                       # 5 interactive analysis notebooks (01 to 05)
│
├── src/
│   ├── data/                        # Ingestion, preprocessing, ticker extraction, splits
│   ├── llm/                         # Providers (OpenAI, Gemini, Claude, Mock), prompts, parser, labeling, aggregation
│   ├── distillation/                # Datasets, student models, training, evaluation
│   ├── baselines/                   # TF-IDF (LogReg, SVM, NB), VADER, FinBERT
│   ├── experiments/                 # Ablations, benchmarks, error analysis
│   ├── finance/                     # Market data, signal aggregation, returns, analysis
│   └── utils/                       # Logging, seed pinning, YAML config loaders
│
├── dashboard/                       # 8-Page interactive Streamlit research app
├── outputs/                         # Models, metrics, figures, reports
├── tests/                           # Unit tests and end-to-end smoke test
└── docs/                            # Architectural & methodological documentation
```

---

## 6. Dataset Pipeline
- **Sources:** Verified Reddit submissions from `r/wallstreetbets`, `r/stocks`, `r/investing`, and `r/StockMarket`.
- **Text Cleaning:** HTML decoding, markdown stripping, URL removal, and financial emoji preservation (🚀 $\to$ `EMOJI_ROCKET_BULLISH`).
- **Entity Extraction:** High-precision cashtag extraction (`$AAPL`), ticker lookup with false-positive stopword filtering (`IT`, `FOR`, `ALL`), and canonical company name mapping.
- **Splits:** 70% Train, 15% Validation, 15% Test (stratified and chronological).

---

## 7. Installation

```bash
# Clone the repository
git clone https://github.com/your-username/LLM_FA.git
cd LLM_FA

# Create and activate a virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

---

## 8. Environment Variables
Create a local `.env` file from `.env.example`:

```bash
cp .env.example .env
```

Configure your provider:
```ini
LLM_PROVIDER=mock          # Options: openai | gemini | anthropic | local | mock
LLM_MODEL=mock-llm-v1      # Model identifier

# Optional API keys (not required if LLM_PROVIDER=mock)
OPENAI_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
```

---

## 9. Data Preparation
To run the full preprocessing pipeline:

```bash
python -m src.data.preprocessing
```
Generates `data/processed/train.parquet`, `val.parquet`, `test.parquet`.

---

## 10. LLM Labeling (8-Path Multi-Path Reasoning)
To run multi-path labeling with the teacher LLM:

```bash
python -c "
import pandas as pd
from src.utils.common import DATA_PROC
from src.llm.client import get_llm_provider
from src.llm.labeling import generate_teacher_weak_labels_dataset
from src.llm.aggregation import aggregate_paths_dataframe, save_weak_labels

train_df = pd.read_parquet(DATA_PROC / 'train.parquet')
provider = get_llm_provider()
paths = generate_teacher_weak_labels_dataset(train_df, provider=provider, num_paths=8, max_samples=100)
weak = aggregate_paths_dataframe(paths, original_df=train_df)
save_weak_labels(weak)
"
```
Produces `data/processed/weak_labels.csv` with `teacher_soft_score` in $[-1.0, 1.0]$.

---

## 11. Distillation & Student Training
Train the student model using the full paper pipeline:
1. Teacher LLM generates K=8 reasoning paths per post
2. Paths are aggregated via majority vote + continuous soft scores
3. Consistency filtering retains posts with ≥ 5/8 agreement
4. Student model trains with MSE regression loss on the **LLM-derived** soft scores

```bash
# Full end-to-end distillation pipeline (recommended):
python -c "
from src.models.distillation import run_distillation_experiments
metrics = run_distillation_experiments(max_train_samples=100, max_val_samples=30, epochs=3)
print(metrics)
"

# NOTE: src.models.distillation re-exports from the pipeline modules.
# The canonical distillation training logic lives in src/distillation/train.py

# Or step-by-step for inspection:
python -c "
from src.llm.client import get_llm_provider
from src.llm.labeling import generate_teacher_weak_labels_dataset
from src.llm.aggregation import aggregate_paths_dataframe, save_weak_labels
from src.distillation.dataset import filter_by_consistency
from src.distillation.train import train_regression_student
import pandas as pd
from src.utils.common import DATA_PROC

# Load data
train_df = pd.read_parquet(DATA_PROC / 'train.parquet')
val_df = pd.read_parquet(DATA_PROC / 'val.parquet')

# Step 1: Teacher LLM generates 8 reasoning paths per post
provider = get_llm_provider()
train_paths = generate_teacher_weak_labels_dataset(train_df, provider=provider, num_paths=8, max_samples=100)
val_paths = generate_teacher_weak_labels_dataset(val_df, provider=provider, num_paths=8, max_samples=30)

# Step 2: Aggregate paths → majority vote + soft scores
train_agg = aggregate_paths_dataframe(train_paths, original_df=train_df)
val_agg = aggregate_paths_dataframe(val_paths, original_df=val_df)

# Step 3: Consistency filter (≥ 5/8 agreement)
train_filtered, _ = filter_by_consistency(train_agg, threshold=5)
val_filtered, _ = filter_by_consistency(val_agg, threshold=5)

# CRITICAL: teacher_soft_score is derived from LLM predictions, NOT from sentiment_label
# It was computed by the aggregation module as: (pos_count - neg_count) / num_paths
print(f'Training soft score stats: mean={train_filtered[\"teacher_soft_score\"].mean():.4f}')

# Step 4: Train student with regression loss on LLM-derived soft scores
model, tok, hist = train_regression_student(train_filtered, val_filtered, epochs=3, batch_size=16)
"
```

> **⚠️ IMPORTANT:** The `teacher_soft_score` column is computed from the LLM's multi-path predictions (positive_count − negative_count) / K, NOT from the ground-truth `sentiment_label`. Deriving it from ground truth would defeat the purpose of knowledge distillation.

## 12. Benchmark Evaluation
Run the comprehensive benchmark across all models:

```bash
python -c "
import pandas as pd
from src.utils.common import DATA_PROC
from src.experiments.benchmark import run_comprehensive_benchmark

train_df = pd.read_parquet(DATA_PROC / 'train.parquet')
test_df = pd.read_parquet(DATA_PROC / 'test.parquet')
leaderboard = run_comprehensive_benchmark(train_df, test_df)
print(leaderboard)
"
```

---

## 13. Interactive Dashboard
Launch the 8-page academic research dashboard:

```bash
streamlit run dashboard/app.py
```
Open your browser at `http://localhost:8501`.

---

## 14. Financial Market Analysis (Project Extension)
Run the empirical financial return analysis:

```bash
python -m src.finance.run_financial_analysis
```

---

## 15. Paper-Style Ablation Experiments
Run ablation studies (Reasoning paths $K \in \{1, 3, 5, 8, 16\}$, demo order shuffling, consistency thresholds):

```bash
python -c "
import pandas as pd
from src.utils.common import DATA_PROC
from src.experiments.ablation import run_reasoning_paths_ablation, run_filtering_threshold_ablation

test_df = pd.read_parquet(DATA_PROC / 'test.parquet')
p_df = run_reasoning_paths_ablation(test_df, max_samples=30)
print(p_df)
"
```

---

## 16. Benchmark Results Summary

### 16.1 FiQA Cross-Dataset Benchmark (Verified Real Execution)
Evaluated on the official holdout test splits of the real FiQA-2018 challenge following the paper's filtering protocol (dropped score == 0, filtered multi-stock entries):

| Dataset | Model | Category | Accuracy | Macro F1 | Precision | Recall | Samples | Paper Reported Acc |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **FiQA-News** | VADER (Lexicon) | `[PROJECT EXTENSION]` | **50.00%** | 0.3273 | 0.3475 | 0.3457 | 60 | N/A |
| **FiQA-News** | FinBERT (ProsusAI) | `[PAPER REPRODUCTION]` | **75.00%** | 0.4996 | 0.5000 | 0.5017 | 60 | 81.1% |
| **FiQA-News** | FinBERT (HKUST) | `[PAPER REPRODUCTION]` | **61.67%** | 0.4052 | 0.4418 | 0.4254 | 60 | 75.7% |
| **FiQA-Post** | VADER (Lexicon) | `[PROJECT EXTENSION]` | **55.56%** | 0.3703 | 0.3703 | 0.3703 | 81 | N/A |
| **FiQA-Post** | FinBERT (ProsusAI) | `[PAPER REPRODUCTION]` | **71.60%** | 0.4769 | 0.4796 | 0.4778 | 81 | 73.5% |
| **FiQA-Post** | FinBERT (HKUST) | `[PAPER REPRODUCTION]` | **66.67%** | 0.4423 | 0.4476 | 0.4437 | 81 | 67.6% |

Run this evaluation via:
```bash
python -m src.experiments.benchmark
```

### 16.2 In-Domain Reddit Benchmark
Evaluates baselines, FinBERT, Teacher LLM, and Distilled Student on the verified real Reddit test split (`data/processed/test.parquet`, 1,590 posts):

| Model | Paradigm | Category | Status |
|:---|:---|:---|:---|
| VADER (Lexicon) | Lexicon | `[PROJECT EXTENSION]` | ✅ Evaluated |
| TF-IDF + Logistic Regression | Classical ML | `[PROJECT EXTENSION]` | ✅ Evaluated |
| TF-IDF + Linear SVM | Classical ML | `[PROJECT EXTENSION]` | ✅ Evaluated |
| TF-IDF + Naive Bayes | Classical ML | `[PROJECT EXTENSION]` | ✅ Evaluated |
| FinBERT (ProsusAI) | Pretrained FinBERT | `[PROJECT EXTENSION]` | ✅ Evaluated |
| FinBERT (HKUST) | Pretrained FinBERT | `[PROJECT EXTENSION]` | ✅ Evaluated |
| Teacher LLM (6-Shot + CoT + 8-Path Vote) | LLM In-Context | `[PAPER REPRODUCTION]` | ✅ Implemented (requires API key) |
| Distilled Student (Classification CE) | Distillation | `[PAPER REPRODUCTION]` | ✅ Implemented (requires trained model) |
| Distilled Student (Regression MSE) | Distillation | `[PAPER REPRODUCTION]` | ✅ Implemented (requires trained model) |

---

## 17. Empirical Financial Return Findings (Verified Real Execution)

Evaluated on **8,820 real Reddit submissions** across the top 10 liquid equities (AAPL, TSLA, MSFT, NVDA, GME, GOOGL, SPY, AMZN, META, AMD) aligned with active US trading days (2020–2025), yielding **3,083 joint observations**:

- **Forward Return Correlations (Pearson $r$):**
  - $1$-Day Horizon: $r = +0.0074$ ($p = 0.6823$, statistically insignificant)
  - $3$-Day Horizon: $r = -0.0080$ ($p = 0.6552$, statistically insignificant)
  - $5$-Day Horizon: $r = +0.0023$ ($p = 0.8966$, statistically insignificant)
- **Engagement-Weighted $1$-Day Return Correlation:** $r = +0.0068$
- **$1$-Day Return Spread (Bullish Spikes vs Bearish Spikes):** $-0.12\%$

> **Academic Conclusion:** Consistent with financial literature (e.g., Bradley et al., 2021), aggregated social media sentiment has no statistically significant predictive power for subsequent stock returns. This work is an exploratory econometric extension of Deng et al. and must not be used for investment decisions.

---

## 18. Limitations
- **CPU Inference:** Transformer evaluations ran on CPU. Sequence length was capped at 256–512 tokens.
- **Selection Bias:** Financial subreddits focus heavily on high-beta tech equities (NVDA, TSLA, GME).
- **Daily Aggregation:** Daily forward returns do not capture fast intraday liquidity dynamics.

---

## 19. Ethical Considerations
- **No Trading Recommendations:** This project is strictly academic.
- **Market Misinformation:** Social media is vulnerable to coordinated astroturfing and meme manipulation.
- **Model Hallucination:** LLM financial rationales require human audit before enterprise deployment.

---

## 20. Reproducibility & Testing
Run all 28 unit tests and the end-to-end smoke test:

```bash
pytest
```
Expected output:
```
============================= 28 passed in 32.27s =============================
```

Or via direct runner:
```bash
python tests/run_tests.py
```

Run via Docker:
```bash
docker-compose up --build
```

---

## 21. Citation & Academic Attributions
If you use this codebase or research, please cite:

```bibtex
@inproceedings{deng2023llms,
  title={What do LLMs Know about Financial Markets? A Case Study on Reddit Market Sentiment Analysis},
  author={Deng, Xiang and Bashlovkina, Vasilisa and Han, Feng and Baumgartner, Simon and Bendersky, Michael},
  booktitle={Companion Proceedings of the ACM Web Conference 2023 (WWW '23 Companion)},
  pages={326--330},
  year={2023},
  url={https://arxiv.org/abs/2212.11311}
}
```
See `CITATIONS.md` for full references to external models, benchmarks, and libraries.
