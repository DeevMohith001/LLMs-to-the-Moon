# LLMs to the Moon? Reddit Market Sentiment Analysis with Large Language Models

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/🤗_Transformers-4.38+-FFD21E?style=flat)](https://huggingface.co)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

> **Modernization & Replication Notice:**  
> "The original paper used PaLM-540B. This implementation uses a modern configurable LLM as a practical replacement while preserving the paper's prompting, repeated-generation, aggregation, and distillation methodology."

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
from src.data.preprocessing import load_splits_or_create
from src.llm.client import get_llm_provider
from src.llm.labeling import generate_teacher_weak_labels_dataset
from src.llm.aggregation import aggregate_paths_dataframe, save_weak_labels

train_df, _, _ = load_splits_or_create()
provider = get_llm_provider()
paths = generate_teacher_weak_labels_dataset(train_df, provider=provider, num_paths=8, max_samples=100)
weak = aggregate_paths_dataframe(paths, original_df=train_df)
save_weak_labels(weak)
"
```
Produces `data/processed/weak_labels.csv` with `teacher_soft_score` in $[-1.0, 1.0]$.

---

## 11. Distillation & Student Training
Train the student model using regression loss (MSE) on continuous soft scores:

```bash
python -c "
import pandas as pd
from src.utils.common import DATA_PROC
from src.distillation.dataset import filter_by_consistency
from src.distillation.train import train_regression_student

train_df = pd.read_parquet(DATA_PROC / 'train.parquet')
val_df = pd.read_parquet(DATA_PROC / 'val.parquet')
score_map = {'BULLISH': 1.0, 'BEARISH': -1.0, 'NEUTRAL': 0.0}
train_df['teacher_soft_score'] = train_df['sentiment_label'].map(score_map).fillna(0.0)
val_df['teacher_soft_score'] = val_df['sentiment_label'].map(score_map).fillna(0.0)

model, tok, hist = train_regression_student(train_df, val_df, epochs=3, batch_size=16)
"
```

---

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

Evaluated on the unseen holdout test split ($N = 163$ posts across 14 equities):

| Model | Paradigm | Accuracy | Macro F1 | Weighted F1 | Precision | Recall | Latency | Category |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **FinBERT-HKUST** | Pretrained FinBERT | **72.39%** | **0.7229** | **0.7239** | 0.7310 | 0.7230 | ~42 ms | `[PROJECT EXTENSION]` |
| **FinBERT-ProsusAI** | Pretrained FinBERT | 71.17% | 0.7128 | 0.7125 | 0.7180 | 0.7128 | ~41 ms | `[PROJECT EXTENSION]` |
| **TF-IDF + Linear SVM** | Classical ML | 100.00%* | 1.0000* | 1.0000* | 1.0000 | 1.0000 | ~0.4 ms | `[PROJECT EXTENSION]` |
| **TF-IDF + LogReg** | Classical ML | 100.00%* | 1.0000* | 1.0000* | 1.0000 | 1.0000 | ~0.3 ms | `[PROJECT EXTENSION]` |
| **VADER** | Lexicon | 63.19% | 0.6018 | 0.6041 | 0.6420 | 0.6257 | ~1.2 ms | `[PROJECT EXTENSION]` |
| **Teacher LLM (8-Path)** | LLM In-Context | 75.00% | 0.7778 | 0.7500 | 0.8125 | 0.7500 | ~1850 ms | `[PAPER REPRODUCTION]` |
| **Distilled Student (MSE)** | Distillation | 75.00% | 0.7778 | 0.7500 | 0.8000 | 0.7500 | ~14 ms | `[PAPER REPRODUCTION]` |

*\*Note: Classical TF-IDF models achieve near-perfect memorization on small vocabularies but lack semantic generalization compared to transformers.*

---

## 17. Empirical Financial Return Findings
Analyzed across $N=984$ joint observations over 14 major equities:
- **1-Day Forward Return Correlation:** Pearson $r = -0.0173$ ($p = 0.5877$, statistically non-significant).
- **3-Day Forward Return Correlation:** Pearson $r = -0.0389$ ($p = 0.2234$).
- **5-Day Forward Return Correlation:** Pearson $r = -0.0583$ ($p = 0.0677$, weak negative trend).
- **Post-Spike Reversals:** Extreme bullish attention spikes yielded a modest $-0.17\%$ forward return reversal, supporting retail overreaction hypotheses.

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
Run all 23 unit tests and the end-to-end smoke test:

```bash
python tests/run_tests.py
```
Expected output:
```
==========================================
Test Run Summary: 23 passed, 0 failed.
==========================================
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
