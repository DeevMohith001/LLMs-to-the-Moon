"""
Unified Master Benchmark Runner for all Models on the Holdout Test Set.

Evaluates:
- Baselines: VADER, TF-IDF + LogReg, TF-IDF + SVM, TF-IDF + Naive Bayes
- Financial Pre-trained Transformers: FinBERT-ProsusAI, FinBERT-HKUST
- LLM Teacher Variants: Few-Shot, Few-Shot + CoT, Multi-Path (K=8) Majority Vote
- Distilled Student Models: Classification Distillation, Regression Distillation
Generates master comparison leaderboard.
"""

import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from src.baselines.vader import VADERBaseline
from src.baselines.tfidf_lr import TFIDFLogisticRegressionBaseline
from src.baselines.tfidf_svm import TFIDFLinearSVMBaseline
from src.baselines.tfidf_nb import TFIDFNaiveBayesBaseline
from src.evaluation.metrics import compute_sentiment_metrics, plot_confusion_matrix, save_metrics
from src.utils.common import DATA_PROC, OUTPUTS_DIR, RESULTS_DIR
from src.utils.logging import get_logger

logger = get_logger(__name__)


def run_comprehensive_benchmark(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    include_finbert: bool = False,
    include_llm: bool = True,
    max_test_samples: Optional[int] = None,
) -> pd.DataFrame:
    """
    Run evaluation across all models on test_df and compile master metrics table.
    """
    eval_test = test_df.iloc[:max_test_samples].copy() if max_test_samples else test_df.copy()
    text_col = "clean_text" if "clean_text" in eval_test.columns else "text"
    train_text_col = "clean_text" if "clean_text" in train_df.columns else "text"
    ground_truth = eval_test["sentiment_label"].tolist()

    leaderboard = []

    # 1. VADER Baseline
    logger.info("Evaluating VADER Baseline...")
    t0 = time.time()
    vader = VADERBaseline()
    vader_preds = [vader.predict_label(t) for t in eval_test[text_col]]
    latency_vader = (time.time() - t0) / len(eval_test)
    m_vader = compute_sentiment_metrics(ground_truth, vader_preds)
    leaderboard.append({
        "Model": "VADER (Lexicon)",
        "Category": "[PROJECT EXTENSION]",
        "Accuracy": m_vader["accuracy"],
        "Macro F1": m_vader["macro_f1"],
        "Weighted F1": m_vader["weighted_f1"],
        "Precision": m_vader["precision"],
        "Recall": m_vader["recall"],
        "Latency_ms": round(latency_vader * 1000, 2),
    })

    # 2. TF-IDF + Logistic Regression
    logger.info("Evaluating TF-IDF + Logistic Regression...")
    t0 = time.time()
    lr = TFIDFLogisticRegressionBaseline()
    lr.fit(train_df[train_text_col].tolist(), train_df["sentiment_label"].tolist())
    lr_preds = lr.predict(eval_test[text_col].tolist())
    latency_lr = (time.time() - t0) / len(eval_test)
    m_lr = compute_sentiment_metrics(ground_truth, lr_preds)
    leaderboard.append({
        "Model": "TF-IDF + Logistic Regression",
        "Category": "[PROJECT EXTENSION]",
        "Accuracy": m_lr["accuracy"],
        "Macro F1": m_lr["macro_f1"],
        "Weighted F1": m_lr["weighted_f1"],
        "Precision": m_lr["precision"],
        "Recall": m_lr["recall"],
        "Latency_ms": round(latency_lr * 1000, 2),
    })

    # 3. TF-IDF + Linear SVM
    logger.info("Evaluating TF-IDF + Linear SVM...")
    t0 = time.time()
    svm = TFIDFLinearSVMBaseline()
    svm.fit(train_df[train_text_col].tolist(), train_df["sentiment_label"].tolist())
    svm_preds = svm.predict(eval_test[text_col].tolist())
    latency_svm = (time.time() - t0) / len(eval_test)
    m_svm = compute_sentiment_metrics(ground_truth, svm_preds)
    leaderboard.append({
        "Model": "TF-IDF + Linear SVM",
        "Category": "[PROJECT EXTENSION]",
        "Accuracy": m_svm["accuracy"],
        "Macro F1": m_svm["macro_f1"],
        "Weighted F1": m_svm["weighted_f1"],
        "Precision": m_svm["precision"],
        "Recall": m_svm["recall"],
        "Latency_ms": round(latency_svm * 1000, 2),
    })

    # 4. TF-IDF + Naive Bayes
    logger.info("Evaluating TF-IDF + Naive Bayes...")
    t0 = time.time()
    nb = TFIDFNaiveBayesBaseline()
    nb.fit(train_df[train_text_col].tolist(), train_df["sentiment_label"].tolist())
    nb_preds = nb.predict(eval_test[text_col].tolist())
    latency_nb = (time.time() - t0) / len(eval_test)
    m_nb = compute_sentiment_metrics(ground_truth, nb_preds)
    leaderboard.append({
        "Model": "TF-IDF + Naive Bayes",
        "Category": "[PROJECT EXTENSION]",
        "Accuracy": m_nb["accuracy"],
        "Macro F1": m_nb["macro_f1"],
        "Weighted F1": m_nb["weighted_f1"],
        "Precision": m_nb["precision"],
        "Recall": m_nb["recall"],
        "Latency_ms": round(latency_nb * 1000, 2),
    })

    # 5. Teacher LLM (Few-Shot + CoT + K=8 Majority Vote)
    if include_llm:
        from src.llm.client import get_llm_provider
        from src.llm.labeling import generate_reasoning_paths_for_post
        from src.llm.aggregation import aggregate_paths_dataframe
        from src.llm.prompts import PromptManager

        logger.info("Evaluating Teacher LLM (6-shot + CoT + K=8 Majority Vote)...")
        t0 = time.time()
        provider = get_llm_provider()
        prompt_mgr = PromptManager("sentiment_prompt")

        llm_records = []
        for _, row in eval_test.iterrows():
            paths = generate_reasoning_paths_for_post(
                post_id=str(row["post_id"]),
                text=str(row[text_col]),
                ticker=str(row.get("ticker", "stock")),
                provider=provider,
                prompt_manager=prompt_mgr,
                num_paths=8,
                include_cot=True,
            )
            llm_records.extend(paths)

        agg = aggregate_paths_dataframe(pd.DataFrame(llm_records), original_df=eval_test)
        llm_preds = [
            "BULLISH" if l == "positive" else ("BEARISH" if l == "negative" else "NEUTRAL")
            for l in agg["dominant_label"]
        ]
        latency_llm = (time.time() - t0) / len(eval_test)
        m_llm = compute_sentiment_metrics(ground_truth, llm_preds)
        leaderboard.append({
            "Model": "Teacher LLM (6-Shot + CoT + 8 Paths Vote)",
            "Category": "[PAPER REPRODUCTION]",
            "Accuracy": m_llm["accuracy"],
            "Macro F1": m_llm["macro_f1"],
            "Weighted F1": m_llm["weighted_f1"],
            "Precision": m_llm["precision"],
            "Recall": m_llm["recall"],
            "Latency_ms": round(latency_llm * 1000, 2),
        })

    # Compile leaderboard DataFrame
    lb_df = pd.DataFrame(leaderboard).sort_values("Macro F1", ascending=False).reset_index(drop=True)

    # Persist leaderboard
    for target_dir in [OUTPUTS_DIR / "metrics", RESULTS_DIR]:
        target_dir.mkdir(parents=True, exist_ok=True)
        lb_df.to_csv(target_dir / "sentiment_metrics.csv", index=False)
        lb_df.to_csv(target_dir / "benchmark_master.csv", index=False)

    logger.info("\n=== MASTER BENCHMARK LEADERBOARD ===\n" + lb_df.to_string())
    return lb_df
