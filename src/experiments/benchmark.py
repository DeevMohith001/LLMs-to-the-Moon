"""
Unified Master Benchmark Runner for all Models on the Holdout Test Set.

Evaluates:
- Baselines: VADER, TF-IDF + LogReg, TF-IDF + SVM, TF-IDF + Naive Bayes
- Financial Pre-trained Transformers: FinBERT-ProsusAI, FinBERT-HKUST
- LLM Teacher Variants: 6-Shot + CoT + Multi-Path (K=8) Majority Vote
- Distilled Student Models: Classification Distillation, Regression Distillation

Generates master comparison leaderboard.

IMPORTANT: This benchmark requires REAL data. It will hard-fail on synthetic data.
Results from this benchmark are intended for research reporting.
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
from src.data.ingestion import assert_real_data, DataUnavailableError
from src.utils.common import DATA_PROC, OUTPUTS_DIR, RESULTS_DIR, MODELS_DIR
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _add_to_leaderboard(leaderboard: list, model_name: str, category: str,
                        metrics: dict, latency_ms: float):
    """Helper to add a model's results to the leaderboard."""
    leaderboard.append({
        "Model": model_name,
        "Category": category,
        "Accuracy": metrics["accuracy"],
        "Macro F1": metrics["macro_f1"],
        "Weighted F1": metrics["weighted_f1"],
        "Precision": metrics["precision"],
        "Recall": metrics["recall"],
        "Latency_ms": round(latency_ms, 2),
    })


def run_comprehensive_benchmark(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    include_finbert: bool = True,
    include_llm: bool = True,
    include_student: bool = True,
    max_test_samples: Optional[int] = None,
) -> pd.DataFrame:
    """
    Run evaluation across all models on test_df and compile master metrics table.

    IMPORTANT: This function hard-fails on synthetic data. Research results
    must ONLY be produced from real data.
    """
    # Hard-fail on synthetic data — do NOT catch and continue
    assert_real_data(train_df)
    assert_real_data(test_df)

    eval_test = test_df.iloc[:max_test_samples].copy() if max_test_samples else test_df.copy()
    text_col = "clean_text" if "clean_text" in eval_test.columns else "text"
    train_text_col = "clean_text" if "clean_text" in train_df.columns else "text"
    ground_truth = eval_test["sentiment_label"].tolist()

    leaderboard = []

    # ── 1. VADER Baseline [PROJECT EXTENSION] ──
    logger.info("Evaluating VADER Baseline...")
    t0 = time.time()
    vader = VADERBaseline()
    vader_preds = [vader.predict_label(t) for t in eval_test[text_col]]
    latency_vader = (time.time() - t0) / len(eval_test)
    m_vader = compute_sentiment_metrics(ground_truth, vader_preds)
    _add_to_leaderboard(leaderboard, "VADER (Lexicon)", "[PROJECT EXTENSION]",
                        m_vader, latency_vader * 1000)

    # ── 2. TF-IDF + Logistic Regression [PROJECT EXTENSION] ──
    logger.info("Evaluating TF-IDF + Logistic Regression...")
    t0 = time.time()
    lr = TFIDFLogisticRegressionBaseline()
    lr.fit(train_df[train_text_col].tolist(), train_df["sentiment_label"].tolist())
    lr_preds = lr.predict(eval_test[text_col].tolist())
    latency_lr = (time.time() - t0) / len(eval_test)
    m_lr = compute_sentiment_metrics(ground_truth, lr_preds)
    _add_to_leaderboard(leaderboard, "TF-IDF + Logistic Regression", "[PROJECT EXTENSION]",
                        m_lr, latency_lr * 1000)

    # ── 3. TF-IDF + Linear SVM [PROJECT EXTENSION] ──
    logger.info("Evaluating TF-IDF + Linear SVM...")
    t0 = time.time()
    svm = TFIDFLinearSVMBaseline()
    svm.fit(train_df[train_text_col].tolist(), train_df["sentiment_label"].tolist())
    svm_preds = svm.predict(eval_test[text_col].tolist())
    latency_svm = (time.time() - t0) / len(eval_test)
    m_svm = compute_sentiment_metrics(ground_truth, svm_preds)
    _add_to_leaderboard(leaderboard, "TF-IDF + Linear SVM", "[PROJECT EXTENSION]",
                        m_svm, latency_svm * 1000)

    # ── 4. TF-IDF + Naive Bayes [PROJECT EXTENSION] ──
    logger.info("Evaluating TF-IDF + Naive Bayes...")
    t0 = time.time()
    nb = TFIDFNaiveBayesBaseline()
    nb.fit(train_df[train_text_col].tolist(), train_df["sentiment_label"].tolist())
    nb_preds = nb.predict(eval_test[text_col].tolist())
    latency_nb = (time.time() - t0) / len(eval_test)
    m_nb = compute_sentiment_metrics(ground_truth, nb_preds)
    _add_to_leaderboard(leaderboard, "TF-IDF + Naive Bayes", "[PROJECT EXTENSION]",
                        m_nb, latency_nb * 1000)

    # ── 5. FinBERT ProsusAI [PROJECT EXTENSION] ──
    if include_finbert:
        from src.baselines.finbert import FinBERTBaseline

        for model_name, display_name in [
            ("ProsusAI/finbert", "FinBERT (ProsusAI)"),
            ("yiyanghkust/finbert-tone", "FinBERT (HKUST)"),
        ]:
            try:
                logger.info(f"Evaluating {display_name}...")
                t0 = time.time()
                fb = FinBERTBaseline(model_name)
                fb_preds, fb_confs, fb_probs = fb.predict_batch(eval_test[text_col].tolist())
                latency_fb = (time.time() - t0) / len(eval_test)
                m_fb = compute_sentiment_metrics(ground_truth, fb_preds)
                _add_to_leaderboard(leaderboard, display_name, "[PROJECT EXTENSION]",
                                    m_fb, latency_fb * 1000)
            except Exception as e:
                logger.warning(f"Skipping {display_name}: {e}")

    # ── 6. Teacher LLM (6-Shot + CoT + K=8 Majority Vote) [PAPER REPRODUCTION] ──
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
        _add_to_leaderboard(leaderboard, "Teacher LLM (6-Shot + CoT + 8-Path Vote)",
                            "[PAPER REPRODUCTION]", m_llm, latency_llm * 1000)

    # ── 7. Distilled Student Models [PAPER REPRODUCTION] ──
    if include_student:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        # 7a. Student Regression Distillation
        regression_dir = MODELS_DIR / "distilbert_student_regression"
        if regression_dir.exists():
            try:
                logger.info("Evaluating Distilled Student (Regression / MSE)...")
                t0 = time.time()
                tokenizer = AutoTokenizer.from_pretrained(str(regression_dir))
                model = AutoModelForSequenceClassification.from_pretrained(str(regression_dir))
                model.eval()

                # Predict raw scores
                raw_scores = []
                with torch.no_grad():
                    for i in range(0, len(eval_test), 16):
                        batch_texts = eval_test[text_col].iloc[i:i+16].tolist()
                        enc = tokenizer(batch_texts, truncation=True, padding=True,
                                       max_length=256, return_tensors="pt")
                        out = model(**enc)
                        raw_scores.extend(out.logits.squeeze(-1).cpu().numpy().tolist())

                # Use optimal threshold (default 0.2 if not tuned)
                theta = 0.2
                reg_preds = [
                    "BULLISH" if s > theta else ("BEARISH" if s < -theta else "NEUTRAL")
                    for s in raw_scores
                ]
                latency_reg = (time.time() - t0) / len(eval_test)
                m_reg = compute_sentiment_metrics(ground_truth, reg_preds)
                _add_to_leaderboard(leaderboard, "Distilled Student (Regression MSE)",
                                    "[PAPER REPRODUCTION]", m_reg, latency_reg * 1000)
            except Exception as e:
                logger.warning(f"Skipping student regression model: {e}")
        else:
            logger.info("Student regression model not found — skipping. "
                       f"Train it first with: python -m src.models.distillation")

        # 7b. Student Classification Distillation
        classification_dir = MODELS_DIR / "distilbert_student_classification"
        if classification_dir.exists():
            try:
                logger.info("Evaluating Distilled Student (Classification / CE)...")
                t0 = time.time()
                tokenizer = AutoTokenizer.from_pretrained(str(classification_dir))
                model = AutoModelForSequenceClassification.from_pretrained(str(classification_dir))
                model.eval()

                ID_TO_LABEL = {0: "BEARISH", 1: "NEUTRAL", 2: "BULLISH"}
                cls_preds = []
                with torch.no_grad():
                    for i in range(0, len(eval_test), 16):
                        batch_texts = eval_test[text_col].iloc[i:i+16].tolist()
                        enc = tokenizer(batch_texts, truncation=True, padding=True,
                                       max_length=256, return_tensors="pt")
                        out = model(**enc)
                        pred_ids = out.logits.argmax(dim=-1).cpu().numpy().tolist()
                        cls_preds.extend([ID_TO_LABEL[p] for p in pred_ids])

                latency_cls = (time.time() - t0) / len(eval_test)
                m_cls = compute_sentiment_metrics(ground_truth, cls_preds)
                _add_to_leaderboard(leaderboard, "Distilled Student (Classification CE)",
                                    "[PAPER REPRODUCTION]", m_cls, latency_cls * 1000)
            except Exception as e:
                logger.warning(f"Skipping student classification model: {e}")
        else:
            logger.info("Student classification model not found — skipping. "
                       "Train it first via the distillation pipeline.")

    # Compile leaderboard DataFrame
    lb_df = pd.DataFrame(leaderboard).sort_values("Macro F1", ascending=False).reset_index(drop=True)

    # Persist leaderboard
    for target_dir in [OUTPUTS_DIR / "metrics", RESULTS_DIR]:
        target_dir.mkdir(parents=True, exist_ok=True)
        lb_df.to_csv(target_dir / "sentiment_metrics.csv", index=False)
        lb_df.to_csv(target_dir / "benchmark_master.csv", index=False)

    logger.info("\n=== MASTER BENCHMARK LEADERBOARD ===\n" + lb_df.to_string())
    return lb_df
