"""
Paper-Style Ablation Experiments (Deng et al. WWW '23 Section 3 & 4):

- Experiment 1: Basic LLM few-shot (no CoT)
- Experiment 2: Few-shot + CoT reasoning
- Experiment 3: Few-shot + multiple reasoning paths (K=8, T=0.5)
- Experiment 4: Reasoning paths ablation (K in [1, 3, 5, 8, 16])
- Experiment 5: Demonstration ordering ablation (shuffled demonstrations)
- Experiment 6: Agreement consistency filtering (M in [5, 6, 7, 8])
- Experiment 7: Classification vs Regression distillation
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from src.llm.client import BaseLLMProvider, get_llm_provider
from src.llm.prompts import PromptManager
from src.llm.labeling import generate_reasoning_paths_for_post
from src.llm.aggregation import aggregate_paths_dataframe
from src.distillation.dataset import filter_by_consistency
from src.evaluation.metrics import compute_sentiment_metrics, save_metrics
from src.utils.common import OUTPUTS_DIR, RESULTS_DIR, log_experiment
from src.utils.logging import get_logger

logger = get_logger(__name__)


def run_reasoning_paths_ablation(
    eval_df: pd.DataFrame,
    provider: Optional[BaseLLMProvider] = None,
    paths_list: Optional[List[int]] = None,
    max_samples: int = 50,
) -> pd.DataFrame:
    """
    Experiment 4: Evaluate accuracy and Macro F1 across varying numbers of reasoning paths.
    Tests K in [1, 3, 5, 8, 16].
    """
    provider = provider or get_llm_provider()
    paths_list = paths_list or [1, 3, 5, 8, 16]
    prompt_mgr = PromptManager("sentiment_prompt")

    sub_df = eval_df.iloc[:max_samples].copy().reset_index(drop=True)
    text_col = "clean_text" if "clean_text" in sub_df.columns else "text"
    ground_truth = sub_df["sentiment_label"].tolist()

    max_k = max(paths_list)
    logger.info(f"Generating {max_k} paths for {len(sub_df)} posts to evaluate paths ablation...")

    # Pre-generate max_k paths
    all_records = []
    for _, row in sub_df.iterrows():
        p_id = str(row["post_id"])
        paths = generate_reasoning_paths_for_post(
            post_id=p_id,
            text=str(row[text_col]),
            ticker=str(row.get("ticker", "stock")),
            provider=provider,
            prompt_manager=prompt_mgr,
            num_paths=max_k,
            include_cot=True,
        )
        all_records.extend(paths)

    full_paths_df = pd.DataFrame(all_records)
    results = []

    for k in paths_list:
        sub_paths = full_paths_df[full_paths_df["path_id"] <= k].copy()
        agg = aggregate_paths_dataframe(sub_paths, original_df=sub_df)
        preds = agg["dominant_label"].tolist()
        m = compute_sentiment_metrics(ground_truth, preds)
        rec = {
            "num_paths": k,
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "weighted_f1": m["weighted_f1"],
            "mean_agreement": float(agg["agreement_score"].mean()),
        }
        results.append(rec)
        logger.info(f"Paths K={k:2d} -> Accuracy: {rec['accuracy']*100:.2f}%, Macro F1: {rec['macro_f1']:.4f}")

    res_df = pd.DataFrame(results)
    for target_dir in [OUTPUTS_DIR / "metrics", RESULTS_DIR / "metrics"]:
        target_dir.mkdir(parents=True, exist_ok=True)
        res_df.to_csv(target_dir / "ablation_reasoning_paths.csv", index=False)

    return res_df


def run_demonstration_order_ablation(
    eval_df: pd.DataFrame,
    provider: Optional[BaseLLMProvider] = None,
    seeds: Optional[List[int]] = None,
    max_samples: int = 30,
) -> pd.DataFrame:
    """
    Experiment 5: Evaluate sensitivity to demonstration ordering by shuffling prompts.
    """
    provider = provider or get_llm_provider()
    seeds = seeds or [42, 101, 2024, 7, 999]
    prompt_mgr = PromptManager("sentiment_prompt")

    sub_df = eval_df.iloc[:max_samples].copy().reset_index(drop=True)
    text_col = "clean_text" if "clean_text" in sub_df.columns else "text"
    ground_truth = sub_df["sentiment_label"].tolist()

    results = []
    for s in seeds:
        preds = []
        for _, row in sub_df.iterrows():
            prompt = prompt_mgr.build_prompt(
                post_text=str(row[text_col]),
                ticker=str(row.get("ticker", "stock")),
                include_cot=True,
                shuffle_seed=s,
            )
            resp = provider.generate(prompt)
            from src.llm.parser import parse_llm_response
            try:
                p = parse_llm_response(resp)
                preds.append(p["sentiment"])
            except Exception:
                preds.append("neutral")

        m = compute_sentiment_metrics(ground_truth, preds)
        results.append({
            "shuffle_seed": s,
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
        })

    res_df = pd.DataFrame(results)
    mean_acc = res_df["accuracy"].mean()
    std_acc = res_df["accuracy"].std()
    logger.info(f"Demo Order Ablation: Mean Acc = {mean_acc*100:.2f}% (+/- {std_acc*100:.2f}%)")

    for target_dir in [OUTPUTS_DIR / "metrics", RESULTS_DIR / "metrics"]:
        target_dir.mkdir(parents=True, exist_ok=True)
        res_df.to_csv(target_dir / "ablation_demo_order.csv", index=False)

    return res_df


def run_filtering_threshold_ablation(
    weak_labels_df: pd.DataFrame,
    thresholds: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Experiment 6: Consistency filtering retention across thresholds 5, 6, 7, 8 out of 8.
    """
    thresholds = thresholds or [5, 6, 7, 8]
    records = []
    for t in thresholds:
        _, stats = filter_by_consistency(weak_labels_df, threshold=t, total_paths=8)
        records.append({
            "threshold": stats["threshold"],
            "retained_count": stats["retained_count"],
            "retention_pct": stats["retention_pct"],
            "mean_agreement": stats["mean_agreement"],
        })
    res_df = pd.DataFrame(records)
    for target_dir in [OUTPUTS_DIR / "metrics", RESULTS_DIR / "metrics"]:
        target_dir.mkdir(parents=True, exist_ok=True)
        res_df.to_csv(target_dir / "ablation_consistency_filtering.csv", index=False)
    return res_df
