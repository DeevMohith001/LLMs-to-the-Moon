"""
Data ingestion: download and load Reddit financial sentiment datasets.

Dataset strategy:
  1. Try loading from local data/raw/ if already downloaded.
  2. Download from HuggingFace (public Reddit WSB/financial datasets).
  3. Standardise into a common schema.

IMPORTANT: Synthetic data is NEVER used for research results.
If real data is unavailable, this module raises DataUnavailableError
instead of silently generating synthetic data.
"""

import os
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

from src.utils.common import (
    DATA_RAW, DATA_INTERIM, DATA_PROC,
    load_config, get_logger, set_seed,
)

logger = get_logger(__name__)


# ── Exceptions ────────────────────────────────────────────────
class DataUnavailableError(RuntimeError):
    """Raised when no real dataset can be loaded or downloaded.
    Prevents silent fallback to synthetic data in research pipelines."""
    pass


# ── Standard Schema ───────────────────────────────────────────
RAW_SCHEMA_COLS = [
    "post_id", "timestamp", "subreddit", "title", "body",
    "score", "num_comments", "source",
]

PROCESSED_SCHEMA_COLS = [
    "post_id", "timestamp", "subreddit", "text", "ticker", "company",
    "sentiment_label", "sentiment_score", "model", "prompt_version",
    "confidence", "rationale",
]


# ── Download helpers ──────────────────────────────────────────
def download_reddit_dataset() -> pd.DataFrame:
    """
    Download a public Reddit financial dataset from HuggingFace.

    IMPORTANT: This function does NOT fall back to synthetic data.
    If no real dataset can be downloaded, it raises DataUnavailableError.
    """
    logger.info("Attempting to download Reddit financial dataset...")

    errors = []

    # Try HuggingFace datasets
    try:
        from datasets import load_dataset

        # Try multiple dataset sources in order of preference
        hf_datasets = [
            ("SocialGrep/reddit-r-wallstreetbets-posts-daily", None),
        ]

        for ds_name, ds_config in hf_datasets:
            try:
                logger.info(f"Trying HuggingFace dataset: {ds_name}")
                ds = load_dataset(ds_name, ds_config, trust_remote_code=True)
                # Get the first available split
                split_name = list(ds.keys())[0]
                df = ds[split_name].to_pandas()
                logger.info(f"Loaded {len(df)} rows from {ds_name} ({split_name})")
                return _standardize_hf_reddit(df, ds_name)
            except Exception as e:
                logger.warning(f"Could not load {ds_name}: {e}")
                errors.append(f"{ds_name}: {e}")
                continue

    except ImportError:
        errors.append("'datasets' library not installed (pip install datasets)")
        logger.warning("datasets library not installed")

    # NO SYNTHETIC FALLBACK — raise error with actionable guidance
    raise DataUnavailableError(
        "No real Reddit dataset could be loaded.\n"
        "Attempted sources and errors:\n"
        + "\n".join(f"  - {e}" for e in errors) + "\n\n"
        "To proceed, you must do ONE of the following:\n"
        "  1. Install the 'datasets' library: pip install datasets\n"
        "  2. Place a real Reddit dataset at: data/raw/reddit_posts.parquet\n"
        "  3. For development/testing ONLY, use generate_development_dataset() explicitly\n"
        "     (results from synthetic data must NEVER be reported as research results)"
    )


def _standardize_hf_reddit(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Map a HuggingFace Reddit dataset to our standard schema."""
    logger.info(f"Standardizing dataset from {source}. Columns: {list(df.columns)}")

    result = pd.DataFrame()

    # Map common column names
    col_map = {
        "id": "post_id", "name": "post_id",
        "created_utc": "timestamp", "created": "timestamp",
        "timestamp": "timestamp",
        "subreddit": "subreddit",
        "title": "title",
        "selftext": "body", "body": "body", "text": "body",
        "score": "score", "ups": "score",
        "num_comments": "num_comments",
    }

    for src_col, dst_col in col_map.items():
        if src_col in df.columns and dst_col not in result.columns:
            result[dst_col] = df[src_col]

    # Ensure post_id exists
    if "post_id" not in result.columns:
        result["post_id"] = [f"post_{i}" for i in range(len(result))]

    # Parse timestamp
    if "timestamp" in result.columns:
        ts = result["timestamp"]
        if ts.dtype in ("int64", "float64"):
            result["timestamp"] = pd.to_datetime(ts, unit="s", utc=True, errors="coerce")
        else:
            result["timestamp"] = pd.to_datetime(ts, utc=True, errors="coerce")

    # Defaults
    if "subreddit" not in result.columns:
        result["subreddit"] = "wallstreetbets"
    if "title" not in result.columns:
        result["title"] = ""
    if "body" not in result.columns:
        result["body"] = ""
    if "score" not in result.columns:
        result["score"] = 0
    if "num_comments" not in result.columns:
        result["num_comments"] = 0

    result["source"] = source
    result = result[[c for c in RAW_SCHEMA_COLS if c in result.columns]]

    logger.info(f"Standardized to {len(result)} rows, columns: {list(result.columns)}")
    return result


def generate_development_dataset() -> pd.DataFrame:
    """
    Generate a small realistic development dataset for testing the pipeline.

    ⚠️  IMPORTANT: These are illustrative Reddit-style posts — NOT real data.
    This function is ONLY for:
      - Unit tests
      - Smoke tests
      - Development pipeline testing
    Results from this data must NEVER be reported as research results.

    The returned DataFrame includes a 'data_source' column set to 'SYNTHETIC'
    so downstream code can verify data provenance.
    """
    set_seed(42)
    rng = np.random.default_rng(42)

    logger.warning(
        "⚠️  GENERATING SYNTHETIC DEVELOPMENT DATASET. "
        "This data must NOT be used for research results."
    )

    # Realistic examples spanning sentiment types
    examples = [
        # Bullish
        {"title": "NVDA to the moon 🚀🚀🚀", "body": "Nvidia earnings are going to crush it. AI demand is insane. Loading up on calls.", "ticker": "NVDA", "sentiment": "BULLISH"},
        {"title": "AAPL looking strong", "body": "Apple's new product lineup is incredible. iPhone sales going to be huge this quarter. Buying more shares.", "ticker": "AAPL", "sentiment": "BULLISH"},
        {"title": "TSLA bull case", "body": "Tesla FSD is getting so much better. Robotaxi revenue is going to be massive. This stock is going to $500.", "ticker": "TSLA", "sentiment": "BULLISH"},
        {"title": "GME diamond hands 💎🙌", "body": "Not selling. The squeeze hasn't squozen. HODL to the moon.", "ticker": "GME", "sentiment": "BULLISH"},
        {"title": "MSFT cloud growth incredible", "body": "Azure revenue up 30%. GitHub Copilot adoption exploding. This is a $400 stock easy.", "ticker": "MSFT", "sentiment": "BULLISH"},
        {"title": "AMD about to pop", "body": "AMD gaining server market share like crazy. Their new chips crush Intel. Going long.", "ticker": "AMD", "sentiment": "BULLISH"},
        {"title": "AMZN earnings play", "body": "AWS margins improving, advertising growing 25%+. This is undervalued at current levels.", "ticker": "AMZN", "sentiment": "BULLISH"},
        {"title": "META VR play", "body": "Everyone sleeping on Meta's AI. Reels monetization improving. Revenue reacceleration incoming.", "ticker": "META", "sentiment": "BULLISH"},
        {"title": "GOOGL AI moat", "body": "Google has the best AI infrastructure. Gemini is incredible. Search isn't going anywhere. BUY.", "ticker": "GOOGL", "sentiment": "BULLISH"},
        {"title": "SPY calls printing", "body": "Market going higher. Fed done hiking. Soft landing confirmed. Buy every dip.", "ticker": "SPY", "sentiment": "BULLISH"},

        # Bearish
        {"title": "TSLA is overvalued garbage", "body": "PE ratio is insane. Margins declining. Competition catching up. This is a $100 stock.", "ticker": "TSLA", "sentiment": "BEARISH"},
        {"title": "AAPL peaked", "body": "iPhone growth is done. China sales declining. Services can't carry this valuation forever. Selling.", "ticker": "AAPL", "sentiment": "BEARISH"},
        {"title": "Market crash incoming", "body": "Yield curve inverted for 2 years. Recession is coming. Get out of stocks now. Buy puts on SPY.", "ticker": "SPY", "sentiment": "BEARISH"},
        {"title": "NVDA bubble", "body": "AI hype is just like the dot-com bubble. When it pops, NVDA goes back to $200. Shorting this.", "ticker": "NVDA", "sentiment": "BEARISH"},
        {"title": "GME bagholders unite", "body": "This company has no real plan. Revenue declining. The squeeze is over. Cut your losses.", "ticker": "GME", "sentiment": "BEARISH"},
        {"title": "COIN going to zero", "body": "Crypto winter isn't over. SEC crackdown coming. Revenue down 50%. Avoid this trash.", "ticker": "COIN", "sentiment": "BEARISH"},
        {"title": "BA quality disasters", "body": "Boeing can't make a safe plane. Orders getting cancelled. Debt is massive. Short this.", "ticker": "BA", "sentiment": "BEARISH"},
        {"title": "NFLX subscriber peak", "body": "Netflix is losing content. Password sharing crackdown won't help. Streaming wars over. Sell.", "ticker": "NFLX", "sentiment": "BEARISH"},
        {"title": "DIS losing money on streaming", "body": "Disney+ burning cash. Parks growth slowing. Bob Iger can't save this. Bearish.", "ticker": "DIS", "sentiment": "BEARISH"},
        {"title": "INTC dead company walking", "body": "Intel lost the foundry race. AMD and ARM eating their lunch. Dividend at risk. Puts.", "ticker": "INTC", "sentiment": "BEARISH"},

        # Neutral
        {"title": "AAPL earnings tomorrow", "body": "Anyone playing earnings? Could go either way. Waiting for the numbers before deciding.", "ticker": "AAPL", "sentiment": "NEUTRAL"},
        {"title": "What do you think about MSFT?", "body": "Looking at the chart. Seems fairly valued at this level. Not sure if I should buy or wait.", "ticker": "MSFT", "sentiment": "NEUTRAL"},
        {"title": "Market open thread", "body": "What's everyone watching today? I'm keeping an eye on tech earnings this week.", "ticker": "SPY", "sentiment": "NEUTRAL"},
        {"title": "AMZN stock split discussion", "body": "Does the stock split change anything fundamentally? Just more shares at lower price right?", "ticker": "AMZN", "sentiment": "NEUTRAL"},
        {"title": "Beginner question about options", "body": "Can someone explain how calls work? I see people talking about TSLA options. How risky is it?", "ticker": "TSLA", "sentiment": "NEUTRAL"},
        {"title": "DD: NVDA competitive landscape", "body": "Looking at the GPU market objectively. NVDA dominates but AMD is gaining. Here's the analysis...", "ticker": "NVDA", "sentiment": "NEUTRAL"},
        {"title": "Portfolio rebalancing", "body": "Moving from 80/20 stocks/bonds to 70/30. Not making a market call just reducing risk as I get older.", "ticker": "SPY", "sentiment": "NEUTRAL"},
        {"title": "META vs GOOGL comparison", "body": "Both seem fairly priced. META has better margins but GOOGL more diversified. Tough call.", "ticker": "META", "sentiment": "NEUTRAL"},
        {"title": "Fed meeting notes", "body": "Powell said data-dependent. Dot plot unchanged. Market didn't react much. Staying the course.", "ticker": "SPY", "sentiment": "NEUTRAL"},
        {"title": "Tax loss harvesting strategy", "body": "Selling some losers to offset gains. Not a reflection on the companies just tax strategy.", "ticker": "SPY", "sentiment": "NEUTRAL"},

        # Sarcastic / challenging
        {"title": "I'm financially ruined", "body": "YOLO'd my entire savings into GME weekly calls. Down 95%. Best investment ever 🤡", "ticker": "GME", "sentiment": "BEARISH"},
        {"title": "DD: Trust me bro", "body": "My cousin works at Apple and he said next iPhone will have AI. Buy calls. Source: trust me bro.", "ticker": "AAPL", "sentiment": "BULLISH"},
        {"title": "Loss porn: -$50k on TSLA puts", "body": "Elon tweeted and my puts went to zero. I'm never betting against this man again. Inverse me.", "ticker": "TSLA", "sentiment": "BEARISH"},
        {"title": "GUH moment", "body": "Accidentally bought 10x more NVDA calls than intended. Up 200% though so we good 😂🚀", "ticker": "NVDA", "sentiment": "BULLISH"},
        {"title": "Inverse Cramer working again", "body": "Cramer said buy COIN so I shorted it. Already up 15%. This strategy never fails.", "ticker": "COIN", "sentiment": "BEARISH"},
    ]

    # Templates for diverse text generation
    bullish_templates = [
        ("{ticker} to the moon! Just loaded up on {strike} calls expiring next month. Strong fundamentals and momentum.", "Earnings beat expected. Demand is through the roof. PT ${target}."),
        ("Why I'm extremely bullish on {ticker}: {reason}", "Position: {shares} shares and {calls} call contracts. Can't see this going anywhere but up."),
        ("{ticker} breaking out of resistance right now! 🚀🚀", "Huge volume spike at market open. Institutional buyers stepping in."),
        ("Deep dive DD into {ticker}: Massive moat and accelerating revenue", "Operating margins expanded by {pct}%. Cloud/hardware backlog is solid."),
        ("Never bet against {ticker}. Diamond hands 💎🙌", "Holding through the volatility. The long-term thesis has never looked better."),
        ("Analyst upgrade on {ticker} to Strong Buy with ${target} target", "Undervalued compared to sector peers. Free cash flow yield is attractive."),
        ("Bought the dip on {ticker} today at ${price}", "Great entry point. Rebound looks imminent as RSI shows oversold bounce."),
        ("Massive insider buying reported for {ticker}", "Executive team just purchased ${amount}k worth of stock on open market."),
    ]

    bearish_templates = [
        ("{ticker} is completely overvalued and ready to crash", "P/E multiple is disconnected from reality. Margins deteriorating rapidly. Shorting."),
        ("Why you should sell {ticker} before next earnings", "Supply chain disruptions and lowering guidance. Competition eating market share."),
        ("Loss porn: down {pct}% on {ticker} weekly calls 🤡", "Theta gang destroyed my portfolio. Should have bought puts instead."),
        ("{ticker} broke critical support level today", "High volume selloff. Next major support isn't until ${target}. Downside target in play."),
        ("Terrible guidance from {ticker} management", "Revenue growth slowed to {pct}%. Debt load is too high in this interest rate environment."),
        ("Puts on {ticker} printing money today 🩸📉", "Broke key 200 EMA. Bear flag breakdown confirmed. Added to my short position."),
        ("Insiders dumping shares of {ticker}", "CEO sold ${amount}M of stock this week. They know what's coming next quarter."),
        ("Macro headwinds will crush {ticker}'s forward earnings", "Consumer spending tightening, valuation is untenable at {multiple}x sales."),
    ]

    neutral_templates = [
        ("{ticker} upcoming earnings discussion thread", "What are your expectations for Q{quarter}? Implied volatility is pricing a {pct}% move."),
        ("Looking at {ticker} chart - consolidation pattern forming", "Trading in a tight range between ${low} and ${high}. Waiting for breakout confirmation."),
        ("Portfolio rebalancing: Thinking about trimmed position in {ticker}", "Not bearish, but looking to take some profits and reallocate to cash or index."),
        ("What is your fair value estimate for {ticker}?", "DCF model suggests around ${target}, but discount rate sensitivity is high. Opinions?"),
        ("Options flow unusual activity in {ticker}", "Large block trade seen in {strike} strikes. Could be a hedge or straddle position."),
        ("Comparing {ticker} vs sector competitors for long term hold", "Both have pros and cons. Valuation seems fair at current prices."),
        ("Fed interest rate decision impact on {ticker}", "Neutral reaction so far. Watching how Treasury yields settle before adding."),
        ("Covered call strategy on {ticker}", "Selling ${strike} calls against my position to generate yield during this sideways trend."),
    ]

    tickers = ["NVDA", "AAPL", "TSLA", "MSFT", "AMZN", "META", "GOOGL", "AMD", "GME", "SPY", "COIN", "PLTR", "INTC", "BA"]
    reasons = [
        "AI infrastructure demand is unmatched", "services revenue recurring cash cows",
        "market expansion and autonomous rollouts", "enterprise enterprise software lock-in",
        "logistics network efficiencies", "monetization of digital ad impressions",
        "datacenter market share capture", "custom silicon roadmap execution"
    ]

    all_posts = []
    base_date = datetime(2023, 1, 1)
    post_idx = 0

    categories = [
        ("BULLISH", bullish_templates),
        ("BEARISH", bearish_templates),
        ("NEUTRAL", neutral_templates),
    ]

    for sentiment_label, templates in categories:
        for tpl_title, tpl_body in templates:
            for ticker in tickers:
                # 3-5 unique posts per combination
                for _ in range(rng.integers(3, 6)):
                    post_idx += 1
                    days_offset = int(rng.integers(0, 700))
                    hour = int(rng.integers(0, 24))
                    minute = int(rng.integers(0, 60))
                    ts = base_date + pd.Timedelta(days=days_offset, hours=hour, minutes=minute)

                    # Dynamic format values
                    params = {
                        "ticker": f"${ticker}" if rng.random() > 0.4 else ticker,
                        "strike": int(rng.integers(50, 600)),
                        "target": int(rng.integers(50, 800)),
                        "price": int(rng.integers(30, 500)),
                        "low": int(rng.integers(80, 200)),
                        "high": int(rng.integers(210, 450)),
                        "pct": int(rng.integers(5, 75)),
                        "shares": int(rng.integers(50, 2000)),
                        "calls": int(rng.integers(2, 50)),
                        "amount": int(rng.integers(10, 500)),
                        "multiple": int(rng.integers(15, 60)),
                        "quarter": int(rng.integers(1, 5)),
                        "reason": rng.choice(reasons),
                    }

                    title = tpl_title.format(**params)
                    body = tpl_body.format(**params)

                    # Subreddit styling
                    sub = rng.choice(["wallstreetbets", "stocks", "investing", "StockMarket"])

                    all_posts.append({
                        "post_id": f"post_{post_idx:05d}",
                        "timestamp": ts,
                        "subreddit": sub,
                        "title": title,
                        "body": body,
                        "score": int(rng.integers(5, 8500)),
                        "num_comments": int(rng.integers(1, 650)),
                        "source": "SYNTHETIC_DEV_v1",
                        "data_source": "SYNTHETIC",
                        "_ticker": ticker,
                        "_sentiment": sentiment_label,
                    })

    df = pd.DataFrame(all_posts)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    logger.info(f"Generated {len(df)} diverse SYNTHETIC development posts")
    return df


# ── Main ingestion logic ──────────────────────────────────────
def load_raw_data(
    force_download: bool = False,
    allow_synthetic: bool = False,
) -> pd.DataFrame:
    """
    Load raw Reddit data. Uses cached version if available.

    Args:
        force_download: If True, re-download even if cached file exists.
        allow_synthetic: If True, fall back to synthetic data when real data
            is unavailable. This should ONLY be used for development and testing,
            NEVER for research results.

    Returns:
        DataFrame with raw Reddit posts.

    Raises:
        DataUnavailableError: If no real data can be loaded and allow_synthetic=False.
    """
    raw_file = DATA_RAW / "reddit_posts.parquet"

    if raw_file.exists() and not force_download:
        logger.info(f"Loading cached raw data from {raw_file}")
        df = pd.read_parquet(raw_file)
        logger.info(f"Loaded {len(df)} posts from cache")
        # Tag data source if not already tagged
        if "data_source" not in df.columns:
            source_val = df.get("source", pd.Series(["unknown"])).iloc[0]
            if "synthetic" in str(source_val).lower():
                df["data_source"] = "SYNTHETIC"
            else:
                df["data_source"] = "REAL"
        return df

    # Try downloading real data
    try:
        df = download_reddit_dataset()
        df["data_source"] = "REAL"
    except DataUnavailableError:
        if allow_synthetic:
            logger.warning(
                "⚠️  Real data unavailable. Using SYNTHETIC development dataset. "
                "Results from this data must NOT be reported as research results."
            )
            df = generate_development_dataset()
        else:
            raise

    # Save raw
    df.to_parquet(raw_file, index=False)
    logger.info(f"Saved {len(df)} posts to {raw_file}")

    return df


def assert_real_data(df: pd.DataFrame) -> None:
    """
    Guard function: raises DataUnavailableError if DataFrame contains
    synthetic data. Use this at the start of any research pipeline.
    """
    if "data_source" in df.columns:
        if (df["data_source"] == "SYNTHETIC").any():
            raise DataUnavailableError(
                "This DataFrame contains SYNTHETIC data. "
                "Research results cannot be generated from synthetic data. "
                "Please provide a real Reddit dataset."
            )
    if "source" in df.columns:
        sources = df["source"].unique()
        if any("synthetic" in str(s).lower() for s in sources):
            raise DataUnavailableError(
                f"This DataFrame was generated from synthetic source(s): {sources}. "
                "Research results cannot be generated from synthetic data."
            )


def load_fiqa_dataset() -> dict:
    """
    Load the FiQA sentiment dataset for cross-dataset evaluation.
    Returns dict with 'train', 'validation', 'test' DataFrames.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("pauri32/fiqa-2018")
        splits = {}
        for split_name in ds:
            df = ds[split_name].to_pandas()
            # Standardize columns
            if "sentence" in df.columns:
                df = df.rename(columns={"sentence": "text"})
            if "score" in df.columns:
                # Convert continuous score to binary/ternary
                df["sentiment_label"] = df["score"].apply(
                    lambda x: "BULLISH" if x > 0 else ("BEARISH" if x < 0 else "NEUTRAL")
                )
            df["data_source"] = "REAL"
            splits[split_name] = df
            logger.info(f"FiQA {split_name}: {len(df)} samples")
        return splits
    except Exception as e:
        logger.warning(f"Could not load FiQA dataset: {e}")
        return {}


if __name__ == "__main__":
    set_seed(42)
    # For development, allow synthetic fallback
    df = load_raw_data(force_download=True, allow_synthetic=True)
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Data source: {df['data_source'].unique()}")
    print(f"\nSample:\n{df.head()}")
    if "timestamp" in df.columns:
        print(f"\nDate range: {df['timestamp'].min()} -> {df['timestamp'].max()}")
    if "subreddit" in df.columns:
        print(f"\nSubreddits: {df['subreddit'].value_counts().to_dict()}")
