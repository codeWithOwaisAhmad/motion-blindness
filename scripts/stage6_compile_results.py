"""
Stage 6 — Results Compiler and Accuracy Calculator
Reads all model output CSVs from results/raw_outputs/
Computes per-category accuracy per model per condition.
Generates the master results table (Table 1 in your paper).

Expected input files (one per model per condition):
    results/raw_outputs/gemini_conditionA_results.csv
    results/raw_outputs/gemini_conditionB_results.csv
    results/raw_outputs/gpt4o_conditionA_results.csv
    results/raw_outputs/gpt4o_conditionB_results.csv
    results/raw_outputs/video_llava_results.csv       (from Kaggle notebook)
    results/raw_outputs/human_baseline/              (folder, compiled separately)

Output:
    results/compiled/master_results_table.csv
    results/compiled/secondary_analysis.csv

Usage:
    python scripts/stage6_compile_results.py
"""

import os
import pandas as pd
import numpy as np
from scipy import stats

# ── Config ────────────────────────────────────────────────────────────────────
RAW_OUTPUTS_DIR = os.path.join("results", "raw_outputs")
COMPILED_DIR = os.path.join("results", "compiled")
HUMAN_BASELINE_CSV = os.path.join(COMPILED_DIR, "human_baseline_compiled.csv")

RANDOM_BASELINE = 25.0  # 4-choice questions

# Model display names for the paper table
MODEL_LABELS = {
    "gemini_conditionA": "Gemini 2.0 Flash (RGB only)",
    "gemini_conditionB": "Gemini 2.0 Flash (RGB + Depth)",
    "gpt4o_conditionA":  "GPT-4o (RGB only)",
    "gpt4o_conditionB":  "GPT-4o (RGB + Depth)",
    "video_llava":       "Video-LLaVA (RGB only)",
}

CATEGORY_LABELS = {
    "q1": "Cat 1: 2D Direction",
    "q2": "Cat 2: 2D Speed",
    "q3": "Cat 3: Depth Direction",
    "q4": "Cat 4: Depth Rate",
}


def compute_accuracy(df: pd.DataFrame) -> dict:
    """Compute per-category accuracy from a model results DataFrame."""
    accuracy = {}
    for q in ["q1", "q2", "q3", "q4"]:
        col = f"{q}_is_correct"
        if col in df.columns:
            valid = df[col].dropna()
            accuracy[q] = round(valid.mean() * 100, 1)
        else:
            accuracy[q] = None
    accuracy["overall"] = round(
        np.mean([v for v in accuracy.values() if v is not None]), 1
    )
    return accuracy


def compute_confidence_interval(df: pd.DataFrame, q: str,
                                 confidence: float = 0.95) -> tuple:
    """
    Compute 95% confidence interval for accuracy on one category.
    Uses Wilson score interval — appropriate for proportions.
    """
    col = f"{q}_is_correct"
    if col not in df.columns:
        return (None, None)

    valid = df[col].dropna()
    n = len(valid)
    p = valid.mean()

    if n == 0:
        return (None, None)

    # Wilson score interval
    z = stats.norm.ppf((1 + confidence) / 2)
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    margin = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denominator

    lower = round(max(0, center - margin) * 100, 1)
    upper = round(min(1, center + margin) * 100, 1)
    return (lower, upper)


def load_model_results() -> dict:
    """
    Load all model result CSVs from raw_outputs directory.
    Returns dict: {model_key: DataFrame}
    """
    model_dfs = {}

    # Expected CSV files
    expected_files = {
        "gemini_conditionA": "gemini_conditionA_results.csv",
        "gemini_conditionB": "gemini_conditionB_results.csv",
        "gpt4o_conditionA":  "gpt4o_conditionA_results.csv",
        "gpt4o_conditionB":  "gpt4o_conditionB_results.csv",
        "video_llava":       "video_llava_results.csv",
    }

    for key, filename in expected_files.items():
        path = os.path.join(RAW_OUTPUTS_DIR, filename)
        if os.path.exists(path):
            df = pd.read_csv(path)
            model_dfs[key] = df
            print(f"  Loaded: {filename} ({len(df)} clips)")
        else:
            print(f"  Missing: {filename} — skipping")

    return model_dfs


def load_human_baseline() -> dict:
    """Load human baseline results if available."""
    if not os.path.exists(HUMAN_BASELINE_CSV):
        print(f"  Missing: human_baseline_compiled.csv — skipping")
        return {}

    df = pd.read_csv(HUMAN_BASELINE_CSV)
    human_results = {}

    for condition in ["A", "B"]:
        cond_df = df[df["condition"] == condition] if "condition" in df.columns else df
        if not cond_df.empty:
            key = f"human_condition{condition}"
            human_results[key] = cond_df
            label = "RGB only" if condition == "A" else "RGB + Depth"
            print(f"  Loaded: Human baseline condition {condition} — {label} ({len(cond_df)} responses)")

    return human_results


def build_master_table(model_dfs: dict, human_dfs: dict) -> pd.DataFrame:
    """
    Build the master results table matching Table 1 in your paper.

    Rows: Random baseline, Human (RGB only), Human (RGB+depth), each model
    Columns: Cat 1, Cat 2, Cat 3, Cat 4, Overall
    """
    rows = []

    # Row 1: Random baseline
    rows.append({
        "Model": "Random baseline",
        "Condition": "—",
        "Cat 1: 2D Direction": 25.0,
        "Cat 2: 2D Speed": 25.0,
        "Cat 3: Depth Direction": 25.0,
        "Cat 4: Depth Rate": 25.0,
        "Overall": 25.0,
    })

    # Human baseline rows
    human_label_map = {
        "human_conditionA": ("Human evaluators", "RGB only"),
        "human_conditionB": ("Human evaluators", "RGB + Depth"),
    }
    for key, (name, cond_label) in human_label_map.items():
        if key in human_dfs:
            acc = compute_accuracy(human_dfs[key])
            rows.append({
                "Model": name,
                "Condition": cond_label,
                "Cat 1: 2D Direction": acc.get("q1"),
                "Cat 2: 2D Speed": acc.get("q2"),
                "Cat 3: Depth Direction": acc.get("q3"),
                "Cat 4: Depth Rate": acc.get("q4"),
                "Overall": acc.get("overall"),
            })

    # Model rows
    for key, df in model_dfs.items():
        label = MODEL_LABELS.get(key, key)
        condition = "RGB only" if "conditionA" in key else "RGB + Depth"
        acc = compute_accuracy(df)
        rows.append({
            "Model": label,
            "Condition": condition,
            "Cat 1: 2D Direction": acc.get("q1"),
            "Cat 2: 2D Speed": acc.get("q2"),
            "Cat 3: Depth Direction": acc.get("q3"),
            "Cat 4: Depth Rate": acc.get("q4"),
            "Overall": acc.get("overall"),
        })

    return pd.DataFrame(rows)


def secondary_analysis(model_dfs: dict) -> pd.DataFrame:
    """
    Secondary analysis: accuracy by object type, speed, motion purity.
    This feeds Section 6 (Analysis) of your paper.
    """
    rows = []

    for key, df in model_dfs.items():
        label = MODEL_LABELS.get(key, key)

        # By depth motion type
        if "depth_motion" in df.columns:
            for motion_type in ["toward", "away", "stationary"]:
                subset = df[df["depth_motion"] == motion_type]
                if len(subset) == 0:
                    continue
                acc = compute_accuracy(subset)
                rows.append({
                    "model": label,
                    "analysis_type": "by_depth_motion",
                    "group": motion_type,
                    "n_clips": len(subset),
                    **{f"acc_{q}": acc.get(q) for q in ["q1", "q2", "q3", "q4"]},
                    "acc_overall": acc.get("overall"),
                })

        # By speed
        if "speed" in df.columns:
            for speed in ["slow", "medium", "fast"]:
                subset = df[df["speed"] == speed]
                if len(subset) == 0:
                    continue
                acc = compute_accuracy(subset)
                rows.append({
                    "model": label,
                    "analysis_type": "by_speed",
                    "group": speed,
                    "n_clips": len(subset),
                    **{f"acc_{q}": acc.get(q) for q in ["q1", "q2", "q3", "q4"]},
                    "acc_overall": acc.get("overall"),
                })

        # By scene type
        if "scene_type" in df.columns:
            for stype in ["object", "person"]:
                subset = df[df["scene_type"] == stype]
                if len(subset) == 0:
                    continue
                acc = compute_accuracy(subset)
                rows.append({
                    "model": label,
                    "analysis_type": "by_scene_type",
                    "group": stype,
                    "n_clips": len(subset),
                    **{f"acc_{q}": acc.get(q) for q in ["q1", "q2", "q3", "q4"]},
                    "acc_overall": acc.get("overall"),
                })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def compute_depth_gap(master_table: pd.DataFrame) -> None:
    """
    Print the key finding: gap between Cat 1-2 (2D) and Cat 3-4 (depth).
    This is the main result of your paper.
    """
    print("\n=== KEY FINDING: 2D Motion vs Depth-Axis Motion Gap ===")
    for _, row in master_table.iterrows():
        model = row["Model"]
        if model == "Random baseline":
            continue
        cat12_avg = np.mean([
            v for v in [row.get("Cat 1: 2D Direction"), row.get("Cat 2: 2D Speed")]
            if v is not None
        ])
        cat34_avg = np.mean([
            v for v in [row.get("Cat 3: Depth Direction"), row.get("Cat 4: Depth Rate")]
            if v is not None
        ])
        gap = cat12_avg - cat34_avg
        print(f"  {model[:45]:<45} 2D: {cat12_avg:.1f}%  Depth: {cat34_avg:.1f}%  Gap: {gap:.1f}pp")


def main():
    print("=" * 60)
    print("Stage 6 — Results Compiler")
    print("=" * 60)

    os.makedirs(COMPILED_DIR, exist_ok=True)

    print("\nLoading model results:")
    model_dfs = load_model_results()

    print("\nLoading human baseline:")
    human_dfs = load_human_baseline()

    if not model_dfs and not human_dfs:
        print("\n[Info] No results found yet. This is expected if you haven't")
        print("  run model evaluations yet. Creating empty master table template.")
        master_table = build_master_table({}, {})
    else:
        master_table = build_master_table(model_dfs, human_dfs)

    # Save master table
    master_path = os.path.join(COMPILED_DIR, "master_results_table.csv")
    master_table.to_csv(master_path, index=False)
    print(f"\n=== Master Results Table ===")
    print(master_table.to_string(index=False))

    # Key finding: depth gap
    compute_depth_gap(master_table)

    # Secondary analysis
    if model_dfs:
        secondary_df = secondary_analysis(model_dfs)
        if not secondary_df.empty:
            secondary_path = os.path.join(COMPILED_DIR, "secondary_analysis.csv")
            secondary_df.to_csv(secondary_path, index=False)
            print(f"\nSecondary analysis saved: {secondary_path}")

    print(f"\nMaster table saved: {master_path}")
    print(f"Next: run scripts/stage7_visualize.py")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
