import os
import glob
import pandas as pd
import numpy as np

RESULTS_DIR = "results/raw_predictions"
MODELS = [
    ("SmolVLM-Instruct (2.2B)", "results_smolvlm_2b.csv"),
    ("Video-LLaVA-7B", "results_video_llava_7b.csv"),
    ("Qwen2.5-VL-3B-Instruct", "results_qwen2_5_vl_3b.csv"),
    ("Qwen2.5-VL-7B-Instruct", "results_qwen2_5_vl_7b.csv")
]

print("=" * 85)
print("RDMB-100 BENCHMARK METRIC VERIFICATION")
print("=" * 85)

summary_rows = []
for model_name, csv_file in MODELS:
    csv_path = os.path.join(RESULTS_DIR, csv_file)
    if not os.path.exists(csv_path):
        candidates = glob.glob(f"**/{csv_file}", recursive=True)
        if candidates:
            csv_path = candidates[0]
        else:
            print(f"⚠️ Missing: {csv_file}")
            continue

    df = pd.read_csv(csv_path)
    
    # Category Accuracies
    q1 = df["q1_acc"].mean() * 100
    q2 = df["q2_acc"].mean() * 100
    q3 = df["q3_acc"].mean() * 100
    q4 = df["q4_acc"].mean() * 100
    overall = (q1 + q2 + q3 + q4) / 4.0
    
    # Cat 3 (Depth Direction) Counts
    q3_preds = df["q3_pred"].replace("INVALID", "D")
    counts = q3_preds.value_counts().to_dict()
    a_cnt = counts.get("A", 0)
    b_cnt = counts.get("B", 0)
    c_cnt = counts.get("C", 0)
    d_cnt = counts.get("D", 0)
    
    beta_toward = (a_cnt / 100.0) - 0.25
    stationary_pct = (c_cnt / 100.0) * 100.0
    
    # Pure Away Canonical Recession
    pure_away_df = df[df["scenario"].astype(str).str.strip() == "pure_away"] if "scenario" in df.columns else pd.DataFrame()
    recession_acc = pure_away_df["q3_acc"].mean() * 100 if len(pure_away_df) > 0 else 0.0

    summary_rows.append({
        "Model": model_name,
        "Cat 1": f"{q1:.1f}%",
        "Cat 2": f"{q2:.1f}%",
        "Cat 3": f"{q3:.1f}%",
        "Cat 4": f"{q4:.1f}%",
        "Overall": f"{overall:.1f}%",
        "A (Tow)": a_cnt,
        "B (Rec)": b_cnt,
        "C (Stat)": c_cnt,
        "Beta": f"{beta_toward:+.2f}",
        "Stat %": f"{stationary_pct:.0f}%",
        "PureAway Acc": f"{recession_acc:.1f}%"
    })

res_df = pd.DataFrame(summary_rows)
print(res_df.to_string(index=False))
print("=" * 85)
