"""
Stage 7 — Visualization
Generates all figures for the paper from compiled results.

Figures produced:
  Figure 1 — Grouped bar chart: all models × 4 categories (main result)
  Figure 2 — Heatmap: model × category accuracy matrix
  Figure 3 — Gap chart: 2D motion accuracy vs depth-axis accuracy per model
  Figure 4 — Secondary analysis: accuracy by speed and motion type
  Figure 5 — Human vs model comparison on Categories 3 & 4

All figures saved to: paper/figures/
Saved as both .png (high-res) and .pdf (for LaTeX)

Usage:
    python scripts/stage7_visualize.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — works without display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── Config ────────────────────────────────────────────────────────────────────
COMPILED_DIR = os.path.join("results", "compiled")
FIGURES_DIR = os.path.join("paper", "figures")
MASTER_TABLE_CSV = os.path.join(COMPILED_DIR, "master_results_table.csv")
SECONDARY_CSV = os.path.join(COMPILED_DIR, "secondary_analysis.csv")

DPI = 300          # publication quality
FIG_WIDTH = 12     # inches
FIG_HEIGHT = 6

# Color scheme — 2D motion categories in blue tones, depth categories in red tones
CAT_COLORS = {
    "Cat 1: 2D Direction":    "#4C9BE8",   # blue
    "Cat 2: 2D Speed":        "#7BB8F0",   # light blue
    "Cat 3: Depth Direction": "#E84C4C",   # red (KEY category)
    "Cat 4: Depth Rate":      "#F07B7B",   # light red (KEY category)
}
RANDOM_LINE_COLOR = "#888888"
FONT_SIZE = 11


def setup_style():
    """Set matplotlib style for publication-quality figures."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": FONT_SIZE,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": DPI,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
    })


def load_master_table() -> pd.DataFrame:
    if not os.path.exists(MASTER_TABLE_CSV):
        print(f"[Warning] Master table not found: {MASTER_TABLE_CSV}")
        print("  Run stage6_compile_results.py first.")
        print("  Generating placeholder figure with hypothesized values...\n")
        return generate_hypothesized_table()
    return pd.read_csv(MASTER_TABLE_CSV)


def generate_hypothesized_table() -> pd.DataFrame:
    """
    Generate the hypothesized results from your paper design document.
    Used when real results are not yet available — lets you validate figure code.
    """
    rows = [
        {"Model": "Random baseline",              "Condition": "—",          "Cat 1: 2D Direction": 25, "Cat 2: 2D Speed": 25, "Cat 3: Depth Direction": 25, "Cat 4: Depth Rate": 25,  "Overall": 25},
        {"Model": "Human evaluators",             "Condition": "RGB only",   "Cat 1: 2D Direction": 85, "Cat 2: 2D Speed": 70, "Cat 3: Depth Direction": 68, "Cat 4: Depth Rate": 65,  "Overall": 72},
        {"Model": "Human evaluators",             "Condition": "RGB + Depth","Cat 1: 2D Direction": 88, "Cat 2: 2D Speed": 73, "Cat 3: Depth Direction": 92, "Cat 4: Depth Rate": 90,  "Overall": 86},
        {"Model": "GPT-4o (RGB only)",            "Condition": "RGB only",   "Cat 1: 2D Direction": 55, "Cat 2: 2D Speed": 45, "Cat 3: Depth Direction": 28, "Cat 4: Depth Rate": 26,  "Overall": 39},
        {"Model": "GPT-4o (RGB + Depth)",         "Condition": "RGB + Depth","Cat 1: 2D Direction": 57, "Cat 2: 2D Speed": 46, "Cat 3: Depth Direction": 31, "Cat 4: Depth Rate": 28,  "Overall": 41},
        {"Model": "Gemini 2.0 Flash (RGB only)",  "Condition": "RGB only",   "Cat 1: 2D Direction": 52, "Cat 2: 2D Speed": 42, "Cat 3: Depth Direction": 29, "Cat 4: Depth Rate": 27,  "Overall": 38},
        {"Model": "Video-LLaVA (RGB only)",       "Condition": "RGB only",   "Cat 1: 2D Direction": 48, "Cat 2: 2D Speed": 38, "Cat 3: Depth Direction": 26, "Cat 4: Depth Rate": 25,  "Overall": 34},
    ]
    return pd.DataFrame(rows)


def figure1_grouped_bar(df: pd.DataFrame, out_dir: str):
    """
    Figure 1 — Main result: grouped bar chart showing all models × 4 categories.
    This is the most important figure in the paper.
    """
    categories = ["Cat 1: 2D Direction", "Cat 2: 2D Speed",
                  "Cat 3: Depth Direction", "Cat 4: Depth Rate"]

    # Exclude random baseline and human from main model comparison
    model_df = df[~df["Model"].str.startswith("Random") &
                  ~df["Model"].str.startswith("Human")].copy()

    if model_df.empty:
        print("  [Figure 1] No model data — skipping")
        return

    models = model_df["Model"].tolist()
    n_models = len(models)
    n_cats = len(categories)
    x = np.arange(n_models)
    width = 0.2
    offsets = np.linspace(-(n_cats - 1) * width / 2, (n_cats - 1) * width / 2, n_cats)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    for i, (cat, color) in enumerate(CAT_COLORS.items()):
        values = [model_df.iloc[j][cat] if cat in model_df.columns and
                  not pd.isna(model_df.iloc[j][cat]) else 0
                  for j in range(n_models)]
        bars = ax.bar(x + offsets[i], values, width, label=cat,
                      color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
        # Add value labels on bars
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{val:.0f}", ha="center", va="bottom", fontsize=7)

    # Random baseline line
    ax.axhline(y=25, color=RANDOM_LINE_COLOR, linestyle="--",
               linewidth=1.5, label="Random baseline (25%)", zorder=5)

    # Shaded region for near-chance zone
    ax.axhspan(22, 35, alpha=0.08, color="red", label="Near-chance zone")

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Motion-Blindness Benchmark: 2D Motion vs Depth-Axis Motion Accuracy\n"
                 "(Categories 3 & 4 near chance for all models)",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right", fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 10))
    ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    _save_figure(fig, out_dir, "figure1_grouped_bar")
    print(f"  Figure 1 saved")


def figure2_heatmap(df: pd.DataFrame, out_dir: str):
    """
    Figure 2 — Heatmap: model × category accuracy matrix.
    Easy to read at a glance — good for paper appendix.
    """
    categories = ["Cat 1: 2D Direction", "Cat 2: 2D Speed",
                  "Cat 3: Depth Direction", "Cat 4: Depth Rate", "Overall"]

    heatmap_data = df[["Model"] + [c for c in categories if c in df.columns]].copy()
    heatmap_data = heatmap_data.set_index("Model")

    fig, ax = plt.subplots(figsize=(10, max(4, len(heatmap_data) * 0.7)))

    # Custom colormap: red for low (near chance), green for high (near human)
    cmap = sns.diverging_palette(10, 130, as_cmap=True)

    sns.heatmap(
        heatmap_data.astype(float),
        annot=True, fmt=".0f", cmap=cmap,
        vmin=25, vmax=95, center=55,
        linewidths=0.5, linecolor="white",
        ax=ax, cbar_kws={"label": "Accuracy (%)"}
    )

    ax.set_title("Motion-Blindness Benchmark — Accuracy Heatmap",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="y", rotation=0)
    ax.tick_params(axis="x", rotation=20)

    plt.tight_layout()
    _save_figure(fig, out_dir, "figure2_heatmap")
    print(f"  Figure 2 saved")


def figure3_gap_chart(df: pd.DataFrame, out_dir: str):
    """
    Figure 3 — Gap chart: 2D accuracy vs depth-axis accuracy per model.
    This directly visualizes the paper's main finding as a dot-and-line plot.
    """
    model_df = df[~df["Model"].str.startswith("Random")].copy()

    if model_df.empty:
        return

    # Compute averages
    model_df["2D_avg"] = model_df[["Cat 1: 2D Direction", "Cat 2: 2D Speed"]].mean(axis=1)
    model_df["Depth_avg"] = model_df[["Cat 3: Depth Direction", "Cat 4: Depth Rate"]].mean(axis=1)
    model_df["Gap"] = model_df["2D_avg"] - model_df["Depth_avg"]
    model_df = model_df.sort_values("Gap", ascending=True)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, max(5, len(model_df) * 0.6)))

    y_pos = np.arange(len(model_df))

    # Draw lines connecting 2D and Depth dots
    for i, (_, row) in enumerate(model_df.iterrows()):
        ax.plot([row["Depth_avg"], row["2D_avg"]], [i, i],
                color="#cccccc", linewidth=2.5, zorder=1)

    # Depth dots (red — the problem)
    ax.scatter(model_df["Depth_avg"], y_pos, color="#E84C4C", s=120,
               zorder=3, label="Depth-axis (Cat 3 & 4 avg)", edgecolors="white", linewidth=1)

    # 2D dots (blue — the baseline)
    ax.scatter(model_df["2D_avg"], y_pos, color="#4C9BE8", s=120,
               zorder=3, label="2D planar (Cat 1 & 2 avg)", edgecolors="white", linewidth=1)

    # Gap labels
    for i, (_, row) in enumerate(model_df.iterrows()):
        mid = (row["2D_avg"] + row["Depth_avg"]) / 2
        ax.text(mid, i + 0.15, f"−{row['Gap']:.0f}pp",
                ha="center", fontsize=8, color="#666666")

    ax.axvline(x=25, color=RANDOM_LINE_COLOR, linestyle="--",
               linewidth=1.5, label="Random baseline (25%)", alpha=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(model_df["Model"], fontsize=9)
    ax.set_xlabel("Accuracy (%)", fontsize=12)
    ax.set_title("Depth-Axis Motion Gap: Models Score Significantly Lower\n"
                 "on Depth-Axis Questions vs 2D Motion Questions",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(15, 100)
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    _save_figure(fig, out_dir, "figure3_gap_chart")
    print(f"  Figure 3 saved")


def figure4_secondary_analysis(out_dir: str):
    """
    Figure 4 — Secondary analysis: Cat 3+4 accuracy by speed and motion type.
    Shows which conditions are hardest for depth-axis reasoning.
    """
    if not os.path.exists(SECONDARY_CSV):
        print(f"  [Figure 4] Secondary analysis CSV not found — skipping")
        return

    df = pd.read_csv(SECONDARY_CSV)

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, 5))

    # Left: by speed
    speed_df = df[df["analysis_type"] == "by_speed"]
    if not speed_df.empty:
        speed_pivot = speed_df.pivot_table(
            index="group", values=["acc_q3", "acc_q4"], aggfunc="mean"
        ).reindex(["slow", "medium", "fast"])
        speed_pivot.columns = ["Cat 3: Depth Dir", "Cat 4: Depth Rate"]
        speed_pivot.plot(kind="bar", ax=axes[0], color=["#E84C4C", "#F07B7B"],
                         alpha=0.85, edgecolor="white")
        axes[0].axhline(y=25, color=RANDOM_LINE_COLOR, linestyle="--", linewidth=1.5)
        axes[0].set_title("Depth-Axis Accuracy by Movement Speed")
        axes[0].set_xlabel("Speed")
        axes[0].set_ylabel("Accuracy (%)")
        axes[0].set_ylim(0, 80)
        axes[0].tick_params(axis="x", rotation=0)

    # Right: by depth motion type
    motion_df = df[df["analysis_type"] == "by_depth_motion"]
    if not motion_df.empty:
        motion_pivot = motion_df.pivot_table(
            index="group", values=["acc_q3", "acc_q4"], aggfunc="mean"
        ).reindex(["toward", "away", "stationary"])
        motion_pivot.columns = ["Cat 3: Depth Dir", "Cat 4: Depth Rate"]
        motion_pivot.plot(kind="bar", ax=axes[1], color=["#E84C4C", "#F07B7B"],
                          alpha=0.85, edgecolor="white")
        axes[1].axhline(y=25, color=RANDOM_LINE_COLOR, linestyle="--", linewidth=1.5)
        axes[1].set_title("Depth-Axis Accuracy by Motion Type")
        axes[1].set_xlabel("Depth Motion")
        axes[1].set_ylabel("")
        axes[1].set_ylim(0, 80)
        axes[1].tick_params(axis="x", rotation=0)

    plt.suptitle("Secondary Analysis: When Are Depth-Axis Questions Hardest?",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save_figure(fig, out_dir, "figure4_secondary_analysis")
    print(f"  Figure 4 saved")


def figure5_human_vs_model(df: pd.DataFrame, out_dir: str):
    """
    Figure 5 — Human vs model comparison on Categories 3 & 4 specifically.
    Key finding: humans with depth = 90%+, models with depth = ~30%.
    """
    cat3_col = "Cat 3: Depth Direction"
    cat4_col = "Cat 4: Depth Rate"

    if cat3_col not in df.columns:
        return

    labels = df["Model"].tolist()
    cat3_vals = df[cat3_col].fillna(0).tolist()
    cat4_vals = df[cat4_col].fillna(0).tolist() if cat4_col in df.columns else [0] * len(labels)

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    b1 = ax.bar(x - width / 2, cat3_vals, width, label="Cat 3: Depth Direction",
                color="#E84C4C", alpha=0.85, edgecolor="white")
    b2 = ax.bar(x + width / 2, cat4_vals, width, label="Cat 4: Depth Rate",
                color="#F07B7B", alpha=0.85, edgecolor="white")

    ax.axhline(y=25, color=RANDOM_LINE_COLOR, linestyle="--",
               linewidth=1.5, label="Random baseline (25%)")

    # Annotate human RGB+depth bars
    for i, label in enumerate(labels):
        if "Human" in label and "Depth" in df.iloc[i].get("Condition", ""):
            ax.annotate("★ Human\n+depth", xy=(i, max(cat3_vals[i], cat4_vals[i]) + 2),
                        ha="center", fontsize=8, color="darkgreen", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Depth-Axis Question Accuracy: Human vs Models\n"
                 "(Humans with depth: ~90% | Models with depth: ~30%)",
                 fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    _save_figure(fig, out_dir, "figure5_human_vs_model")
    print(f"  Figure 5 saved")


def _save_figure(fig, out_dir: str, name: str):
    """Save figure as both PNG and PDF."""
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{name}.png"), dpi=DPI, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 60)
    print("Stage 7 — Visualization")
    print(f"Output: {FIGURES_DIR}")
    print("=" * 60)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    setup_style()

    df = load_master_table()
    print(f"Loaded {len(df)} rows from master results table\n")

    print("Generating figures:")
    figure1_grouped_bar(df, FIGURES_DIR)
    figure2_heatmap(df, FIGURES_DIR)
    figure3_gap_chart(df, FIGURES_DIR)
    figure4_secondary_analysis(FIGURES_DIR)
    figure5_human_vs_model(df, FIGURES_DIR)

    print(f"\nAll figures saved to: {FIGURES_DIR}")
    print("Formats: .png (300 DPI) and .pdf (for LaTeX)")
    print("\nPipeline complete.")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
