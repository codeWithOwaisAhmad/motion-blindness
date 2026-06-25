"""
Stage 4 — Human Baseline Collection
Shows RGB frames (and optionally colorized depth frames) from each clip
to a human evaluator. Records their answers for all 4 question categories.

Two conditions:
  Condition A — RGB only (show only color frames)
  Condition B — RGB + Depth (show color frames then depth colorized frames)

Run this once per evaluator. Target: 3-4 evaluators on a 30-clip sample.

Usage:
    python scripts/stage4_human_baseline.py --condition A --evaluator "friend_1"
    python scripts/stage4_human_baseline.py --condition B --evaluator "friend_1"
"""

import os
import cv2
import pandas as pd
import json
import argparse
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join("data", "dummy")
ANNOTATION_CSV = os.path.join("results", "compiled", "annotation_spreadsheet.csv")
OUTPUT_DIR = os.path.join("results", "raw_outputs", "human_baseline")
SAMPLE_SIZE = 30           # number of clips shown to each evaluator
FRAMES_TO_SHOW = 5         # show 5 evenly-spaced frames per clip (not all 150)
DISPLAY_TIME_MS = 600      # milliseconds per frame when playing clip
WINDOW_NAME = "Motion-Blindness — Human Baseline"

# Answer key display
ANSWER_OPTIONS = {
    "q1": ["A - Toward camera", "B - Away from camera", "C - To the left", "D - To the right"],
    "q2": ["A - Moving slowly", "B - Moving at medium speed", "C - Moving quickly", "D - Not moving"],
    "q3": ["A - Closer (toward)", "B - Farther (away)", "C - Same distance", "D - Cannot determine"],
    "q4": ["A - Distance decreased", "B - Distance increased", "C - Distance same", "D - Sideways only"],
}
VALID_KEYS = {"a": "A", "b": "B", "c": "C", "d": "D"}


def get_sample_clips(annotation_csv: str, n: int) -> pd.DataFrame:
    """
    Get a stratified sample of n clips — balanced across depth motion types.
    This ensures human baseline covers toward/away/stationary evenly.
    """
    df = pd.read_csv(annotation_csv)
    df = df[df.get("include_in_benchmark", pd.Series(["yes"] * len(df))) != "no"]

    motion_col = "verified_depth_motion" if "verified_depth_motion" in df.columns else "depth_motion"
    if motion_col not in df.columns:
        return df.sample(min(n, len(df)), random_state=42)

    # Stratified sample: equal clips per motion type
    groups = df.groupby(motion_col)
    per_group = max(1, n // len(groups))
    sampled = groups.apply(lambda g: g.sample(min(per_group, len(g)), random_state=42))
    sampled = sampled.reset_index(drop=True)

    # If still under n, top up randomly
    if len(sampled) < n:
        remaining = df[~df["clip_id"].isin(sampled["clip_id"])]
        extra = remaining.sample(min(n - len(sampled), len(remaining)), random_state=42)
        sampled = pd.concat([sampled, extra], ignore_index=True)

    return sampled.head(n)


def get_evenly_spaced_frames(frame_dir: str, n: int) -> list:
    """Get n evenly spaced frame paths from a clip folder."""
    frames = sorted([
        os.path.join(frame_dir, f)
        for f in os.listdir(frame_dir)
        if f.endswith(".jpg") or f.endswith(".png")
    ])
    if len(frames) <= n:
        return frames
    indices = [int(i * (len(frames) - 1) / (n - 1)) for i in range(n)]
    return [frames[i] for i in indices]


def show_frames_as_clip(frame_paths: list, window_name: str):
    """Display frames as a slideshow. Press any key to skip."""
    for path in frame_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        # Scale up for visibility
        img = cv2.resize(img, (960, 720))
        cv2.imshow(window_name, img)
        key = cv2.waitKey(DISPLAY_TIME_MS)
        if key == 27:  # ESC to skip
            break


def get_answer_from_keyboard(question_text: str, options: list,
                              window_name: str) -> str:
    """
    Display question on screen, wait for A/B/C/D keypress.
    Returns the answer letter.
    """
    # Create question display image
    img = create_question_image(question_text, options)
    cv2.imshow(window_name, img)

    while True:
        key = cv2.waitKey(0)
        char = chr(key & 0xFF).lower()
        if char in VALID_KEYS:
            return VALID_KEYS[char]
        if key == 27:  # ESC
            return "SKIP"


def create_question_image(question: str, options: list) -> "np.ndarray":
    """Create a black image with question text and options."""
    import numpy as np
    img = np.zeros((720, 960, 3), dtype=np.uint8)

    # Question text (wrap at 60 chars)
    words = question.split()
    lines = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 <= 55:
            current = current + " " + word if current else word
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)

    y = 80
    for line in lines:
        cv2.putText(img, line, (40, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (255, 255, 255), 2)
        y += 40

    y += 30
    colors = [(100, 200, 100), (100, 150, 255), (255, 200, 100), (200, 100, 200)]
    for i, opt in enumerate(options):
        cv2.putText(img, opt, (60, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, colors[i], 2)
        y += 50

    cv2.putText(img, "Press A / B / C / D", (40, 650),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1)
    return img


def run_evaluation(clips_df: pd.DataFrame, condition: str,
                   evaluator: str, output_dir: str):
    """
    Run the full human baseline evaluation for one evaluator.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []

    print(f"\nEvaluator: {evaluator} | Condition: {condition}")
    print(f"Clips to evaluate: {len(clips_df)}")
    print("Controls: A/B/C/D to answer | ESC to skip clip | Q to quit\n")
    input("Press ENTER to start...")

    for clip_idx, (_, row) in enumerate(clips_df.iterrows()):
        clip_id = row["clip_id"]
        print(f"[{clip_idx + 1}/{len(clips_df)}] {clip_id}")

        # Get frame paths
        rgb_dir = os.path.join(DATA_DIR, "rgb_frames", clip_id)
        depth_color_dir = os.path.join(DATA_DIR, "depth_frames", clip_id + "_colorized")

        if not os.path.exists(rgb_dir):
            print(f"  Skipping — RGB frames not found: {rgb_dir}")
            continue

        # Show clip
        rgb_frames = get_evenly_spaced_frames(rgb_dir, FRAMES_TO_SHOW)

        # Play RGB twice so evaluator can watch carefully
        for _ in range(2):
            show_frames_as_clip(rgb_frames, WINDOW_NAME)

        # Condition B: also show depth colorized frames
        if condition == "B" and os.path.exists(depth_color_dir):
            depth_frames = get_evenly_spaced_frames(depth_color_dir, FRAMES_TO_SHOW)
            for _ in range(2):
                show_frames_as_clip(depth_frames, WINDOW_NAME)

        # Collect answers for all 4 questions
        answers = {"clip_id": clip_id, "evaluator": evaluator, "condition": condition}
        questions = [
            ("q1", row.get("q1_text", ""), ANSWER_OPTIONS["q1"]),
            ("q2", row.get("q2_text", ""), ANSWER_OPTIONS["q2"]),
            ("q3", row.get("q3_text", ""), ANSWER_OPTIONS["q3"]),
            ("q4", row.get("q4_text", ""), ANSWER_OPTIONS["q4"]),
        ]

        for qkey, qtext, opts in questions:
            answer = get_answer_from_keyboard(qtext, opts, WINDOW_NAME)
            answers[f"{qkey}_answer"] = answer
            # Record correctness
            correct_col = f"{qkey}_correct"
            if correct_col in row:
                answers[f"{qkey}_correct"] = row[correct_col] == answer

        results.append(answers)

        # Auto-save every 5 clips
        if (clip_idx + 1) % 5 == 0:
            _save_results(results, evaluator, condition, output_dir, partial=True)
            print(f"  [Auto-saved at clip {clip_idx + 1}]")

    cv2.destroyAllWindows()
    _save_results(results, evaluator, condition, output_dir, partial=False)
    return results


def _save_results(results: list, evaluator: str, condition: str,
                  output_dir: str, partial: bool = False):
    """Save current results to CSV."""
    suffix = "_partial" if partial else ""
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"human_{evaluator}_condition{condition}_{ts}{suffix}.csv"
    path = os.path.join(output_dir, filename)
    pd.DataFrame(results).to_csv(path, index=False)


def compile_human_baseline(output_dir: str):
    """
    After all evaluators are done, compile results and compute accuracy per category.
    Call this separately after all evaluations are complete.
    """
    files = [f for f in os.listdir(output_dir)
             if f.endswith(".csv") and "partial" not in f]

    if not files:
        print("No completed evaluation files found.")
        return

    all_results = pd.concat([
        pd.read_csv(os.path.join(output_dir, f)) for f in files
    ], ignore_index=True)

    print("\n=== Human Baseline Summary ===")
    for condition in ["A", "B"]:
        cond_data = all_results[all_results["condition"] == condition]
        if cond_data.empty:
            continue
        print(f"\nCondition {condition} ({'RGB only' if condition == 'A' else 'RGB + Depth'}):")
        print(f"  Evaluators: {cond_data['evaluator'].nunique()}")
        print(f"  Total responses: {len(cond_data)}")
        for q in ["q1", "q2", "q3", "q4"]:
            col = f"{q}_correct"
            if col in cond_data.columns:
                acc = cond_data[col].mean() * 100
                label = {"q1": "Cat 1 (2D Dir)", "q2": "Cat 2 (2D Speed)",
                         "q3": "Cat 3 (Depth Dir)", "q4": "Cat 4 (Depth Rate)"}[q]
                print(f"  {label}: {acc:.1f}%")

    compiled_path = os.path.join("results", "compiled", "human_baseline_compiled.csv")
    all_results.to_csv(compiled_path, index=False)
    print(f"\nCompiled results saved: {compiled_path}")


def main():
    parser = argparse.ArgumentParser(description="Human baseline evaluation")
    parser.add_argument("--condition", choices=["A", "B"], default="A",
                        help="A = RGB only, B = RGB + Depth")
    parser.add_argument("--evaluator", type=str, default="evaluator_1",
                        help="Name/ID for this evaluator")
    parser.add_argument("--compile", action="store_true",
                        help="Compile all existing results instead of running evaluation")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    output_dir = os.path.join("results", "raw_outputs", "human_baseline")

    if args.compile:
        compile_human_baseline(output_dir)
        return

    if not os.path.exists(ANNOTATION_CSV):
        print(f"[Error] Annotation spreadsheet not found: {ANNOTATION_CSV}")
        print("  Run stage3_build_annotation_spreadsheet.py first.")
        return

    clips_df = get_sample_clips(ANNOTATION_CSV, SAMPLE_SIZE)
    print(f"Sample: {len(clips_df)} clips selected")

    run_evaluation(clips_df, args.condition, args.evaluator, output_dir)
    print(f"\nEvaluation complete. Run with --compile to aggregate all evaluators.")
    print(f"Next: run scripts/stage5_evaluate_gemini.py")


if __name__ == "__main__":
    main()
