"""
Stage 3 — Annotation Spreadsheet Generator
Combines clip metadata + verified depth ground truth into one master CSV.
This is the annotation spreadsheet you fill during and after recording sessions.

Output columns (one row per clip):
    clip_id, filename, scene_type, object_type, description,
    scenario_label, speed, depth_start_mm, depth_end_mm, depth_change_mm,
    verified_depth_motion, rgb_visible,
    q1_text, q1_options, q1_correct,
    q2_text, q2_options, q2_correct,
    q3_text, q3_options, q3_correct,
    q4_text, q4_options, q4_correct

Usage:
    python scripts/stage3_build_annotation_spreadsheet.py
"""

import os
import json
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join("data", "dummy")
METADATA_JSON = os.path.join(DATA_DIR, "clip_metadata.json")
VERIFICATION_CSV = os.path.join("results", "compiled", "depth_verification.csv")
OUTPUT_CSV = os.path.join("results", "compiled", "annotation_spreadsheet.csv")

# ── Standardized question templates (from your paper design) ─────────────────
# These are locked — do not change after data collection starts

Q1_TEMPLATE = "In which direction does the {object} move in this clip?"
Q1_OPTIONS = "(A) Toward the camera | (B) Away from the camera | (C) To the left | (D) To the right"

Q2_TEMPLATE = "How would you describe the speed of the {object} in this clip?"
Q2_OPTIONS = "(A) Moving slowly | (B) Moving at medium speed | (C) Moving quickly | (D) Not moving"

Q3_TEMPLATE = "By the end of the clip, is the {object} closer to or farther from the camera than at the start?"
Q3_OPTIONS = "(A) Closer — it moved toward the camera | (B) Farther — it moved away from the camera | (C) Same distance — no depth change | (D) Cannot determine from this video"

Q4_TEMPLATE = "If you could measure the distance from the camera to the {object} at the start and end of the clip, which statement is correct?"
Q4_OPTIONS = "(A) Distance decreased — object approached | (B) Distance increased — object receded | (C) Distance stayed the same | (D) The object moved sideways only, depth unchanged"


def build_qa_pairs(row: dict) -> dict:
    """
    Build all 4 QA pairs for a clip using standardized templates.
    Correct answers come from verified depth ground truth (Categories 3 & 4)
    or from metadata (Categories 1 & 2).
    """
    obj = row.get("object_type", "object")
    # Use the description for person clips
    if obj == "person":
        label = "person"
    else:
        label = obj

    qa = {
        "q1_text": Q1_TEMPLATE.format(object=label),
        "q1_options": Q1_OPTIONS,
        "q1_correct": row.get("cat1_2d_direction", ""),

        "q2_text": Q2_TEMPLATE.format(object=label),
        "q2_options": Q2_OPTIONS,
        "q2_correct": row.get("cat2_2d_speed", ""),

        "q3_text": Q3_TEMPLATE.format(object=label),
        "q3_options": Q3_OPTIONS,
        "q3_correct": row.get("verified_cat3_answer", row.get("cat3_depth_direction", "")),

        "q4_text": Q4_TEMPLATE.format(object=label),
        "q4_options": Q4_OPTIONS,
        "q4_correct": row.get("verified_cat4_answer", row.get("cat4_depth_rate", "")),
    }
    return qa


def load_metadata(path: str) -> pd.DataFrame:
    """Load clip metadata JSON and flatten correct_answers into columns."""
    if not os.path.exists(path):
        print(f"[Warning] Metadata not found: {path}")
        return pd.DataFrame()

    with open(path) as f:
        metadata = json.load(f)

    rows = []
    for m in metadata:
        row = {k: v for k, v in m.items() if k != "correct_answers"}
        row.update(m.get("correct_answers", {}))
        rows.append(row)

    return pd.DataFrame(rows)


def load_verification(path: str) -> pd.DataFrame:
    """Load depth verification CSV."""
    if not os.path.exists(path):
        print(f"[Warning] Verification CSV not found: {path}")
        print("  Run stage2_verify_depth_groundtruth.py first.")
        return pd.DataFrame()
    return pd.read_csv(path)


def main():
    print("=" * 60)
    print("Stage 3 — Annotation Spreadsheet Generator")
    print("=" * 60)

    # Load both sources
    meta_df = load_metadata(METADATA_JSON)
    verif_df = load_verification(VERIFICATION_CSV)

    if meta_df.empty and verif_df.empty:
        print("[Error] No data sources available. Run stages 1 and 2 first.")
        return

    # Merge on clip_id
    if not meta_df.empty and not verif_df.empty:
        # Use verified depth answers (from Stage 2) for Categories 3 & 4
        # Use metadata answers for Categories 1 & 2
        merged = meta_df.merge(
            verif_df[["clip_id", "depth_start_mm", "depth_end_mm",
                       "depth_change_mm", "verified_depth_motion",
                       "verified_cat3_answer", "verified_cat4_answer"]],
            on="clip_id",
            how="left",
            suffixes=("_meta", "_verified")
        )
        # Prefer verified depth values over metadata depth values
        for col in ["depth_start_mm", "depth_end_mm", "depth_change_mm"]:
            col_v = col + "_verified"
            col_m = col + "_meta"
            if col_v in merged.columns:
                merged[col] = merged[col_v].combine_first(merged.get(col_m, pd.Series()))
        print(f"Merged {len(merged)} clips from metadata + verification")
    elif not meta_df.empty:
        merged = meta_df
        print(f"Using metadata only ({len(merged)} clips) — run stage 2 for verified depth GT")
    else:
        merged = verif_df
        print(f"Using verification only ({len(merged)} clips)")

    # Build QA pairs for each clip
    qa_rows = []
    for _, row in merged.iterrows():
        qa = build_qa_pairs(row.to_dict())
        qa_rows.append(qa)

    qa_df = pd.DataFrame(qa_rows)

    # Select and order final columns
    base_cols = [
        "clip_id", "scene_type", "object_type", "description",
        "scenario_label", "speed", "depth_start_mm", "depth_end_mm",
        "depth_change_mm", "verified_depth_motion", "depth_motion", "rgb_visible",
    ]
    available_base = [c for c in base_cols if c in merged.columns]
    final_df = pd.concat([
        merged[available_base].reset_index(drop=True),
        qa_df.reset_index(drop=True)
    ], axis=1)

    # Add empty columns for manual data entry during real recording
    # These are filled by hand during/after recording sessions
    if "ambiguity_flag" not in final_df.columns:
        final_df["ambiguity_flag"] = ""          # fill if question is ambiguous
    if "notes" not in final_df.columns:
        final_df["notes"] = ""                   # any notes during recording
    if "include_in_benchmark" not in final_df.columns:
        final_df["include_in_benchmark"] = "yes" # set to "no" to exclude a clip

    # Save
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    final_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nAnnotation spreadsheet saved: {OUTPUT_CSV}")
    print(f"  Total clips: {len(final_df)}")
    print(f"  Columns: {list(final_df.columns)}")
    print(f"\nScene type distribution:")
    if "scene_type" in final_df.columns:
        print(final_df["scene_type"].value_counts().to_string())
    print(f"\nDepth motion distribution:")
    motion_col = "verified_depth_motion" if "verified_depth_motion" in final_df.columns else "depth_motion"
    if motion_col in final_df.columns:
        print(final_df[motion_col].value_counts().to_string())
    print(f"\nNext: run scripts/stage4_human_baseline.py")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
