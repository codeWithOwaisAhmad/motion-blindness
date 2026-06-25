"""
Stage 2 — Depth Ground Truth Verification
Reads depth values at frame 0 and final frame for every clip.
Auto-determines the correct answer for Categories 3 and 4.

With real RealSense data:
- Reads 16-bit .png depth frames extracted from .bag files
- Samples a region of interest (ROI) around the object
- Computes median depth in that ROI (median is more robust than single pixel)

With dummy data:
- Same exact code — dummy data was generated to match RealSense format exactly

Output:
- results/compiled/depth_verification.csv — one row per clip with verified GT

Usage:
    python scripts/stage2_verify_depth_groundtruth.py
"""

import os
import cv2
import numpy as np
import pandas as pd
import json
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join("data", "dummy")           # change to data/processed when real data arrives
DEPTH_FRAMES_DIR = os.path.join(DATA_DIR, "depth_frames")
OUTPUT_CSV = os.path.join("results", "compiled", "depth_verification.csv")
METADATA_JSON = os.path.join(DATA_DIR, "clip_metadata.json")

# ROI: wider center region to reliably capture moving object
ROI_X1, ROI_Y1 = 160, 120
ROI_X2, ROI_Y2 = 480, 360

# Stationary threshold — depth change under this = no depth motion
STATIONARY_THRESHOLD_MM = 50  # 5 centimeters


def read_depth_frame(frame_path: str) -> np.ndarray:
    """
    Read a 16-bit depth frame from disk.
    cv2.IMREAD_ANYDEPTH is required for 16-bit — without it, values are wrong.
    """
    depth = cv2.imread(frame_path, cv2.IMREAD_ANYDEPTH)
    if depth is None:
        raise FileNotFoundError(f"Cannot read depth frame: {frame_path}")
    return depth


def get_roi_median_depth(depth_frame: np.ndarray, roi=None) -> float:
    """
    Get median depth value in a region of interest.
    Median is used instead of mean — it ignores depth noise and invalid pixels.
    Excludes background pixels (>=2900mm) to isolate the object.
    Falls back to all non-zero pixels if no foreground found.

    Args:
        depth_frame: 16-bit numpy array (values in mm)
        roi: (x1, y1, x2, y2) tuple or None for default center region

    Returns:
        Median depth in millimeters (float), or -1.0 if invalid
    """
    if roi:
        x1, y1, x2, y2 = roi
    else:
        x1, y1, x2, y2 = ROI_X1, ROI_Y1, ROI_X2, ROI_Y2

    region = depth_frame[y1:y2, x1:x2]

    # Prefer foreground pixels (non-zero and not background)
    foreground = region[(region > 0) & (region < 2900)]
    if len(foreground) > 0:
        return float(np.median(foreground))

    # Fallback: any non-zero pixel
    valid = region[region > 0]
    if len(valid) > 0:
        return float(np.median(valid))

    return -1.0  # fully invalid frame


def determine_depth_answer(depth_start_mm: float,
                           depth_end_mm: float,
                           direction_2d: str = None) -> dict:
    """
    Determine correct answers for Categories 3 and 4 from depth measurements.
    """
    depth_change_mm = depth_end_mm - depth_start_mm

    if depth_change_mm < -STATIONARY_THRESHOLD_MM:
        cat3 = "A"
        cat4 = "A"
        motion_label = "toward"
    elif depth_change_mm > STATIONARY_THRESHOLD_MM:
        cat3 = "B"
        cat4 = "B"
        motion_label = "away"
    else:
        cat3 = "C"
        cat4 = "D" if direction_2d in ["left", "right"] else "C"
        motion_label = "stationary"

    return {
        "depth_change_mm": round(depth_change_mm, 1),
        "verified_depth_motion": motion_label,
        "verified_cat3_answer": cat3,
        "verified_cat4_answer": cat4,
    }


def get_clip_folders(depth_dir: str) -> list:
    """Return sorted list of clip folders (exclude colorized folders)."""
    folders = [
        f for f in os.listdir(depth_dir)
        if os.path.isdir(os.path.join(depth_dir, f))
        and not f.endswith("_colorized")
    ]
    return sorted(folders)


def verify_clip(clip_id: str, depth_dir: str) -> dict:
    """Run depth verification for a single clip."""
    clip_depth_dir = os.path.join(depth_dir, clip_id)
    frames = sorted([f for f in os.listdir(clip_depth_dir) if f.endswith(".png")])

    if len(frames) < 2:
        return {"clip_id": clip_id, "error": "insufficient frames"}

    depth_first = read_depth_frame(os.path.join(clip_depth_dir, frames[0]))
    depth_last = read_depth_frame(os.path.join(clip_depth_dir, frames[-1]))

    depth_start_mm = get_roi_median_depth(depth_first)
    depth_end_mm = get_roi_median_depth(depth_last)

    if depth_start_mm < 0 or depth_end_mm < 0:
        return {"clip_id": clip_id, "error": "invalid depth readings"}

    answers = determine_depth_answer(depth_start_mm, depth_end_mm)

    return {
        "clip_id": clip_id,
        "frame_first": frames[0],
        "frame_last": frames[-1],
        "depth_start_mm": round(depth_start_mm, 1),
        "depth_end_mm": round(depth_end_mm, 1),
        "total_frames": len(frames),
        **answers,
    }


def cross_check_with_metadata(results_df: pd.DataFrame, metadata_path: str) -> pd.DataFrame:
    """Cross-check verified answers against dummy metadata ground truth."""
    if not os.path.exists(metadata_path):
        return results_df

    with open(metadata_path) as f:
        metadata = json.load(f)

    meta_df = pd.DataFrame([
        {
            "clip_id": m["clip_id"],
            "expected_depth_motion": m["depth_motion"],
            "expected_cat3": m["correct_answers"]["cat3_depth_direction"],
            "expected_cat4": m["correct_answers"]["cat4_depth_rate"],
        }
        for m in metadata
    ])

    merged = results_df.merge(meta_df, on="clip_id", how="left")
    merged["cat3_match"] = merged["verified_cat3_answer"] == merged["expected_cat3"]
    merged["cat4_match"] = merged["verified_cat4_answer"] == merged["expected_cat4"]

    cat3_acc = merged["cat3_match"].mean() * 100
    cat4_acc = merged["cat4_match"].mean() * 100
    print(f"\n[Cross-check] Category 3 verification accuracy: {cat3_acc:.1f}%")
    print(f"[Cross-check] Category 4 verification accuracy: {cat4_acc:.1f}%")
    print("  (Should be ~100% — any mismatch = bug in verify script)")

    return merged


def main():
    print("=" * 60)
    print("Stage 2 — Depth Ground Truth Verification")
    print(f"Reading depth frames from: {DEPTH_FRAMES_DIR}")
    print("=" * 60)

    clip_folders = get_clip_folders(DEPTH_FRAMES_DIR)
    print(f"Found {len(clip_folders)} clips to verify\n")

    results = []
    errors = []

    for clip_id in tqdm(clip_folders, desc="Verifying clips"):
        result = verify_clip(clip_id, DEPTH_FRAMES_DIR)
        if "error" in result:
            errors.append(result)
        else:
            results.append(result)

    results_df = pd.DataFrame(results)
    results_df = cross_check_with_metadata(results_df, METADATA_JSON)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    results_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nVerification complete.")
    print(f"  Clips verified: {len(results)}")
    print(f"  Errors: {len(errors)}")
    print(f"  Output saved to: {OUTPUT_CSV}")

    if "verified_depth_motion" in results_df.columns:
        print(f"\nDepth motion distribution:")
        print(results_df["verified_depth_motion"].value_counts().to_string())

    if errors:
        print(f"\nErrors:")
        for e in errors:
            print(f"  {e}")

    print(f"\nNext: run scripts/stage3_build_annotation_spreadsheet.py")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
