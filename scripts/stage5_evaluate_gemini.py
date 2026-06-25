"""
Stage 5a — Gemini Evaluation (gemini-2.0-flash)
Evaluates all clips under two conditions:
  Condition A — RGB frames only
  Condition B — RGB frames + colorized depth frames

Sends FRAMES_TO_SEND evenly-spaced frames per clip as images.
Uses the exact standardized question templates from your benchmark.

Cost: Free tier (gemini-2.0-flash) — no API credit needed.

Usage:
    python scripts/stage5_evaluate_gemini.py --condition A
    python scripts/stage5_evaluate_gemini.py --condition B
"""

import os
import sys
import time
import base64
import json
import argparse
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join("data", "dummy")
ANNOTATION_CSV = os.path.join("results", "compiled", "annotation_spreadsheet.csv")
OUTPUT_DIR = os.path.join("results", "raw_outputs")
FRAMES_TO_SEND = 8          # 8 evenly-spaced frames per clip
SLEEP_BETWEEN_CLIPS = 2.0   # seconds — stay within free tier rate limits
MAX_RETRIES = 3
RETRY_SLEEP = 10            # seconds before retry on rate limit

GEMINI_MODEL = "gemini-2.0-flash"

# System prompt for Gemini — locked, do not change
SYSTEM_PROMPT = """You are evaluating a video clip for a motion understanding benchmark.
You will be shown frames from a short video clip in order (first to last).
Answer each multiple-choice question based ONLY on what you observe in the frames.
Respond with ONLY the letter of your chosen answer (A, B, C, or D).
Do not explain your answer. Do not add any other text."""


def get_evenly_spaced_frames(frame_dir: str, n: int) -> list:
    """Get n evenly spaced frame paths from a clip folder."""
    frames = sorted([
        os.path.join(frame_dir, f)
        for f in os.listdir(frame_dir)
        if f.endswith(".jpg") or f.endswith(".png")
    ])
    if not frames:
        return []
    if len(frames) <= n:
        return frames
    indices = [int(i * (len(frames) - 1) / (n - 1)) for i in range(n)]
    return [frames[i] for i in indices]


def load_image_as_pil(path: str) -> Image.Image:
    """Load image as PIL Image for Gemini API."""
    return Image.open(path).convert("RGB")


def build_prompt_for_question(question_text: str, options_text: str) -> str:
    """Build the prompt for one question."""
    return f"{question_text}\n\nOptions:\n{options_text}\n\nYour answer (A/B/C/D only):"


def extract_answer(response_text: str) -> str:
    """
    Extract A/B/C/D from model response.
    Model should respond with just a letter, but we handle extra text defensively.
    """
    text = response_text.strip().upper()
    for char in text:
        if char in {"A", "B", "C", "D"}:
            return char
    return "INVALID"


def evaluate_clip_gemini(model, clip_row: dict, condition: str) -> dict:
    """
    Evaluate one clip with Gemini under the specified condition.

    Sends frames as PIL images followed by each question separately.
    Returns dict with answers for all 4 questions.
    """
    clip_id = clip_row["clip_id"]

    # Get frame paths
    rgb_dir = os.path.join(DATA_DIR, "rgb_frames", clip_id)
    depth_dir = os.path.join(DATA_DIR, "depth_frames", clip_id + "_colorized")

    if not os.path.exists(rgb_dir):
        return {"clip_id": clip_id, "error": "rgb_dir_missing"}

    rgb_frames = get_evenly_spaced_frames(rgb_dir, FRAMES_TO_SEND)
    if not rgb_frames:
        return {"clip_id": clip_id, "error": "no_rgb_frames"}

    # Load images
    pil_rgb_frames = [load_image_as_pil(p) for p in rgb_frames]

    if condition == "B" and os.path.exists(depth_dir):
        depth_frames = get_evenly_spaced_frames(depth_dir, FRAMES_TO_SEND)
        pil_depth_frames = [load_image_as_pil(p) for p in depth_frames]
    else:
        pil_depth_frames = []

    # Questions to ask
    questions = [
        ("q1", clip_row.get("q1_text", ""), clip_row.get("q1_options", ""), clip_row.get("q1_correct", "")),
        ("q2", clip_row.get("q2_text", ""), clip_row.get("q2_options", ""), clip_row.get("q2_correct", "")),
        ("q3", clip_row.get("q3_text", ""), clip_row.get("q3_options", ""), clip_row.get("q3_correct", "")),
        ("q4", clip_row.get("q4_text", ""), clip_row.get("q4_options", ""), clip_row.get("q4_correct", "")),
    ]

    result = {
        "clip_id": clip_id,
        "model": GEMINI_MODEL,
        "condition": condition,
        "scene_type": clip_row.get("scene_type", ""),
        "depth_motion": clip_row.get("verified_depth_motion", clip_row.get("depth_motion", "")),
        "speed": clip_row.get("speed", ""),
    }

    for qkey, qtext, qopts, qcorrect in questions:
        if not qtext:
            result[f"{qkey}_answer"] = "NO_QUESTION"
            result[f"{qkey}_correct_answer"] = qcorrect
            result[f"{qkey}_is_correct"] = False
            continue

        # Build content list: frames first, then question
        content = pil_rgb_frames.copy()
        if pil_depth_frames:
            content += pil_depth_frames
        content.append(build_prompt_for_question(qtext, qopts))

        # Send to Gemini with retry logic
        raw_response = ""
        answer = "INVALID"
        for attempt in range(MAX_RETRIES):
            try:
                response = model.generate_content(content)
                raw_response = response.text
                answer = extract_answer(raw_response)
                break
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "quota" in err_str.lower():
                    print(f"\n  [Rate limit] Waiting {RETRY_SLEEP}s before retry...")
                    time.sleep(RETRY_SLEEP)
                else:
                    print(f"\n  [Error] {clip_id} {qkey}: {err_str}")
                    answer = "ERROR"
                    break

        result[f"{qkey}_answer"] = answer
        result[f"{qkey}_correct_answer"] = qcorrect
        result[f"{qkey}_is_correct"] = (answer == qcorrect)
        result[f"{qkey}_raw_response"] = raw_response[:100]  # truncate for CSV

    return result


def main():
    parser = argparse.ArgumentParser(description="Gemini evaluation")
    parser.add_argument("--condition", choices=["A", "B"], default="A",
                        help="A = RGB only, B = RGB + Depth colorized")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only first N clips (for testing)")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Check API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_key_here":
        print("[Error] GEMINI_API_KEY not set in .env file")
        sys.exit(1)

    # Initialize Gemini
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=SYSTEM_PROMPT
    )
    print(f"Model: {GEMINI_MODEL} | Condition: {args.condition}")

    # Load annotation spreadsheet
    if not os.path.exists(ANNOTATION_CSV):
        print(f"[Error] Not found: {ANNOTATION_CSV}")
        print("  Run stage3_build_annotation_spreadsheet.py first.")
        sys.exit(1)

    df = pd.read_csv(ANNOTATION_CSV)
    if "include_in_benchmark" in df.columns:
        df = df[df["include_in_benchmark"] != "no"]

    if args.limit:
        df = df.head(args.limit)
        print(f"[Test mode] Evaluating first {args.limit} clips")

    print(f"Total clips to evaluate: {len(df)}")
    print(f"Estimated time: {len(df) * SLEEP_BETWEEN_CLIPS / 60:.1f} minutes\n")

    # Run evaluation
    results = []
    output_path = os.path.join(
        OUTPUT_DIR, f"gemini_condition{args.condition}_results.csv"
    )
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Gemini")):
        result = evaluate_clip_gemini(model, row.to_dict(), args.condition)
        results.append(result)

        # Save every 10 clips
        if (idx + 1) % 10 == 0:
            pd.DataFrame(results).to_csv(output_path, index=False)
            tqdm.write(f"  [Checkpoint] Saved {idx + 1} clips")

        time.sleep(SLEEP_BETWEEN_CLIPS)

    # Final save
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)

    # Quick accuracy summary
    print(f"\n=== Gemini Results (Condition {args.condition}) ===")
    for q in ["q1", "q2", "q3", "q4"]:
        col = f"{q}_is_correct"
        label = {"q1": "Cat 1 (2D Dir)", "q2": "Cat 2 (2D Speed)",
                 "q3": "Cat 3 (Depth Dir)", "q4": "Cat 4 (Depth Rate)"}[q]
        if col in results_df.columns:
            acc = results_df[col].mean() * 100
            print(f"  {label}: {acc:.1f}%")

    print(f"\nResults saved: {output_path}")
    print(f"Next: run scripts/stage5_evaluate_gpt4o.py")


if __name__ == "__main__":
    main()
