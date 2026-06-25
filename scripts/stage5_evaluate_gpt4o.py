"""
Stage 5b — GPT-4o Evaluation
Evaluates all clips under two conditions:
  Condition A — RGB frames only
  Condition B — RGB frames + colorized depth frames

Sends FRAMES_TO_SEND evenly-spaced frames per clip as base64 images.
Uses gpt-4o (vision-capable). Add credits to OpenAI account before running.

Estimated cost: ~$10-15 for 70 clips × 4 questions × 2 conditions.

Usage:
    python scripts/stage5_evaluate_gpt4o.py --condition A
    python scripts/stage5_evaluate_gpt4o.py --condition B
    python scripts/stage5_evaluate_gpt4o.py --condition A --limit 5  # test mode
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
from openai import OpenAI

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join("data", "dummy")
ANNOTATION_CSV = os.path.join("results", "compiled", "annotation_spreadsheet.csv")
OUTPUT_DIR = os.path.join("results", "raw_outputs")
FRAMES_TO_SEND = 8         # 8 evenly-spaced frames per clip
SLEEP_BETWEEN_CLIPS = 1.5  # seconds between clips
MAX_RETRIES = 3
RETRY_SLEEP = 15

GPT4O_MODEL = "gpt-4o"

SYSTEM_PROMPT = """You are evaluating a video clip for a motion understanding benchmark.
You will be shown frames from a short video clip in order (first to last).
Answer each multiple-choice question based ONLY on what you observe in the frames.
Respond with ONLY the letter of your chosen answer (A, B, C, or D).
Do not explain your answer. Do not add any other text."""


def get_evenly_spaced_frames(frame_dir: str, n: int) -> list:
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


def encode_image_base64(path: str) -> str:
    """Encode image file as base64 string for OpenAI API."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_messages(rgb_frame_paths: list, depth_frame_paths: list,
                   question_text: str, options_text: str) -> list:
    """
    Build the messages list for one GPT-4o request.
    Images come first, question text comes last.
    """
    content = []

    # Add RGB frames
    content.append({"type": "text", "text": "RGB video frames (in order, first to last):"})
    for path in rgb_frame_paths:
        b64 = encode_image_base64(path)
        ext = "jpeg" if path.endswith(".jpg") else "png"
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/{ext};base64,{b64}", "detail": "low"}
        })

    # Add depth frames if condition B
    if depth_frame_paths:
        content.append({"type": "text", "text": "Depth map frames (same clip, colorized — red = close, blue = far):"})
        for path in depth_frame_paths:
            b64 = encode_image_base64(path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}
            })

    # Add question
    content.append({
        "type": "text",
        "text": f"\nQuestion: {question_text}\n\nOptions:\n{options_text}\n\nAnswer (A/B/C/D only):"
    })

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content}
    ]


def extract_answer(response_text: str) -> str:
    text = response_text.strip().upper()
    for char in text:
        if char in {"A", "B", "C", "D"}:
            return char
    return "INVALID"


def evaluate_clip_gpt4o(client: OpenAI, clip_row: dict, condition: str) -> dict:
    clip_id = clip_row["clip_id"]

    rgb_dir = os.path.join(DATA_DIR, "rgb_frames", clip_id)
    depth_dir = os.path.join(DATA_DIR, "depth_frames", clip_id + "_colorized")

    if not os.path.exists(rgb_dir):
        return {"clip_id": clip_id, "error": "rgb_dir_missing"}

    rgb_frames = get_evenly_spaced_frames(rgb_dir, FRAMES_TO_SEND)
    if not rgb_frames:
        return {"clip_id": clip_id, "error": "no_rgb_frames"}

    depth_frames = []
    if condition == "B" and os.path.exists(depth_dir):
        depth_frames = get_evenly_spaced_frames(depth_dir, FRAMES_TO_SEND)

    questions = [
        ("q1", clip_row.get("q1_text", ""), clip_row.get("q1_options", ""), clip_row.get("q1_correct", "")),
        ("q2", clip_row.get("q2_text", ""), clip_row.get("q2_options", ""), clip_row.get("q2_correct", "")),
        ("q3", clip_row.get("q3_text", ""), clip_row.get("q3_options", ""), clip_row.get("q3_correct", "")),
        ("q4", clip_row.get("q4_text", ""), clip_row.get("q4_options", ""), clip_row.get("q4_correct", "")),
    ]

    result = {
        "clip_id": clip_id,
        "model": GPT4O_MODEL,
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

        messages = build_messages(rgb_frames, depth_frames, qtext, qopts)
        raw_response = ""
        answer = "INVALID"

        for attempt in range(MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model=GPT4O_MODEL,
                    messages=messages,
                    max_tokens=5,
                    temperature=0.0,  # deterministic — benchmark requires reproducibility
                )
                raw_response = response.choices[0].message.content or ""
                answer = extract_answer(raw_response)
                break
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "rate" in err_str.lower():
                    print(f"\n  [Rate limit] Waiting {RETRY_SLEEP}s...")
                    time.sleep(RETRY_SLEEP)
                elif "insufficient_quota" in err_str:
                    print(f"\n  [No credits] Add OpenAI credits and retry.")
                    answer = "NO_CREDITS"
                    break
                else:
                    print(f"\n  [Error] {clip_id} {qkey}: {err_str[:80]}")
                    answer = "ERROR"
                    break

        result[f"{qkey}_answer"] = answer
        result[f"{qkey}_correct_answer"] = qcorrect
        result[f"{qkey}_is_correct"] = (answer == qcorrect)
        result[f"{qkey}_raw_response"] = raw_response[:100]

    return result


def main():
    parser = argparse.ArgumentParser(description="GPT-4o evaluation")
    parser.add_argument("--condition", choices=["A", "B"], default="A")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only first N clips (for testing pipeline)")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_key_here":
        print("[Error] OPENAI_API_KEY not set in .env file")
        print("  Add credits at platform.openai.com and set the key in .env")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    print(f"Model: {GPT4O_MODEL} | Condition: {args.condition}")

    if not os.path.exists(ANNOTATION_CSV):
        print(f"[Error] Not found: {ANNOTATION_CSV}")
        sys.exit(1)

    df = pd.read_csv(ANNOTATION_CSV)
    if "include_in_benchmark" in df.columns:
        df = df[df["include_in_benchmark"] != "no"]

    if args.limit:
        df = df.head(args.limit)
        print(f"[Test mode] Evaluating first {args.limit} clips")

    print(f"Total clips: {len(df)}")
    estimated_cost = len(df) * 4 * 0.03  # rough estimate per question
    print(f"Estimated cost: ~${estimated_cost:.2f} USD\n")

    results = []
    output_path = os.path.join(OUTPUT_DIR, f"gpt4o_condition{args.condition}_results.csv")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="GPT-4o")):
        result = evaluate_clip_gpt4o(client, row.to_dict(), args.condition)
        results.append(result)

        if (idx + 1) % 10 == 0:
            pd.DataFrame(results).to_csv(output_path, index=False)
            tqdm.write(f"  [Checkpoint] Saved {idx + 1} clips")

        time.sleep(SLEEP_BETWEEN_CLIPS)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)

    print(f"\n=== GPT-4o Results (Condition {args.condition}) ===")
    for q in ["q1", "q2", "q3", "q4"]:
        col = f"{q}_is_correct"
        label = {"q1": "Cat 1 (2D Dir)", "q2": "Cat 2 (2D Speed)",
                 "q3": "Cat 3 (Depth Dir)", "q4": "Cat 4 (Depth Rate)"}[q]
        if col in results_df.columns:
            acc = results_df[col].mean() * 100
            print(f"  {label}: {acc:.1f}%")

    print(f"\nResults saved: {output_path}")
    print(f"Next: run scripts/stage6_compile_results.py")


if __name__ == "__main__":
    main()
