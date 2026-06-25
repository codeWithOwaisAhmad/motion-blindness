"""
Stage 1 — Dummy Data Generator
Mimics Intel RealSense D435i output exactly.

What this generates:
- RGB frames: 640x480, BGR, saved as .jpg (same as RealSense color stream)
- Depth frames: 640x480, 16-bit grayscale, saved as .png (same as RealSense depth stream)
- Colorized depth frames: 640x480, BGR colormap, saved as .jpg (for model input)
- One folder per clip, named clip_001 to clip_070
- A clip_metadata.json file recording ground truth depth values

RealSense D435i specs this mimics:
- Color resolution: 640x480 at 30fps
- Depth resolution: 640x480 at 30fps
- Depth range: 0.1m to 10m (values stored in millimeters as uint16)
- Depth units: 1 unit = 1 millimeter

Usage:
    python scripts/stage1_generate_dummy_data.py
"""

import os
import cv2
import numpy as np
import json
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join("data", "dummy")
NUM_CLIPS = 70          # 40-50 object + 20-30 person → we generate 70 total
FRAMES_PER_CLIP = 150   # 5 seconds at 30fps (minimum clip length)
IMG_W, IMG_H = 640, 480
DEPTH_SCALE = 1000      # RealSense stores depth in millimeters

# Motion scenario distribution matching your dataset design
MOTION_SCENARIOS = [
    # (label, depth_motion, 2d_direction, scene_type)
    # depth_motion: "toward" / "away" / "stationary" / "sideways"
    # 2d_direction: "left" / "right" / "toward" / "away" / "stationary"
    ("pure_toward",        "toward",     "toward",     "object"),
    ("pure_away",          "away",       "away",       "object"),
    ("angle_toward",       "toward",     "left",       "object"),
    ("angle_away",         "away",       "right",      "object"),
    ("pure_sideways",      "stationary", "left",       "object"),
    ("stationary",         "stationary", "stationary", "object"),
    ("circle",             "toward",     "left",       "object"),
    ("two_objects",        "toward",     "toward",     "object"),
    ("person_toward",      "toward",     "toward",     "person"),
    ("person_away",        "away",       "away",       "person"),
    ("person_angle_toward","toward",     "left",       "person"),
    ("person_angle_away",  "away",       "right",      "person"),
    ("person_sideways",    "stationary", "left",       "person"),
    ("person_stationary",  "stationary", "stationary", "person"),
]

SPEED_LABELS = ["slow", "medium", "fast"]

OBJECT_TYPES = [
    "ball", "bottle", "box", "book", "cup", "apple", "toy", "pen"
]

PERSON_DESCRIPTIONS = [
    "person in corridor", "person on campus grounds"
]


def generate_rgb_frame(frame_idx: int, scenario: dict) -> np.ndarray:
    """
    Generate a synthetic RGB frame.
    Object is represented as a colored rectangle that changes size/position
    across frames to mimic real motion.
    """
    frame = np.ones((IMG_H, IMG_W, 3), dtype=np.uint8) * 200  # light gray bg

    # Object color varies by object type
    color_map = {
        "ball":   (0, 0, 220),    # red
        "bottle": (0, 180, 0),    # green
        "box":    (200, 100, 0),  # blue
        "book":   (0, 200, 200),  # yellow
        "cup":    (200, 0, 200),  # magenta
        "apple":  (0, 100, 255),  # orange
        "toy":    (100, 0, 200),  # purple
        "pen":    (50, 50, 50),   # dark gray
    }
    obj_color = color_map.get(scenario["object_type"], (100, 100, 200))

    # Base object size
    base_size = 60  # pixels

    # Depth motion → object grows (toward) or shrinks (away) across frames
    progress = frame_idx / FRAMES_PER_CLIP  # 0.0 to 1.0
    depth_motion = scenario["depth_motion"]

    if depth_motion == "toward":
        size_factor = 1.0 + progress * 0.6   # grows 60% larger
    elif depth_motion == "away":
        size_factor = 1.0 - progress * 0.4   # shrinks 40%
    else:
        size_factor = 1.0  # constant size

    obj_size = max(10, int(base_size * size_factor))

    # 2D motion → object shifts position across frames
    center_x = IMG_W // 2
    center_y = IMG_H // 2
    direction_2d = scenario["direction_2d"]

    if direction_2d == "left":
        center_x = int(IMG_W * 0.7 - progress * IMG_W * 0.4)
    elif direction_2d == "right":
        center_x = int(IMG_W * 0.3 + progress * IMG_W * 0.4)

    # Draw object as filled rectangle
    x1 = max(0, center_x - obj_size // 2)
    y1 = max(0, center_y - obj_size // 2)
    x2 = min(IMG_W, center_x + obj_size // 2)
    y2 = min(IMG_H, center_y + obj_size // 2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), obj_color, -1)

    # Add slight noise to make it look more realistic
    noise = np.random.randint(0, 8, frame.shape, dtype=np.uint8)
    frame = cv2.add(frame, noise)

    # Add frame number text (useful for debugging)
    cv2.putText(frame, f"Frame {frame_idx:03d}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1)

    return frame


def generate_depth_frame(frame_idx: int, scenario: dict) -> np.ndarray:
    """
    Generate a synthetic depth frame in RealSense format.
    - 16-bit unsigned integers
    - Values in millimeters
    - Background = 3000mm (3 meters)
    - Object depth changes across frames based on motion scenario
    """
    depth = np.full((IMG_H, IMG_W), 3000, dtype=np.uint16)  # 3m background

    progress = frame_idx / FRAMES_PER_CLIP
    depth_motion = scenario["depth_motion"]

    # Object depth (millimeters)
    start_depth_mm = scenario["start_depth_mm"]
    end_depth_mm = scenario["end_depth_mm"]
    current_depth_mm = int(start_depth_mm + (end_depth_mm - start_depth_mm) * progress)

    # Object position (same as RGB)
    base_size = 60
    direction_2d = scenario["direction_2d"]
    center_x = IMG_W // 2
    center_y = IMG_H // 2

    if direction_2d == "left":
        center_x = int(IMG_W * 0.7 - progress * IMG_W * 0.4)
    elif direction_2d == "right":
        center_x = int(IMG_W * 0.3 + progress * IMG_W * 0.4)

    obj_size = 60
    x1 = max(0, center_x - obj_size // 2)
    y1 = max(0, center_y - obj_size // 2)
    x2 = min(IMG_W, center_x + obj_size // 2)
    y2 = min(IMG_H, center_y + obj_size // 2)

    # Set object region to current depth + small noise
    depth[y1:y2, x1:x2] = current_depth_mm
    noise = np.random.randint(-10, 10, (y2 - y1, x2 - x1), dtype=np.int16)
    depth[y1:y2, x1:x2] = np.clip(
        depth[y1:y2, x1:x2].astype(np.int32) + noise, 100, 8000
    ).astype(np.uint16)

    return depth


def colorize_depth(depth_frame: np.ndarray) -> np.ndarray:
    """
    Convert 16-bit depth frame to BGR colormap.
    This is what gets sent to vision-language models as the "depth image".
    Same colorization as RealSense Viewer uses.
    """
    # Normalize to 0-255 for display (clip to 0-5000mm range)
    depth_clipped = np.clip(depth_frame, 0, 5000).astype(np.float32)
    depth_normalized = (depth_clipped / 5000 * 255).astype(np.uint8)
    depth_colorized = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    return depth_colorized


def assign_depth_values(scenario_label: str, speed: str) -> tuple:
    """
    Assign realistic start and end depth values (in mm) based on scenario.
    These mimic what the RealSense D435i would actually measure.
    """
    speed_multiplier = {"slow": 0.3, "medium": 0.6, "fast": 1.0}[speed]

    if "toward" in scenario_label:
        start = int(np.random.uniform(1200, 2000))  # 1.2m to 2.0m away
        travel = int(600 * speed_multiplier)         # moves 180-600mm closer
        end = max(300, start - travel)               # minimum 30cm
    elif "away" in scenario_label:
        start = int(np.random.uniform(400, 800))     # 0.4m to 0.8m away
        travel = int(600 * speed_multiplier)         # moves 180-600mm farther
        end = min(3000, start + travel)              # maximum 3m
    else:  # stationary or sideways
        start = int(np.random.uniform(600, 1500))
        end = start + int(np.random.uniform(-30, 30))  # <30mm change = stationary

    return start, end


def get_correct_answers(scenario: dict) -> dict:
    """
    Generate ground-truth answers for all 4 question categories.
    These are derived from depth values — objective, not visual judgment.
    """
    depth_change_mm = scenario["end_depth_mm"] - scenario["start_depth_mm"]
    depth_motion = scenario["depth_motion"]
    direction_2d = scenario["direction_2d"]
    speed = scenario["speed"]

    # Category 1 — 2D Direction
    dir_map = {
        "toward": "A",     # A = Toward camera
        "away":   "B",     # B = Away from camera
        "left":   "C",     # C = To the left
        "right":  "D",     # D = To the right
        "stationary": "D", # D = Not moving (repurposed)
    }
    cat1_answer = dir_map.get(direction_2d, "C")

    # Category 2 — 2D Speed
    speed_map = {"slow": "A", "medium": "B", "fast": "C"}
    if direction_2d == "stationary":
        cat2_answer = "D"  # D = Not moving
    else:
        cat2_answer = speed_map[speed]

    # Category 3 — Depth Direction (KEY TEST)
    if depth_change_mm < -50:       # moved more than 5cm closer
        cat3_answer = "A"           # A = Closer — toward camera
    elif depth_change_mm > 50:      # moved more than 5cm farther
        cat3_answer = "B"           # B = Farther — away from camera
    else:
        cat3_answer = "C"           # C = Same distance

    # Category 4 — Depth Rate (KEY TEST)
    # Same logic as Category 3 but different question framing
    if depth_change_mm < -50:
        cat4_answer = "A"           # A = Distance decreased — approached
    elif depth_change_mm > 50:
        cat4_answer = "B"           # B = Distance increased — receded
    elif direction_2d in ["left", "right"]:
        cat4_answer = "D"           # D = Sideways only
    else:
        cat4_answer = "C"           # C = Stayed same

    return {
        "cat1_2d_direction": cat1_answer,
        "cat2_2d_speed": cat2_answer,
        "cat3_depth_direction": cat3_answer,
        "cat4_depth_rate": cat4_answer,
    }


def build_scenario(clip_idx: int) -> dict:
    """Build a full scenario dict for one clip."""
    scenario_template = MOTION_SCENARIOS[clip_idx % len(MOTION_SCENARIOS)]
    label, depth_motion, direction_2d, scene_type = scenario_template
    speed = SPEED_LABELS[clip_idx % len(SPEED_LABELS)]

    if scene_type == "object":
        obj_type = OBJECT_TYPES[clip_idx % len(OBJECT_TYPES)]
        description = obj_type
    else:
        obj_type = "person"
        description = PERSON_DESCRIPTIONS[clip_idx % len(PERSON_DESCRIPTIONS)]

    start_depth_mm, end_depth_mm = assign_depth_values(label, speed)
    depth_change_mm = end_depth_mm - start_depth_mm

    scenario = {
        "clip_id": f"clip_{clip_idx + 1:03d}",
        "scenario_label": label,
        "scene_type": scene_type,
        "object_type": obj_type,
        "description": description,
        "depth_motion": depth_motion,
        "direction_2d": direction_2d,
        "speed": speed,
        "start_depth_mm": start_depth_mm,
        "end_depth_mm": end_depth_mm,
        "depth_change_mm": depth_change_mm,
        "rgb_visible": "partially" if depth_motion in ["toward", "away"] else "yes",
    }

    scenario["correct_answers"] = get_correct_answers(scenario)
    return scenario


def generate_clip(scenario: dict, output_dir: str):
    """Generate all frames for one clip and save them."""
    clip_id = scenario["clip_id"]
    clip_rgb_dir = os.path.join(output_dir, "rgb_frames", clip_id)
    clip_depth_dir = os.path.join(output_dir, "depth_frames", clip_id)
    clip_depth_color_dir = os.path.join(output_dir, "depth_frames", clip_id + "_colorized")

    os.makedirs(clip_rgb_dir, exist_ok=True)
    os.makedirs(clip_depth_dir, exist_ok=True)
    os.makedirs(clip_depth_color_dir, exist_ok=True)

    for frame_idx in range(FRAMES_PER_CLIP):
        # RGB frame
        rgb = generate_rgb_frame(frame_idx, scenario)
        cv2.imwrite(os.path.join(clip_rgb_dir, f"frame_{frame_idx:04d}.jpg"), rgb)

        # Depth frame (16-bit raw)
        depth = generate_depth_frame(frame_idx, scenario)
        cv2.imwrite(os.path.join(clip_depth_dir, f"frame_{frame_idx:04d}.png"), depth)

        # Colorized depth (for model input)
        depth_color = colorize_depth(depth)
        cv2.imwrite(os.path.join(clip_depth_color_dir, f"frame_{frame_idx:04d}.jpg"), depth_color)


def main():
    print("=" * 60)
    print("Stage 1 — Dummy Data Generator")
    print(f"Generating {NUM_CLIPS} clips × {FRAMES_PER_CLIP} frames")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    all_metadata = []

    for clip_idx in tqdm(range(NUM_CLIPS), desc="Generating clips"):
        scenario = build_scenario(clip_idx)
        generate_clip(scenario, OUTPUT_DIR)
        all_metadata.append(scenario)

    # Save metadata JSON — this is ground truth for all downstream scripts
    metadata_path = os.path.join(OUTPUT_DIR, "clip_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(all_metadata, f, indent=2)

    print(f"\nDone. {NUM_CLIPS} clips generated.")
    print(f"Metadata saved to: {metadata_path}")
    print(f"\nFolder sizes:")
    print(f"  rgb_frames/   — {NUM_CLIPS} clip folders × {FRAMES_PER_CLIP} jpg frames")
    print(f"  depth_frames/ — {NUM_CLIPS} clip folders × {FRAMES_PER_CLIP} png frames (16-bit)")
    print(f"               — {NUM_CLIPS} clip folders × {FRAMES_PER_CLIP} jpg frames (colorized)")
    print(f"\nNext: run scripts/stage2_verify_depth_groundtruth.py")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
