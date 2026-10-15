# RDMB-100: Radial Depth-Motion Benchmark for Video-LLMs

[![Benchmark: RDMB-100](https://img.shields.io/badge/Benchmark-RDMB--100-blue.svg)](#)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official repository for **RDMB-100**, a diagnostic evaluation benchmark comprising 100 physically captured RGB-D video sequences recorded with an active Intel RealSense sensor. RDMB-100 is designed to rigorously evaluate whether contemporary Vision-Language Models (Video-LLMs) perceive real physical displacement along the camera's optical depth axis ($Z$-axis) or suffer from systematic motion blindness.

---

## 📊 Benchmark Evaluation Summary

All models were evaluated under a standardized uniform keyframe sampling protocol ($T=8$) using identical multiple-choice questions across four diagnostic categories:
- **Cat 1:** 2D Lateral Direction (Left / Right / Up / Down)
- **Cat 2:** Lateral Speed Assessment
- **Cat 3:** Radial Depth Direction (Moving Closer / Moving Away / Stationary / Inconclusive)
- **Cat 4:** Radial Rate of Depth Displacement

### Main Results

| Model | Cat 1 (2D Dir) | Cat 2 (Speed) | Cat 3 (Depth Dir) | Cat 4 (Depth Rate) | Overall | Pred A (Tow) | Pred B (Rec) | Pred C (Stat) | $\beta_{\text{toward}}$ | Stationary Prior | Pure Away Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Baseline** | 25.0% | 25.0% | 25.0% | 25.0% | 25.0% | 25 | 25 | 25 | 0.00 | 25.0% | 25.0% |
| **SmolVLM-Instruct (2.2B)** | 20.0% | 28.0% | 23.0% | 15.0% | **21.5%** | 34 | 0 | 66 | +0.09 | 66.0% | **0.0%** |
| **Video-LLaVA-7B** | 20.0% | 15.0% | 35.0% | 13.0% | **20.8%** | 100 | 0 | 0 | +0.75 | 0.0% | **0.0%** |
| **Qwen2.5-VL-3B-Instruct** | 47.0% | 16.0% | 48.0% | 15.0% | **31.5%** | 65 | 1 | 33 | +0.40 | 33.0% | **0.0%** |
| **Qwen2.5-VL-7B-Instruct (4-bit)** | 51.0% | 25.0% | 52.0% | 20.0% | **37.0%** | 22 | 16 | 62 | -0.03 | 62.0% | **0.0%** |

### Key Diagnostic Findings

1. **Complete Failure on Pure Recession (0.0% Accuracy):**
   Across all evaluated architectures, every model scored **0.0% on canonical receding sequences** (`pure_away`, $N=10$). Models consistently fail to translate shrinking retinal projections into motion directed away from the observer.
2. **Choice Collapse vs. Stationary Induction:**
   - **Video-LLaVA-7B** undergoes extreme option-A collapse, outputting "moving closer" for 100/100 clips ($\beta_{\text{toward}} = +0.75$).
   - **SmolVLM-2.2B** and **Qwen2.5-VL-7B** manifest an opposite inductive failure: when confronted with subtle depth changes, they exhibit strong stationary priors (predicting "stationary" in 66% and 62% of sequences, respectively).
3. **2D vs. 3D Spatial Discrepancy:**
   While larger architectures demonstrate emerging 2D optical flow understanding (51.0% on Cat 1 for Qwen2.5-VL-7B), depth perception drops back toward or below chance levels (20.0% on Cat 4).

---

## 📁 Repository Organization

```text
.
├── data/
│   └── master_annotations_100clips.csv   # Ground-truth depth trajectories, speed, and QA pairs
├── notebooks/
│   ├── 01_eval_smolvlm_2.2b.ipynb        # Standalone Kaggle evaluation for SmolVLM (2.2B)
│   ├── 02_eval_videollava_7b.ipynb       # Standalone Kaggle evaluation for Video-LLaVA (7B)
│   ├── 03_eval_qwen2.5_vl_3b.ipynb       # Standalone Kaggle evaluation for Qwen2.5-VL-3B
│   └── 04_eval_qwen2.5_vl_7b.ipynb       # Standalone Kaggle evaluation for Qwen2.5-VL-7B (NF4)
├── scripts/
│   ├── compile_tables.py                 # Automated verification of Table 4 and Table 5 metrics
│   └── generate_figures.py               # Generates 300 DPI publication-quality figures
├── results/
│   └── raw_predictions/                  # Unedited model predictions per clip
│       ├── results_smolvlm_2b.csv
│       ├── results_video_llava_7b.csv
│       ├── results_qwen2_5_vl_3b.csv
│       └── results_qwen2_5_vl_7b.csv
├── requirements.txt                      # Pinned Python package dependencies
└── README.md
