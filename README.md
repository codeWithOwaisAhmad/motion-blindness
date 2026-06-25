# Motion-Blindness: Depth-Axis Motion Benchmark for Video-Language Models

Owais Ahmad | Islamia University Bahawalpur | 2026

## Pipeline Stages
1. Dummy data generation (mimics RealSense D435i output)
2. Frame extraction from .bag files
3. Depth ground truth verification
4. Annotation spreadsheet generation
5. Human baseline collection
6. Model evaluation (GPT-4o, Gemini, Video-LLaVA)
7. Accuracy calculation and visualization

## Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
