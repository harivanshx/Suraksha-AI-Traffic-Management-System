# Helmet Detection System

This project is an AI-based system for detecting safety helmets on motorbike riders using YOLOv8.

## Prerequisites

Ensure you have the following installed:

- Python 3.8+
- Webcam (for real-time detection) or video file

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

You can run the project in different modes using the `run_pipeline.py` script or by running individual scripts.

### 1. Real-time Detection (Inference)

To run detection using your webcam:

```bash
python main.py
```

Or for a specific video file:

```bash
python main.py --source path/to/video.mp4
```

You can also use the pipeline script:

```bash
python run_pipeline.py --mode predict
```

### 2. Training the Model

To train the model on your custom dataset (ensure `data/data.yaml` is configured):

```bash
python train.py
```

Or use the pipeline:

```bash
python run_pipeline.py --mode train
```

### 3. Full Pipeline (Train + Predict)

To train and then immediately run prediction:

```bash
python run_pipeline.py --mode all
```

## Structure

- `main.py`: Main script for inference (detection).
- `train.py`: Script to train the YOLOv8 model.
- `run_pipeline.py`: wrapper script to manage training and inference modes.
- `src/`: Source code for detector and utilities.
- `data/`: Directory for dataset and config (see `DATASET_SETUP.md`).

## Arguments

`main.py` accepts the following arguments:

- `--source`: Video source (default: '0' for webcam).
- `--model`: Path to model file (default: 'yolov8n.pt').
- `--conf`: Confidence threshold (default: 0.5).
- `--output`: Path to save output video (default: 'output.mp4').
