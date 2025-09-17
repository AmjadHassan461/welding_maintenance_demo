from pathlib import Path
import torch

# Paths

ROOT = Path(__file__).resolve().parents[1]

# Data directories
DATASETS_DIR = ROOT / "Datasets"

# Vision dataset (YOLOv5 format)
YOLOV5_DIR = ROOT / "yolov5"
YOLO_DATA_YAML = DATASETS_DIR / "CV Seam Detection Dataset" / "data.yaml"
YOLO_PROJECT_NAME = "weld_yolo"

# Sensor dataset
SENSORS_RAW_DIR = DATASETS_DIR / "Simulated Dataset for Edge-Based Defect"

# Models and docs
MODELS_DIR = ROOT / "models"
DOCS_DIR = ROOT / "docs"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)


# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Vision config
VISION_IMG_SIZE = (128, 128) 
VISION_BATCH_SIZE = 32
VISION_EPOCHS = 100            


# Sensor config
SENSOR_WINDOW = 256
SENSOR_STEP = 128
SENSOR_FEATURES = [
    "Arc Voltage (V)",
    "Weld Current (A)",
    "Weld Speed (cm/min)",
    "Wire Feed Speed (m/min)",
    "Gas Flow Rate (L/min)",
    "Torch Angle (°)",
    "Base Metal Temp (°C)",
]
SENSOR_CLASSES = ["normal", "fault"]
SENSOR_TEST_SIZE = 0.2
RANDOM_STATE = 42
SENSOR_LR = 1e-3
SENSOR_EPOCHS = 50             
