"""
Configuration file for AI-Based Traffic Management System
CPU-optimized settings for real-time performance
"""

import os

# =================== SYSTEM CONFIGURATION ===================
# Force CPU usage (no GPU)
DEVICE = 'cpu'
USE_GPU = False

# =================== YOLO MODEL CONFIGURATION ===================
# Using YOLOv11n (nano) for best CPU performance
YOLO_MODEL = 'yolo11n.pt'  # YOLOv11 nano - latest model
CONFIDENCE_THRESHOLD = 0.4  # Lowered for better detection on CPU
IOU_THRESHOLD = 0.45
IMG_SIZE = 640  # Standard size, can reduce to 416 for better CPU performance

# Vehicle classes from COCO dataset
VEHICLE_CLASSES = {
    2: 'car',
    3: 'motorcycle', 
    5: 'bus',
    7: 'truck'
}

# Emergency vehicle classes (subset of VEHICLE_CLASSES)
# Note: COCO doesn't have specific ambulance/fire truck classes
# We use truck/bus as proxies with additional color filtering
EMERGENCY_VEHICLE_CLASSES = {
    5: 'bus',      # Can include fire trucks
    7: 'truck'     # Can include ambulances
}

# Emergency vehicle priority settings
EMERGENCY_PRIORITY_ENABLED = True
EMERGENCY_MIN_GREEN_DURATION = 30  # Minimum green time for emergency vehicles (seconds)
EMERGENCY_OVERRIDE_DELAY = 3  # Seconds to wait before switching (yellow phase)
EMERGENCY_DETECTION_CONFIDENCE = 0.5  # Higher confidence for emergency detection

# Color detection for emergency vehicles (HSV ranges for red)
EMERGENCY_COLOR_LOWER_RED1 = (0, 100, 100)    # Lower bound for red hue (0-10)
EMERGENCY_COLOR_UPPER_RED1 = (10, 255, 255)
EMERGENCY_COLOR_LOWER_RED2 = (160, 100, 100)  # Upper bound for red hue (160-180)
EMERGENCY_COLOR_UPPER_RED2 = (180, 255, 255)
EMERGENCY_COLOR_THRESHOLD = 0.15  # Minimum 15% red pixels to flag as emergency

# =================== DETECTION ZONES ===================
# Define detection zones for 4-way intersection
# Format: {'name': [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]}
# Coordinates are normalized (0-1) and will be scaled to frame size

DETECTION_ZONES = {
    'NORTH': [(0.3, 0.0), (0.7, 0.0), (0.7, 0.4), (0.3, 0.4)],
    'SOUTH': [(0.3, 0.6), (0.7, 0.6), (0.7, 1.0), (0.3, 1.0)],
    'EAST': [(0.6, 0.3), (1.0, 0.3), (1.0, 0.7), (0.6, 0.7)],
    'WEST': [(0.0, 0.3), (0.4, 0.3), (0.4, 0.7), (0.0, 0.7)]
}

# =================== TRAFFIC DENSITY CONFIGURATION ===================
# Thresholds for traffic density classification
DENSITY_THRESHOLDS = {
    'LOW': 5,       # 0-5 vehicles: Low traffic
    'MEDIUM': 15,   # 6-15 vehicles: Medium traffic  
    'HIGH': 25,     # 16-25 vehicles: High traffic
    'CRITICAL': 26  # 26+ vehicles: Critical congestion
}

# =================== TRAFFIC SIGNAL TIMING ===================
# Signal durations in seconds
SIGNAL_TIMINGS = {
    'LOW': {
        'green': 15,
        'yellow': 3,
        'min_green': 10
    },
    'MEDIUM': {
        'green': 30,
        'yellow': 3,
        'min_green': 20
    },
    'HIGH': {
        'green': 45,
        'yellow': 3,
        'min_green': 30
    },
    'CRITICAL': {
        'green': 60,
        'yellow': 3,
        'min_green': 40
    }
}

# Maximum green light duration (safety limit)
MAX_GREEN_DURATION = 90
MIN_GREEN_DURATION = 10

# Red light duration is automatic (sum of other signals' green+yellow)

# =================== VISUALIZATION SETTINGS ===================
# Colors (BGR format for OpenCV)
COLORS = {
    'RED': (0, 0, 255),
    'YELLOW': (0, 255, 255),
    'GREEN': (0, 255, 0),
    'WHITE': (255, 255, 255),
    'BLACK': (0, 0, 0),
    'BLUE': (255, 0, 0),
    'ORANGE': (0, 165, 255),
    'PURPLE': (255, 0, 255)
}

# Density level colors
DENSITY_COLORS = {
    'LOW': (0, 255, 0),      # Green
    'MEDIUM': (0, 255, 255),  # Yellow
    'HIGH': (0, 165, 255),    # Orange
    'CRITICAL': (0, 0, 255)   # Red
}

# Vehicle type colors
VEHICLE_COLORS = {
    'car': (0, 255, 0),        # Green for cars
    'motorcycle': (255, 0, 0),  # Blue for motorcycles
    'bus': (0, 165, 255),       # Orange for buses
    'truck': (255, 0, 255)      # Purple for trucks
}

# UI Settings
FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2
LINE_THICKNESS = 2

# Display settings
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
FPS_DISPLAY = True

# =================== VIDEO PROCESSING ===================
# Frame processing settings
PROCESS_EVERY_N_FRAMES = 2  # Process every 2nd frame for better CPU performance
SKIP_FRAMES = False  # Set to True to skip frames if processing is slow

# Video output settings
SAVE_OUTPUT = False  # Set to True to save processed video
OUTPUT_PATH = 'output/traffic_output.mp4'
OUTPUT_FPS = 15  # Lower FPS for output video

# =================== TRACKING SETTINGS ===================
# Simple tracking to avoid duplicate counts
TRACKING_ENABLED = True
TRACKING_MAX_DISTANCE = 50  # pixels
TRACKING_MAX_FRAMES_MISSING = 10

# =================== LOGGING SETTINGS ===================
LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR
LOG_FILE = 'logs/traffic_system.log'
ENABLE_STATS = True
STATS_UPDATE_INTERVAL = 5  # seconds

# =================== PATHS ===================
# Create necessary directories
DIRS_TO_CREATE = [
    'logs',
    'output',
    'models',
    'sample_videos'
]

# Create directories if they don't exist
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for dir_name in DIRS_TO_CREATE:
    dir_path = os.path.join(BASE_DIR, dir_name)
    os.makedirs(dir_path, exist_ok=True)
