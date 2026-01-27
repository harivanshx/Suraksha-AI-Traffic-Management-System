# 🚦 AI-Based Traffic Management System

An intelligent traffic management system that uses YOLOv8 for real-time vehicle detection and dynamically controls traffic signals based on congestion levels. **Optimized for CPU-only execution.**

![Traffic Management Demo](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-orange)

> 📖 **For detailed problem statement, objectives, and technical design, see [PROBLEM_STATEMENT.md](PROBLEM_STATEMENT.md)**

## ✨ Features

### 🎯 Core Capabilities
- **Real-time Vehicle Detection**: Uses YOLOv8n (nano) model optimized for CPU performance
- **Traffic Density Analysis**: Analyzes traffic in 4 directions (North, South, East, West)
- **Dynamic Signal Control**: Adjusts green light duration (15-90s) based on congestion
- **Multi-vehicle Tracking**: Tracks cars, buses, trucks, and motorcycles
- **Live Visualization**: Real-time UI showing detections, zones, signals, and statistics

### 🎨 Traffic Density Classification
- **LOW** (0-5 vehicles): 15-20s green light
- **MEDIUM** (6-15 vehicles): 25-35s green light
- **HIGH** (16-25 vehicles): 40-60s green light
- **CRITICAL** (26+ vehicles): 60-90s green light

### 💻 CPU Optimization
- Uses lightweight YOLOv8n model
- Frame skipping for better performance
- No GPU/CUDA required
- Runs efficiently on standard laptops

---

## 📋 Requirements

- Python 3.8 or higher
- Webcam or traffic video file (for testing)
- Windows/Linux/MacOS

---

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd AI-Based-Traffic-Management-System
```

### 2. Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

The first time you run the system, YOLOv8n model (~6MB) will be automatically downloaded.

---

## 🎮 Usage

### Basic Usage (Webcam)
```bash
python main.py
```

### Use Video File
```bash
python main.py --input path/to/traffic_video.mp4
```

### Save Output Video
```bash
python main.py --input video.mp4 --save-output
```

### Headless Mode (No Display)
```bash
python main.py --input video.mp4 --no-display
```

### Keyboard Controls
- **Q**: Quit the application
- **S**: Skip to next signal (manual override)
- **P**: Pause/Resume

---

## 📁 Project Structure

```
AI-Based-Traffic-Management-System/
│
├── main.py                      # Main application entry point
├── requirements.txt             # Python dependencies
├── README.md                   # This file
│
├── src/
│   ├── config.py               # Configuration settings
│   ├── vehicle_detector.py    # YOLOv8 vehicle detection
│   ├── traffic_analyzer.py    # Traffic density analysis
│   ├── signal_controller.py   # Dynamic signal control
│   └── visualizer.py           # Real-time visualization
│
├── sample_videos/              # Place test videos here
├── output/                     # Saved output videos
├── logs/                       # System logs
└── models/                     # YOLO models (auto-downloaded)
```

---

## ⚙️ Configuration

Edit `src/config.py` to customize:

### Detection Settings
```python
CONFIDENCE_THRESHOLD = 0.4      # Detection confidence (0-1)
YOLO_MODEL = 'yolov8n.pt'      # Model size (n/s/m/l/x)
PROCESS_EVERY_N_FRAMES = 2     # Frame skip for performance
```

### Traffic Signal Timing
```python
SIGNAL_TIMINGS = {
    'LOW': {'green': 15, 'yellow': 3},
    'MEDIUM': {'green': 30, 'yellow': 3},
    'HIGH': {'green': 45, 'yellow': 3},
    'CRITICAL': {'green': 60, 'yellow': 3}
}
```

### Detection Zones
Modify zone coordinates for different intersection layouts:
```python
DETECTION_ZONES = {
    'NORTH': [(0.3, 0.0), (0.7, 0.0), (0.7, 0.4), (0.3, 0.4)],
    # ... customize other zones
}
```

---

## 🎯 How It Works

### 1. **Vehicle Detection**
- Uses YOLOv8n to detect vehicles in each frame
- Filters for: cars, buses, trucks, motorcycles
- Tracks vehicles to avoid duplicate counting

### 2. **Traffic Analysis**
- Divides intersection into 4 detection zones
- Counts vehicles in each zone
- Classifies density: LOW/MEDIUM/HIGH/CRITICAL

### 3. **Signal Control**
- Adjusts green light duration based on density
- Ensures minimum green time (safety)
- Rotates through all directions fairly
- Gives priority to heavily congested lanes

### 4. **Visualization**
- Shows live video with detection boxes
- Displays traffic zones with color coding
- Shows signal states with countdown timers
- Real-time statistics (FPS, vehicle count, etc.)

---

## 📊 Performance

### Expected Performance (CPU)
- **FPS**: 15-25 FPS on modern CPU (Intel i5/AMD Ryzen 5)
- **Accuracy**: 85-95% vehicle detection accuracy
- **Latency**: <100ms processing per frame

### Optimization Tips
1. Reduce video resolution for better FPS
2. Increase `PROCESS_EVERY_N_FRAMES` to skip more frames
3. Adjust `CONFIDENCE_THRESHOLD` for speed vs accuracy trade-off
4. Use YOLOv8n (smallest model) for best CPU performance

---

## 🧪 Testing

### Test Individual Modules

**Vehicle Detector:**
```bash
cd src
python vehicle_detector.py
```

**Traffic Analyzer:**
```bash
cd src
python traffic_analyzer.py
```

**Signal Controller:**
```bash
cd src
python signal_controller.py
```

### Sample Videos
Download sample traffic videos from:
- YouTube (search "traffic intersection")
- [COCO Dataset](https://cocodataset.org/)
- [AI City Challenge](https://www.aicitychallenge.org/)

Place videos in `sample_videos/` directory.

---

## 🔧 Troubleshooting

### Low FPS
- Reduce frame resolution
- Increase `PROCESS_EVERY_N_FRAMES`
- Use smaller YOLO model (already using yolov8n)

### Poor Detection
- Adjust `CONFIDENCE_THRESHOLD` (try 0.3-0.5)
- Ensure good lighting in video
- Camera angle should clearly show vehicles

### Signals Not Changing
- Check if vehicles are detected in zones
- Verify zone coordinates match your video
- Check console output for errors

---

## 🎓 System Architecture

```
┌─────────────┐
│ Video Input │
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│ YOLO Detection   │ ──► Detect & Track Vehicles
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Traffic Analysis │ ──► Count Vehicles per Zone
└──────┬───────────┘     Classify Density
       │
       ▼
┌──────────────────┐
│ Signal Control   │ ──► Adjust Signal Timing
└──────┬───────────┘     Manage State Machine
       │
       ▼
┌──────────────────┐
│ Visualization    │ ──► Display UI
└──────────────────┘     Show Stats
```

---

## 🌟 Future Enhancements

- [ ] Emergency vehicle detection and priority
- [ ] Pedestrian crossing integration
- [ ] Multi-intersection coordination
- [ ] Historical data analysis
- [ ] Web dashboard for monitoring
- [ ] Integration with actual traffic lights (hardware)
- [ ] Mobile app for remote monitoring

---

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 👤 Author

Created as an AI-based traffic management solution.

---

## 🙏 Acknowledgments

- **YOLOv8** by Ultralytics for object detection
- **OpenCV** for computer vision utilities
- Traffic management concepts from urban planning research

---

## 📧 Support

For issues or questions, please open an issue on GitHub.

---

**⭐ If you find this project useful, please consider giving it a star!**
#   A I - B a s e d - T r a f f i c - M a n a g e m e n t  
 