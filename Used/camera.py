import cv2
import time
import numpy as np
from modules.traffic import TrafficMonitor
from modules.accident import AccidentDetector
from modules.helmet import HelmetMonitor

class VideoCamera:
    def __init__(self, source=0):
        # Initialize video source
        self.video = cv2.VideoCapture(source)
        
        # Initialize Models
        print("Loading models...")
        self.traffic_model = TrafficMonitor(model_path="yolo11n.pt")
        self.accident_model = AccidentDetector(model_path="Accident Detection/models/best_accident_model.h5")
        # Assuming we use the same YOLOv8n for helmet or a specific one if available. 
        # Using yolov8n.pt as placeholder or the one from the folder if it exists.
        # The user has 'yolov8n.pt' in root.
        self.helmet_model = HelmetMonitor(model_path="yolov8n.pt") 
        print("Models loaded.")

        # State storage
        self.stats = {
            'lane_1_count': 0,
            'lane_2_count': 0,
            'green_lane': 'Initializing...',
            'green_time': 0,
            'accident_alert': False,
            'violations': 0
        }
        
        # Frame counters for interleaved processing
        self.frame_count = 0

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, frame = self.video.read()
        if not success:
            return None

        self.frame_count += 1
        
        # Resize for consistent processing speed (optional but recommended)
        # frame = cv2.resize(frame, (640, 480))

        # -----------------------------
        # TRAFFIC DENSITY (Every frame or every 2nd frame)
        # -----------------------------
        if self.frame_count % 3 == 0:
            traffic_res = self.traffic_model.detect_vehicles(frame)
            self.stats.update({
                'lane_1_count': traffic_res['lane_1_count'],
                'lane_2_count': traffic_res['lane_2_count'],
                'green_lane': traffic_res['green_lane'],
                'green_time': traffic_res['green_time']
            })
            
            # Draw Traffic Detections
            for det in traffic_res['detections']:
                x1, y1, x2, y2 = det['box']
                cv2.rectangle(frame, (x1, y1), (x2, y2), det['color'], 2)

        # -----------------------------
        # ACCIDENT DETECTION (Every 10 frames)
        # -----------------------------
        if self.frame_count % 10 == 0:
            is_accident, conf = self.accident_model.predict(frame)
            self.stats['accident_alert'] = bool(is_accident)
            
        if self.stats['accident_alert']:
            cv2.putText(frame, "ACCIDENT DETECTED!", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # -----------------------------
        # HELMET DETECTION (Every 5 frames)
        # -----------------------------
        if self.frame_count % 5 == 0:
            violations = self.helmet_model.detect(frame)
            self.stats['violations'] = len(violations)
            
            # Draw Violations
            for v in violations:
                x1, y1, x2, y2 = v['box']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "NO HELMET", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Overlays
        self._draw_overlay(frame)

        # Encode for web
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def _draw_overlay(self, frame):
        # Draw stats on frame (optional, since we have dashboard)
        # Can be useful for the video feed itself
        h, w, _ = frame.shape
        cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 255, 255), 2)
        
    def get_stats(self):
        return self.stats
