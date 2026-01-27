import cv2
import numpy as np
from ultralytics import YOLO

class TrafficMonitor:
    def __init__(self, model_path="yolo11n.pt", conf_threshold=0.4):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        # vehicle class IDs (COCO): car=2, motorcycle=3, bus=5, truck=7
        self.vehicle_classes = [2, 3, 5, 7]
        
        # Traffic timing configuration (seconds)
        self.timings = {
            'low': 10,
            'medium': 20,
            'high': 30
        }

    def get_green_time(self, vehicle_count):
        """Decide green signal duration based on density"""
        if vehicle_count < 5:
            return self.timings['low']
        elif vehicle_count < 10:
            return self.timings['medium']
        else:
            return self.timings['high']

    def detect_vehicles(self, frame):
        """
        Detects vehicles in the frame and counts them per lane.
        Assumes 2 lanes split by the vertical center.
        """
        height, width, _ = frame.shape
        lane_1_count = 0
        lane_2_count = 0
        
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        
        detections = [] # Store for drawing later if needed
        
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls in self.vehicle_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = (x1 + x2) // 2
                    
                    # Assign lane
                    if cx < width // 2:
                        lane_1_count += 1
                        lane_id = 1
                        color = (0, 255, 0)
                    else:
                        lane_2_count += 1
                        lane_id = 2
                        color = (0, 0, 255)
                        
                    detections.append({
                        'box': (x1, y1, x2, y2),
                        'color': color,
                        'lane': lane_id,
                        'label': self.model.names[cls]
                    })

        # Logic for signal
        if lane_1_count > lane_2_count:
            green_lane = "LANE 1"
            green_time = self.get_green_time(lane_1_count)
        else:
            green_lane = "LANE 2"
            green_time = self.get_green_time(lane_2_count)
            
        return {
            'lane_1_count': lane_1_count,
            'lane_2_count': lane_2_count,
            'green_lane': green_lane,
            'green_time': green_time,
            'detections': detections
        }
