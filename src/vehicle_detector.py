"""
Vehicle Detection Module using YOLOv8
Optimized for CPU-only execution
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))
import config


class VehicleDetector:
    """
    Detects and tracks vehicles using YOLOv8
    Optimized for CPU performance
    """
    
    def __init__(self):
        """Initialize YOLO model and tracking structures"""
        print(f"[INFO] Initializing YOLOv8 model: {config.YOLO_MODEL}")
        print(f"[INFO] Device: {config.DEVICE} (CPU-only mode)")
        
        # Load YOLO model
        self.model = YOLO(config.YOLO_MODEL)
        
        # Force CPU usage
        self.device = config.DEVICE
        
        # Vehicle tracking
        self.tracked_vehicles = {}
        self.next_vehicle_id = 0
        
        # Statistics
        self.total_detections = 0
        self.fps = 0
        
        print("[INFO] Vehicle detector initialized successfully")
    
    def detect_vehicles(self, frame):
        """
        Detect vehicles in a frame
        
        Args:
            frame: Input image/frame (numpy array)
            
        Returns:
            list: List of detections, each containing:
                  {'bbox': [x1, y1, x2, y2], 'class': str, 'confidence': float, 'id': int}
        """
        start_time = time.time()
        
        # Run YOLO inference (CPU)
        results = self.model.predict(
            frame,
            conf=config.CONFIDENCE_THRESHOLD,
            iou=config.IOU_THRESHOLD,
            imgsz=config.IMG_SIZE,
            device=self.device,
            verbose=False,
            half=False  # Disable half precision for CPU
        )
        
        # Parse results
        detections = []
        
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                
                # Filter only vehicle classes
                if class_id in config.VEHICLE_CLASSES:
                    detection = {
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'class': config.VEHICLE_CLASSES[class_id],
                        'confidence': confidence,
                        'class_id': class_id,
                        'center': (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    }
                    
                    # Add tracking ID if enabled
                    if config.TRACKING_ENABLED:
                        tracking_id = self._assign_tracking_id(detection)
                        detection['id'] = tracking_id
                    
                    detections.append(detection)
                    self.total_detections += 1
        
        # Calculate FPS
        end_time = time.time()
        self.fps = 1 / (end_time - start_time) if (end_time - start_time) > 0 else 0
        
        return detections
    
    def detect_license_plates(self, frame, use_ocr=True):
        """
        Detect license plates in a frame
        
        Args:
            frame: Input image/frame (numpy array)
            use_ocr: Whether to use OCR to extract text from plates
            
        Returns:
            list: List of plate detections, each containing:
                  {'bbox': [x1, y1, x2, y2], 'confidence': float, 'text': str}
        """
        plates = []
        
        # First, detect all objects in the frame
        results = self.model.predict(
            frame,
            conf=0.25,  # Lower confidence for plate detection
            iou=config.IOU_THRESHOLD,
            imgsz=config.IMG_SIZE,
            device=self.device,
            verbose=False,
            half=False
        )
        
        # Look for potential license plate regions
        # Strategy: Look for small rectangular objects or use vehicle detections
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            # Get vehicle detections first
            vehicle_boxes = []
            for box in boxes:
                class_id = int(box.cls[0])
                if class_id in config.VEHICLE_CLASSES:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    vehicle_boxes.append([int(x1), int(y1), int(x2), int(y2)])
            
            # For each vehicle, look for plate-like regions in the lower portion
            for vbox in vehicle_boxes:
                vx1, vy1, vx2, vy2 = vbox
                
                # Focus on lower 40% of vehicle where plates typically are
                plate_search_y1 = int(vy1 + (vy2 - vy1) * 0.6)
                plate_search_y2 = vy2
                
                # Extract region of interest
                roi = frame[plate_search_y1:plate_search_y2, vx1:vx2]
                
                if roi.size == 0:
                    continue
                
                # Use simple heuristics to find plate-like regions
                # Convert to grayscale
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                
                # Apply edge detection
                edges = cv2.Canny(gray, 100, 200)
                
                # Find contours
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Filter by aspect ratio (plates are typically wider than tall)
                    aspect_ratio = w / h if h > 0 else 0
                    
                    # Typical plate aspect ratio is 2:1 to 5:1
                    if 1.5 < aspect_ratio < 6 and w > 30 and h > 10:
                        # Adjust coordinates to full frame
                        plate_x1 = vx1 + x
                        plate_y1 = plate_search_y1 + y
                        plate_x2 = plate_x1 + w
                        plate_y2 = plate_y1 + h
                        
                        # Extract plate region
                        plate_roi = frame[plate_y1:plate_y2, plate_x1:plate_x2]
                        
                        plate_text = ""
                        confidence = 0.5  # Base confidence for detection
                        
                        # Use OCR if enabled
                        if use_ocr and plate_roi.size > 0:
                            try:
                                # Lazy import to avoid loading if not needed
                                import easyocr
                                
                                # Initialize reader if not already done
                                if not hasattr(self, 'ocr_reader'):
                                    print("[INFO] Initializing EasyOCR reader...")
                                    self.ocr_reader = easyocr.Reader(['en'], gpu=False)
                                
                                # Perform OCR
                                ocr_results = self.ocr_reader.readtext(plate_roi)
                                
                                if ocr_results:
                                    # Get the text with highest confidence
                                    best_result = max(ocr_results, key=lambda x: x[2])
                                    plate_text = best_result[1].strip()
                                    confidence = float(best_result[2])
                                    
                                    # Only add if confidence is reasonable and text is not empty
                                    if confidence > 0.3 and len(plate_text) > 0:
                                        plates.append({
                                            'bbox': [plate_x1, plate_y1, plate_x2, plate_y2],
                                            'text': plate_text,
                                            'confidence': confidence
                                        })
                            except ImportError:
                                print("[WARNING] EasyOCR not installed. Install with: pip install easyocr")
                                # Add plate without text
                                plates.append({
                                    'bbox': [plate_x1, plate_y1, plate_x2, plate_y2],
                                    'text': 'DETECTED',
                                    'confidence': confidence
                                })
                            except Exception as e:
                                print(f"[WARNING] OCR failed: {e}")
                                continue
                        else:
                            # Add plate without OCR
                            plates.append({
                                'bbox': [plate_x1, plate_y1, plate_x2, plate_y2],
                                'text': 'DETECTED',
                                'confidence': confidence
                            })
        
        return plates
    
    def detect_emergency_vehicles(self, frame, detections=None):
        """
        Detect emergency vehicles (ambulances, fire trucks) in a frame
        Uses vehicle class detection + red color filtering
        
        Args:
            frame: Input image/frame (numpy array)
            detections: Optional pre-computed vehicle detections (to avoid re-detection)
            
        Returns:
            list: List of emergency vehicle detections, each containing:
                  {'bbox': [x1, y1, x2, y2], 'class': str, 'confidence': float, 
                   'is_emergency': True, 'emergency_type': str}
        """
        emergency_vehicles = []
        
        # Get vehicle detections if not provided
        if detections is None:
            detections = self.detect_vehicles(frame)
        
        # Check each detection for emergency vehicle characteristics
        for detection in detections:
            class_id = detection.get('class_id', -1)
            
            # Only check trucks and buses (potential emergency vehicles)
            if class_id not in config.EMERGENCY_VEHICLE_CLASSES:
                continue
            
            # Extract vehicle region
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            
            # Ensure coordinates are within frame bounds
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            vehicle_roi = frame[y1:y2, x1:x2]
            
            if vehicle_roi.size == 0:
                continue
            
            # Check if vehicle has significant red color (emergency vehicles are often red)
            is_red = self._check_red_color(vehicle_roi)
            
            if is_red:
                # Flag as emergency vehicle
                emergency_type = 'ambulance' if class_id == 7 else 'fire_truck'
                
                emergency_vehicles.append({
                    'bbox': bbox,
                    'class': detection['class'],
                    'confidence': detection['confidence'],
                    'class_id': class_id,
                    'center': detection.get('center', ((x1+x2)//2, (y1+y2)//2)),
                    'is_emergency': True,
                    'emergency_type': emergency_type
                })
        
        return emergency_vehicles
    
    def _check_red_color(self, roi):
        """
        Check if a region of interest has significant red color
        Used to identify emergency vehicles
        
        Args:
            roi: Region of interest (BGR image)
            
        Returns:
            bool: True if significant red color detected
        """
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Create masks for red color (red wraps around in HSV: 0-10 and 160-180)
        mask1 = cv2.inRange(hsv, config.EMERGENCY_COLOR_LOWER_RED1, config.EMERGENCY_COLOR_UPPER_RED1)
        mask2 = cv2.inRange(hsv, config.EMERGENCY_COLOR_LOWER_RED2, config.EMERGENCY_COLOR_UPPER_RED2)
        
        # Combine masks
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        # Calculate percentage of red pixels
        red_pixels = cv2.countNonZero(red_mask)
        total_pixels = roi.shape[0] * roi.shape[1]
        
        if total_pixels == 0:
            return False
        
        red_percentage = red_pixels / total_pixels
        
        # Return True if red percentage exceeds threshold
        return red_percentage >= config.EMERGENCY_COLOR_THRESHOLD
    
    def _assign_tracking_id(self, detection):
        """
        Simple tracking to assign consistent IDs to vehicles
        Uses center point proximity matching
        
        Args:
            detection: Detection dictionary
            
        Returns:
            int: Tracking ID
        """
        center = detection['center']
        bbox = detection['bbox']
        
        # Find closest existing track
        min_distance = float('inf')
        matched_id = None
        
        for vehicle_id, track in list(self.tracked_vehicles.items()):
            # Calculate distance between centers
            track_center = track['center']
            distance = np.sqrt(
                (center[0] - track_center[0])**2 + 
                (center[1] - track_center[1])**2
            )
            
            # Check if within tracking distance
            if distance < config.TRACKING_MAX_DISTANCE and distance < min_distance:
                min_distance = distance
                matched_id = vehicle_id
        
        # Update or create track
        if matched_id is not None:
            # Update existing track
            self.tracked_vehicles[matched_id] = {
                'center': center,
                'bbox': bbox,
                'frames_missing': 0,
                'last_seen': time.time()
            }
            return matched_id
        else:
            # Create new track
            new_id = self.next_vehicle_id
            self.next_vehicle_id += 1
            self.tracked_vehicles[new_id] = {
                'center': center,
                'bbox': bbox,
                'frames_missing': 0,
                'last_seen': time.time()
            }
            return new_id
    
    def cleanup_tracks(self):
        """Remove old tracks that haven't been seen recently"""
        current_time = time.time()
        ids_to_remove = []
        
        for vehicle_id, track in self.tracked_vehicles.items():
            # Remove tracks not seen for N frames
            if current_time - track['last_seen'] > config.TRACKING_MAX_FRAMES_MISSING / 30:  # Assuming ~30 FPS
                ids_to_remove.append(vehicle_id)
        
        for vehicle_id in ids_to_remove:
            del self.tracked_vehicles[vehicle_id]
    
    def get_stats(self):
        """
        Get detection statistics
        
        Returns:
            dict: Statistics including FPS, total detections, active tracks
        """
        return {
            'fps': round(self.fps, 2),
            'total_detections': self.total_detections,
            'active_tracks': len(self.tracked_vehicles)
        }
    
    def draw_detections(self, frame, detections, show_labels=True):
        """
        Draw detection boxes and labels on frame
        
        Args:
            frame: Input frame
            detections: List of detections
            show_labels: Whether to show class labels
            
        Returns:
            frame: Frame with drawn detections
        """
        for detection in detections:
            bbox = detection['bbox']
            vehicle_class = detection['class']
            confidence = detection['confidence']
            
            # Draw bounding box
            color = config.COLORS['GREEN']
            cv2.rectangle(
                frame,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                color,
                config.LINE_THICKNESS
            )
            
            # Draw label
            if show_labels:
                label = f"{vehicle_class}: {confidence:.2f}"
                if 'id' in detection:
                    label = f"ID{detection['id']} {label}"
                
                # Background for text
                (text_width, text_height), _ = cv2.getTextSize(
                    label, config.FONT, config.FONT_SCALE, config.FONT_THICKNESS
                )
                cv2.rectangle(
                    frame,
                    (bbox[0], bbox[1] - text_height - 10),
                    (bbox[0] + text_width, bbox[1]),
                    color,
                    -1
                )
                
                # Text
                cv2.putText(
                    frame,
                    label,
                    (bbox[0], bbox[1] - 5),
                    config.FONT,
                    config.FONT_SCALE,
                    config.COLORS['BLACK'],
                    config.FONT_THICKNESS
                )
        
        return frame


# Test function
if __name__ == "__main__":
    print("Testing Vehicle Detector...")
    
    # Initialize detector
    detector = VehicleDetector()
    
    # Test with webcam or video file
    cap = cv2.VideoCapture(0)  # 0 for webcam, or provide video path
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect vehicles
        detections = detector.detect_vehicles(frame)
        
        # Draw detections
        frame = detector.draw_detections(frame, detections)
        
        # Show stats
        stats = detector.get_stats()
        cv2.putText(
            frame,
            f"FPS: {stats['fps']} | Vehicles: {len(detections)}",
            (10, 30),
            config.FONT,
            config.FONT_SCALE,
            config.COLORS['WHITE'],
            config.FONT_THICKNESS
        )
        
        # Display
        cv2.imshow('Vehicle Detection Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
