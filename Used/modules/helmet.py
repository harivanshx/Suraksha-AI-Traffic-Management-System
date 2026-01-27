from ultralytics import YOLO

class HelmetMonitor:
    def __init__(self, model_path):
        try:
            self.model = YOLO(model_path)
            print(f"✅ Helmet Model loaded from {model_path}")
        except Exception as e:
            print(f"❌ Error loading Helmet Model: {e}")
            self.model = None

    def detect(self, frame, conf_threshold=0.5):
        if self.model is None:
            return []
            
        results = self.model(frame, conf=conf_threshold, verbose=False)
        violations = []
        
        for r in results:
            for box in r.boxes:
                # Get class and confidence
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label_name = self.model.names[cls]
                
                # Check for "No Helmet" class names
                # Adjust these strings based on your actual model classes
                if 'no' in label_name.lower() and 'helmet' in label_name.lower():
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    violations.append({
                        'box': (x1, y1, x2, y2),
                        'confidence': conf,
                        'label': label_name
                    })
                    
        return violations
