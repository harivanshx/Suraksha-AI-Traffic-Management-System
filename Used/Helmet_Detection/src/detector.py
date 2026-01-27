from ultralytics import YOLO
import cv2

class HelmetDetector:
    def __init__(self, model_path):
        """
        Initialize the YOLOv8 model.
        :param model_path: Path to the trained .pt model file
        """
        self.model = YOLO(model_path)
    
    def detect(self, frame, conf_threshold=0.5):
        """
        Perform detection on a frame.
        :param frame: Input image/frame
        :param conf_threshold: Confidence threshold for detections
        :return: Results object from YOLO
        """
        results = self.model(frame, conf=conf_threshold, verbose=False)
        return results
