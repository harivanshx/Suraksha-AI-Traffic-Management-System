import csv
import os
from datetime import datetime

class ViolationLogger:
    def __init__(self, log_dir="logs"):
        """
        Initialize the CSV logger.
        :param log_dir: Directory to save log files
        """
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create a new log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"violation_log_{timestamp}.csv")
        
        # Write header
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Violation Type", "Confidence", "Image Frame"])
            
    def log_violation(self, violation_type, confidence, frame_idx=None):
        """
        Log a violation event.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, violation_type, f"{confidence:.2f}", frame_idx])
            
    def get_log_path(self):
        return self.log_file
