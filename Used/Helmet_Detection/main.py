import cv2
import argparse
import time
from src.detector import HelmetDetector
from src.utils import plot_one_box

from src.logger import ViolationLogger

def main(video_path, model_path, conf_thres, output_path=None):
    # Initialize Detector
    detector = HelmetDetector(model_path)
    
    # Initialize Logger
    logger = ViolationLogger()
    print(f"Logging violations to: {logger.get_log_path()}")

    # Initialize Video Capture
    source = video_path
    if video_path.isdigit():
        source = int(video_path) # Webcam
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    # Video Writer Setup
    out = None
    if output_path:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Saving output video to: {output_path}")

    print(f"Processing video: {video_path}")
    print("Press 'q' to exit.")

    frame_count = 0
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
            
        # Detection
        results = detector.detect(frame, conf_threshold=conf_thres)
        
        # Process results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates
                xyxy = box.xyxy[0].cpu().numpy()
                # Get confidence
                conf = box.conf[0].cpu().numpy()
                # Get class
                cls = int(box.cls[0].cpu().numpy())
                label_name = detector.model.names[cls]
                
                label = f'{label_name} {conf:.2f}'
                
                # Dynamic Logic & Logging
                color = (0, 255, 0) # Green default
                if 'no check helmet' in label_name.lower() or 'no-helmet' in label_name.lower():
                    color = (0, 0, 255) # Red
                    # Log violation
                    logger.log_violation("No Helmet Detected", conf, frame_count)
                elif 'helmet' in label_name.lower():
                    color = (0, 255, 0) # Green
                else:
                    color = (255, 255, 0) # Cyan for others (Person/Bike)

                plot_one_box(xyxy, frame, label=label, color=color, line_thickness=2)

        # FPS calculation
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Write to output file
        if out:
            out.write(frame)

        # Display
        cv2.imshow('Helmet Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0', help='source video path or webcam index')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='path to model file')
    parser.add_argument('--conf', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--output', type=str, default='output.mp4', help='path to save output video')
    opt = parser.parse_args()
    
    main(opt.source, opt.model, opt.conf, opt.output)
