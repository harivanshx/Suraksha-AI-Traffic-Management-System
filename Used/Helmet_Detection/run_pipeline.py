import os
import argparse
import sys

def check_data():
    """Check if data is labeled."""
    train_labels_dir = "data/train/labels"
    val_labels_dir = "data/val/labels"
    
    if not os.path.exists(train_labels_dir) or not os.listdir(train_labels_dir):
        print("[WARNING] No training labels found in 'data/train/labels'.")
        print("Please label your images using LabelImg or Roboflow before training.")
        return False
    return True

def run_training(epochs=50):
    """Run the training script."""
    print("[INFO] Starting training...")
    if not check_data():
        response = input("Data seems missing/unlabeled. Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
            
    os.system(f"python train.py")

def run_inference(source, model, output):
    """Run the inference script."""
    print(f"[INFO] Starting inference on {source}...")
    cmd = f"python main.py --source {source} --model {model} --output {output}"
    os.system(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Helmet Detection Pipeline")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'predict', 'all'], help='Pipeline mode: train, predict, or all')
    parser.add_argument('--source', type=str, default='0', help='Video source for prediction')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Model path')
    parser.add_argument('--output', type=str, default='output.mp4', help='Output video path')
    
    args = parser.parse_args()
    
    if args.mode == 'train' or args.mode == 'all':
        run_training()
        # After training, usually the best model is at runs/detect/train/weights/best.pt
        # We can update the model path for prediction if running 'all'
        if args.mode == 'all':
            possible_model = "runs/detect/train/weights/best.pt"
            if os.path.exists(possible_model):
                args.model = possible_model
            else:
                print("[WARNING] Could not find trained model. Using default.")

    if args.mode == 'predict' or args.mode == 'all':
        run_inference(args.source, args.model, args.output)
