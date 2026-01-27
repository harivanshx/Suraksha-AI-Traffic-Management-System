from ultralytics import YOLO

def train_model(data_yaml_path, epochs=50, imgsz=640):
    """
    Train a YOLOv8 model on a custom dataset.
    
    Args:
        data_yaml_path (str): Path to the data.yaml file describing the dataset.
        epochs (int): Number of training epochs.
        imgsz (int): Image size.
    """
    # Load a model
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data=data_yaml_path, epochs=epochs, imgsz=imgsz)
    
    print("Training finished.")
    print(f"Best model saved at: {results.save_dir}")

if __name__ == "__main__":
    # Example usage:
    # Update 'data.yaml' to point to your dataset config
    # Ensure your data.yaml is correctly formatted as per YOLO standards.
    train_model(data_yaml_path="data/data.yaml")
