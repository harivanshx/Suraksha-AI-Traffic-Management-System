from ultralytics import YOLO
import os
import cv2

def auto_label(img_dir, label_dir, model_path='yolov8n.pt'):
    """
    Auto-label images using a pre-trained model.
    Note: This will only label classes the model knows (Person, Motorcycle).
    User will need to manually refine for 'Helmet' vs 'No-Helmet'.
    """
    model = YOLO(model_path)
    os.makedirs(label_dir, exist_ok=True)
    
    print(f"Auto-labeling images in {img_dir}...")
    
    for filename in os.listdir(img_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        img_path = os.path.join(img_dir, filename)
        
        # Run inference
        results = model(img_path, verbose=False)
        
        # Prepare content for label file
        label_content = []
        for r in results:
            for box in r.boxes:
                # Get class ID
                # COCO classes: 0=person, 3=motorcycle
                # We want to map these to our classes: 
                # Our classes: 0=Motorcyclist, 1=Helmet, 2=No-Helmet
                
                cls_id = int(box.cls[0].item())
                
                # Heuristic Mapping
                new_cls_id = -1
                if cls_id == 3: # Motorcycle -> Motorcyclist (Contextual guess)
                    new_cls_id = 0 
                elif cls_id == 0: # Person -> No-Helmet (Assumption/Placeholder)
                    new_cls_id = 2 # Assuming person on bike has no helmet for now to force model to learn? 
                                   # Or just label as Person? 
                                   # Let's map Person -> No-Helmet (2) as a starting point so training doesn't crash.
                    pass
                
                if cls_id in [0, 3]: # Only keep people and bikes
                    # Normalized coordinates
                    x, y, w, h = box.xywhn[0].tolist()
                    
                    # If it's a person, we save as class 2 (No-Helmet) just to have a valid class ID
                    # If it's a motorcycle, we save as class 0
                    if cls_id == 0:
                        save_cls = 2
                    else:
                        save_cls = 0
                        
                    label_content.append(f"{save_cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

        # Save to .txt file
        label_file = os.path.splitext(filename)[0] + ".txt"
        with open(os.path.join(label_dir, label_file), 'w') as f:
            f.write("\n".join(label_content))
            
    print(f"Finished labeling {img_dir}.")

if __name__ == "__main__":
    # Label Training Data
    auto_label("data/train/images", "data/train/labels")
    # Label Validation Data
    auto_label("data/val/images", "data/val/labels")
