import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

class AccidentDetector:
    def __init__(self, model_path):
        try:
            self.model = load_model(model_path)
            print(f"✅ Accident Model loaded from {model_path}")
        except Exception as e:
            print(f"❌ Error loading Accident Model: {e}")
            self.model = None
            
        self.img_size = (224, 224)
        self.class_names = ['Accident', 'Non Accident']

    def predict(self, frame):
        if self.model is None:
            return None, 0.0

        # Preprocess
        img = cv2.resize(frame, self.img_size)
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)

        # Inference
        prediction = self.model.predict(img, verbose=0)
        prob = prediction[0][0]
        
        # Assuming the model outputs probability of "Non Accident" (based on class names order typically 0=Accident, 1=Non-Accident in some flows, 
        # BUT let's check the training script. 
        # Training script: CLASS_NAMES = ['Accident', 'Non Accident'], Generator uses class_mode='binary'
        # 'Accident' (folder name usually dictates 0/1 depending on sort).
        # Typically 'Accident' comes first alphabetically -> 0, 'Non Accident' -> 1.
        # So sigmoid output close to 0 is Accident, close to 1 is Non-Accident.
        
        is_accident = prob < 0.5 
        confidence = 1 - prob if is_accident else prob
        
        return is_accident, confidence
