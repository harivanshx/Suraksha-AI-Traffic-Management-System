"""
Accident Detection Model Training Script
Converted from Jupyter notebook for direct execution
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Deep Learning Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Scikit-learn for metrics
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("="*70)
print("ACCIDENT DETECTION MODEL TRAINING")
print("="*70)
print(f"\nTensorFlow Version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Data paths
DATA_DIR = Path('data')
TRAIN_DIR = DATA_DIR / 'train'
VAL_DIR = DATA_DIR / 'val'
TEST_DIR = DATA_DIR / 'test'

# Model hyperparameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
FINE_TUNE_EPOCHS = 20
LEARNING_RATE = 0.001
CLASS_NAMES = ['Accident', 'Non Accident']

# Create directories
MODEL_DIR = Path('models')
MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)

print(f"\nConfiguration:")
print(f"  Train Directory: {TRAIN_DIR}")
print(f"  Validation Directory: {VAL_DIR}")
print(f"  Test Directory: {TEST_DIR}")
print(f"  Image Size: {IMG_SIZE}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Initial Epochs: {EPOCHS}")
print(f"  Fine-tune Epochs: {FINE_TUNE_EPOCHS}")

# ============================================================================
# DATASET EXPLORATION
# ============================================================================

def count_images(directory):
    """Count images in each class directory."""
    counts = {}
    for class_name in CLASS_NAMES:
        class_path = directory / class_name
        if class_path.exists():
            counts[class_name] = len(list(class_path.glob('*.jpg'))) + len(list(class_path.glob('*.png')))
        else:
            counts[class_name] = 0
    return counts

print("\n" + "="*70)
print("DATASET DISTRIBUTION")
print("="*70)

train_counts = count_images(TRAIN_DIR)
val_counts = count_images(VAL_DIR)
test_counts = count_images(TEST_DIR)

print(f"\nTrain Set:")
for class_name, count in train_counts.items():
    print(f"  {class_name}: {count} images")
print(f"  Total: {sum(train_counts.values())} images")

print(f"\nValidation Set:")
for class_name, count in val_counts.items():
    print(f"  {class_name}: {count} images")
print(f"  Total: {sum(val_counts.values())} images")

print(f"\nTest Set:")
for class_name, count in test_counts.items():
    print(f"  {class_name}: {count} images")
print(f"  Total: {sum(test_counts.values())} images")

# Visualize dataset distribution
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
datasets = [('Train', train_counts), ('Validation', val_counts), ('Test', test_counts)]

for idx, (name, counts) in enumerate(datasets):
    axes[idx].bar(counts.keys(), counts.values(), color=['#e74c3c', '#3498db'])
    axes[idx].set_title(f'{name} Set Distribution', fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('Number of Images')
    axes[idx].set_xlabel('Class')
    axes[idx].grid(axis='y', alpha=0.3)
    
    for i, (class_name, count) in enumerate(counts.items()):
        axes[idx].text(i, count + 5, str(count), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'dataset_distribution.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved dataset distribution plot to {RESULTS_DIR / 'dataset_distribution.png'}")
plt.close()

# ============================================================================
# DATA PREPROCESSING
# ============================================================================

print("\n" + "="*70)
print("DATA PREPROCESSING")
print("="*70)

# Data augmentation for training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

# Only rescaling for validation and test sets
val_test_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators
print("\nCreating data generators...")
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True,
    seed=42
)

val_generator = val_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

test_generator = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

print(f"\nClass Indices: {train_generator.class_indices}")
print(f"Total Training Batches: {len(train_generator)}")
print(f"Total Validation Batches: {len(val_generator)}")
print(f"Total Test Batches: {len(test_generator)}")

# ============================================================================
# MODEL BUILDING
# ============================================================================

print("\n" + "="*70)
print("MODEL ARCHITECTURE")
print("="*70)

# Load pre-trained base model
base_model = MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights='imagenet')
base_model.trainable = False

# Build the model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

print("\nModel Summary:")
model.summary()

total_params = model.count_params()
trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
print(f"\nTotal Parameters: {total_params:,}")
print(f"Trainable Parameters: {trainable_params:,}")

# ============================================================================
# TRAINING CALLBACKS
# ============================================================================

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=str(MODEL_DIR / 'best_accident_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# ============================================================================
# INITIAL TRAINING
# ============================================================================

print("\n" + "="*70)
print("PHASE 1: INITIAL TRAINING (Frozen Base Model)")
print("="*70)

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

print("\n✓ Initial training completed!")

# ============================================================================
# FINE-TUNING
# ============================================================================

print("\n" + "="*70)
print("PHASE 2: FINE-TUNING (Unfrozen Layers)")
print("="*70)

# Unfreeze the last layers of the base model
base_model.trainable = True

# Freeze all layers except the last 20
for layer in base_model.layers[:-20]:
    layer.trainable = False

trainable_layers = sum([1 for layer in base_model.layers if layer.trainable])
print(f"\nFine-tuning with {trainable_layers} trainable layers in base model")

# Recompile with lower learning rate
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE/10),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Continue training
history_fine = model.fit(
    train_generator,
    epochs=FINE_TUNE_EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

print("\n✓ Fine-tuning completed!")

# ============================================================================
# TRAINING HISTORY VISUALIZATION
# ============================================================================

print("\n" + "="*70)
print("VISUALIZING TRAINING HISTORY")
print("="*70)

# Combine histories
for key in history.history.keys():
    history.history[key].extend(history_fine.history[key])

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Accuracy
axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
axes[0, 0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
axes[0, 0].set_title('Model Accuracy', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Loss
axes[0, 1].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[0, 1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
axes[0, 1].set_title('Model Loss', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Precision
axes[1, 0].plot(history.history['precision'], label='Train Precision', linewidth=2)
axes[1, 0].plot(history.history['val_precision'], label='Val Precision', linewidth=2)
axes[1, 0].set_title('Model Precision', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Recall
axes[1, 1].plot(history.history['recall'], label='Train Recall', linewidth=2)
axes[1, 1].plot(history.history['val_recall'], label='Val Recall', linewidth=2)
axes[1, 1].set_title('Model Recall', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Recall')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'training_history.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved training history plot to {RESULTS_DIR / 'training_history.png'}")
plt.close()

# ============================================================================
# MODEL EVALUATION
# ============================================================================

print("\n" + "="*70)
print("MODEL EVALUATION ON TEST SET")
print("="*70)

test_loss, test_accuracy, test_precision, test_recall = model.evaluate(test_generator, verbose=1)
test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall)

print("\n" + "="*70)
print("TEST SET RESULTS")
print("="*70)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test F1-Score: {test_f1:.4f}")
print("="*70)

# ============================================================================
# PREDICTIONS AND CONFUSION MATRIX
# ============================================================================

print("\n" + "="*70)
print("GENERATING PREDICTIONS")
print("="*70)

test_generator.reset()
y_pred_probs = model.predict(test_generator, verbose=1)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()
y_true = test_generator.classes

print(f"\nPredictions shape: {y_pred.shape}")
print(f"True labels shape: {y_true.shape}")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - Accident Detection', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved confusion matrix to {RESULTS_DIR / 'confusion_matrix.png'}")
plt.close()

# Classification Report
print("\n" + "="*70)
print("CLASSIFICATION REPORT")
print("="*70)
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

# ============================================================================
# ROC CURVE
# ============================================================================

print("\n" + "="*70)
print("ROC CURVE ANALYSIS")
print("="*70)

fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - Accident Detection Model', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'roc_curve.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved ROC curve to {RESULTS_DIR / 'roc_curve.png'}")
print(f"Area Under ROC Curve (AUC): {roc_auc:.4f}")
plt.close()

# ============================================================================
# PERFORMANCE SUMMARY
# ============================================================================

summary_data = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC'],
    'Score': [test_accuracy, test_precision, test_recall, test_f1, roc_auc]
}

summary_df = pd.DataFrame(summary_data)
summary_df['Score (%)'] = summary_df['Score'] * 100

plt.figure(figsize=(10, 6))
bars = plt.bar(summary_df['Metric'], summary_df['Score'], 
               color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6'])
plt.ylim([0, 1.1])
plt.ylabel('Score', fontsize=12)
plt.title('Model Performance Metrics', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}',
             ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'performance_metrics.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved performance metrics to {RESULTS_DIR / 'performance_metrics.png'}")
plt.close()

# ============================================================================
# SAVE MODELS
# ============================================================================

print("\n" + "="*70)
print("SAVING MODELS")
print("="*70)

# Save the complete model
model_path = MODEL_DIR / 'accident_detection_model_final.h5'
model.save(model_path)
print(f"✓ Model saved to: {model_path}")

# Save model in TensorFlow SavedModel format
saved_model_path = MODEL_DIR / 'accident_detection_saved_model'
model.save(saved_model_path, save_format='tf')
print(f"✓ SavedModel format saved to: {saved_model_path}")

# Save model weights
weights_path = MODEL_DIR / 'accident_detection_weights.h5'
model.save_weights(weights_path)
print(f"✓ Model weights saved to: {weights_path}")

# Save training history
history_df = pd.DataFrame(history.history)
history_df.to_csv(RESULTS_DIR / 'training_history.csv', index=False)
print(f"✓ Training history saved to: {RESULTS_DIR / 'training_history.csv'}")

# Save metrics summary
summary_df.to_csv(RESULTS_DIR / 'performance_summary.csv', index=False)
print(f"✓ Performance summary saved to: {RESULTS_DIR / 'performance_summary.csv'}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*70)
print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("="*70)

print("\n📊 Final Performance:")
print(f"   • Test Accuracy: {test_accuracy*100:.2f}%")
print(f"   • Test Precision: {test_precision:.4f}")
print(f"   • Test Recall: {test_recall:.4f}")
print(f"   • Test F1-Score: {test_f1:.4f}")
print(f"   • AUC-ROC: {roc_auc:.4f}")

print("\n💾 Saved Models:")
print(f"   • Full Model (H5): {model_path}")
print(f"   • SavedModel Format: {saved_model_path}")
print(f"   • Model Weights: {weights_path}")
print(f"   • Best Model (Checkpoint): {MODEL_DIR / 'best_accident_model.h5'}")

print("\n📈 Saved Results:")
print(f"   • Dataset Distribution: {RESULTS_DIR / 'dataset_distribution.png'}")
print(f"   • Training History: {RESULTS_DIR / 'training_history.png'}")
print(f"   • Confusion Matrix: {RESULTS_DIR / 'confusion_matrix.png'}")
print(f"   • ROC Curve: {RESULTS_DIR / 'roc_curve.png'}")
print(f"   • Performance Metrics: {RESULTS_DIR / 'performance_metrics.png'}")
print(f"   • Training History CSV: {RESULTS_DIR / 'training_history.csv'}")
print(f"   • Performance Summary CSV: {RESULTS_DIR / 'performance_summary.csv'}")

print("\n🎯 Model Details:")
print(f"   • Architecture: Transfer Learning with MobileNetV2")
print(f"   • Input Size: {IMG_SIZE}")
print(f"   • Classes: {CLASS_NAMES}")
print(f"   • Total Parameters: {total_params:,}")

print("\n✅ Ready for deployment!")
print("="*70)
