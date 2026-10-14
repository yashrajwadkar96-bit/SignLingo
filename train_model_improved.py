#!/usr/bin/env python3
"""
IMPROVED: Sign Language Recognition Training
Optimized for small datasets - Better accuracy
"""

import os
import sys
import numpy as np
import cv2
import mediapipe as mp
import pickle
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🤟 IMPROVED SIGN LANGUAGE RECOGNITION")
print("="*80)

# ============================================================================
# IMPROVED CONFIGURATION
# ============================================================================

class Config:
    """Optimized configuration for small datasets"""
    DATA_DIR = 'data/organized'
    OUTPUT_DIR = 'models'
    RESULTS_DIR = 'results'
    
    # Model parameters - OPTIMIZED FOR SPEED
    MAX_FRAMES = 30           # Reduced for faster processing (still good accuracy)
    NUM_LANDMARKS = 63
    
    # Training parameters - OPTIMIZED FOR SPEED + ACCURACY
    EPOCHS = 80               # Reduced epochs (early stopping will handle)
    BATCH_SIZE = 16           # Larger batches = faster training
    LEARNING_RATE = 0.001     # Higher learning rate for faster convergence
    TEST_SIZE = 0.2           # Standard test split
    VALIDATION_SPLIT = 0.15   # Standard validation split
    
    # Early stopping - FASTER CONVERGENCE
    PATIENCE = 15             # Reduced patience for faster stopping
    MIN_DELTA = 0.001         # Less sensitive = stops sooner if not improving
    
    # Data filtering - INCLUDE MORE CLASSES
    MIN_VIDEOS_PER_CLASS = 9  # Lower threshold to include more classes
    MAX_CLASSES = None        # No limit - use all available classes
    
    # Speed optimizations
    SKIP_FRAMES = True        # Skip frames during video processing
    FRAME_SKIP_RATE = 2       # Process every Nth frame
    
    # Feature normalization
    USE_NORMALIZATION = True  # Enable feature normalization
    
    # MediaPipe settings
    MIN_DETECTION_CONFIDENCE = 0.5  # Lower to detect more
    MIN_TRACKING_CONFIDENCE = 0.5

config = Config()

os.makedirs(config.OUTPUT_DIR, exist_ok=True)
os.makedirs(config.RESULTS_DIR, exist_ok=True)

# ============================================================================
# CHECK GPU
# ============================================================================

print("\n📊 System Information:")
print(f"   TensorFlow: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"   ✅ GPU: {len(gpus)} device(s)")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print(f"   ⚠️  CPU only")

# Note: XLA JIT disabled on Mac M1/M2 due to compatibility issues
# The training will still be fast with optimized batch size and model architecture

# ============================================================================
# DATA AUGMENTATION
# ============================================================================

def augment_sequence(sequence, augment_prob=0.4):
    """Enhanced data augmentation for hand landmarks"""
    if np.random.random() > augment_prob:
        return sequence
    
    aug_seq = sequence.copy()
    
    # Random noise (with different intensities)
    if np.random.random() > 0.3:
        noise_factor = np.random.uniform(0.005, 0.02)
        noise = np.random.normal(0, noise_factor, aug_seq.shape)
        aug_seq += noise
    
    # Random scaling
    if np.random.random() > 0.3:
        scale = np.random.uniform(0.92, 1.08)
        aug_seq *= scale
    
    # Time shift (temporal augmentation)
    if np.random.random() > 0.5 and len(aug_seq) > 5:
        shift = np.random.randint(-2, 3)
        if shift != 0:
            aug_seq = np.roll(aug_seq, shift, axis=0)
    
    return aug_seq

# ============================================================================
# LANDMARK EXTRACTOR
# ============================================================================

class LandmarkExtractor:
    def __init__(self, max_frames=30):
        self.max_frames = max_frames
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
        )
    
    def extract_from_video(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open: {video_path}")
        
        landmarks_sequence = []
        
        # Get total frames and sample evenly (optimized)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return np.array([[0.0] * config.NUM_LANDMARKS] * self.max_frames)
        
        # Calculate frame indices to sample
        if config.SKIP_FRAMES and total_frames > self.max_frames * config.FRAME_SKIP_RATE:
            # Skip frames for faster processing
            step = max(1, total_frames // (self.max_frames * config.FRAME_SKIP_RATE))
            frame_indices = list(range(0, total_frames, step))[:self.max_frames]
        else:
            # Sample evenly
            frame_indices = np.linspace(0, total_frames-1, self.max_frames, dtype=int)
        
        current_frame = 0
        frame_indices_set = set(frame_indices)
        
        while cap.isOpened() and len(landmarks_sequence) < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Only process selected frames
            if current_frame in frame_indices_set:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(frame_rgb)
                
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                    landmarks_sequence.append(landmarks)
                else:
                    landmarks_sequence.append([0.0] * config.NUM_LANDMARKS)
            
            current_frame += 1
        
        cap.release()
        
        # Pad if needed
        while len(landmarks_sequence) < self.max_frames:
            landmarks_sequence.append([0.0] * config.NUM_LANDMARKS)
        
        return np.array(landmarks_sequence[:self.max_frames])
    
    def process_dataset(self, data_dir):
        data_path = Path(data_dir)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Not found: {data_dir}")
        
        class_folders = sorted([d for d in data_path.iterdir() if d.is_dir()])
        
        if not class_folders:
            raise ValueError(f"No folders in {data_dir}")
        
        print(f"\n📂 Found {len(class_folders)} sign classes")
        
        # Filter by video count
        valid_classes = []
        for class_folder in class_folders:
            video_files = list(class_folder.glob('*.mp4')) + list(class_folder.glob('*.avi'))
            if len(video_files) >= config.MIN_VIDEOS_PER_CLASS:
                valid_classes.append((class_folder, len(video_files)))
        
        # Sort by video count (most videos first)
        valid_classes.sort(key=lambda x: x[1], reverse=True)
        
        # Limit classes if specified
        if config.MAX_CLASSES:
            valid_classes = [c[0] for c in valid_classes[:config.MAX_CLASSES]]
            print(f"✅ Selected {len(valid_classes)} classes with ≥{config.MIN_VIDEOS_PER_CLASS} videos")
        else:
            valid_classes = [c[0] for c in valid_classes]
            print(f"✅ Using all {len(valid_classes)} classes with ≥{config.MIN_VIDEOS_PER_CLASS} videos")
        
        X = []
        y = []
        sign_classes = []
        
        for class_folder in valid_classes:
            class_name = class_folder.name
            sign_classes.append(class_name)
            
            video_files = list(class_folder.glob('*.mp4')) + list(class_folder.glob('*.avi'))
            
            print(f"\n📹 {class_name}: {len(video_files)} videos")
            
            successful = 0
            for video_file in tqdm(video_files, desc=f"   Processing", leave=False):
                try:
                    landmarks = self.extract_from_video(video_file)
                    
                    # Check if hand was detected (not all zeros)
                    if np.sum(np.abs(landmarks)) > 0:
                        X.append(landmarks)
                        y.append(class_name)
                        successful += 1
                        
                        # Add augmented version (only 50% of samples for speed)
                        if np.random.random() > 0.5:
                            aug_landmarks = augment_sequence(landmarks)
                            X.append(aug_landmarks)
                            y.append(class_name)
                        
                except Exception as e:
                    pass
            
            print(f"   ✅ {successful}/{len(video_files)} successful")
        
        print(f"\n✅ Total samples: {len(X)} (including augmentation)")
        
        return np.array(X), np.array(y), sign_classes

# ============================================================================
# EXTRACT DATA
# ============================================================================

print("\n" + "="*80)
print("🎯 EXTRACTING LANDMARKS (WITH AUGMENTATION)")
print("="*80)

extractor = LandmarkExtractor(max_frames=config.MAX_FRAMES)
X, y, sign_classes = extractor.process_dataset(config.DATA_DIR)

print("\n📊 Class Distribution:")
class_counts = Counter(y)
for sign, count in sorted(class_counts.items()):
    bar = "█" * (count // 10)
    print(f"   {sign:20} : {bar} ({count} samples)")

# ============================================================================
# PREPARE DATA
# ============================================================================

print("\n" + "="*80)
print("🧮 PREPARING DATA")
print("="*80)

# Normalize features if enabled
if config.USE_NORMALIZATION:
    print("   Normalizing features...")
    X_reshaped = X.reshape(-1, config.NUM_LANDMARKS)
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X_reshaped)
    X = X_normalized.reshape(X.shape)
    print("   ✅ Features normalized")
else:
    scaler = None

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical,
    test_size=config.TEST_SIZE,
    random_state=42,
    stratify=y_encoded
)

print(f"\n✅ Split complete!")
print(f"   Training:   {len(X_train)} samples")
print(f"   Testing:    {len(X_test)} samples")
print(f"   Classes:    {len(sign_classes)}")

# Save encoders and scaler
with open(f'{config.OUTPUT_DIR}/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
with open(f'{config.OUTPUT_DIR}/sign_classes.pkl', 'wb') as f:
    pickle.dump(sign_classes, f)
if scaler is not None:
    with open(f'{config.OUTPUT_DIR}/feature_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print(f"   ✅ Feature scaler saved")

# ============================================================================
# BUILD SMALLER MODEL (Better for small data)
# ============================================================================

print("\n" + "="*80)
print("🏗️  BUILDING OPTIMIZED MODEL (Fast Training)")
print("="*80)

def attention_layer(inputs):
    """Attention mechanism for temporal sequences"""
    # Compute attention weights
    attention = layers.Dense(1, activation='tanh')(inputs)
    attention = layers.Flatten()(attention)
    attention = layers.Activation('softmax')(attention)
    attention = layers.RepeatVector(inputs.shape[-1])(attention)
    attention = layers.Permute([2, 1])(attention)
    
    # Apply attention
    attended = layers.Multiply()([inputs, attention])
    return layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(attended)

def build_model(input_shape, num_classes):
    """Optimized model - faster training with good accuracy"""
    inputs = layers.Input(shape=input_shape)
    
    # Normalization layer
    normalized = layers.LayerNormalization()(inputs)
    
    # First Bidirectional LSTM layer (reduced units for speed)
    lstm1 = layers.Bidirectional(
        layers.LSTM(96, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)
    )(normalized)
    bn1 = layers.BatchNormalization()(lstm1)
    
    # Second Bidirectional LSTM layer (removed third layer for speed)
    lstm2 = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)
    )(bn1)
    bn2 = layers.BatchNormalization()(lstm2)
    
    # Attention layer (simplified)
    attended = attention_layer(bn2)
    
    # Dense layers (reduced for speed)
    dense1 = layers.Dense(96, activation='relu')(attended)
    bn3 = layers.BatchNormalization()(dense1)
    dropout1 = layers.Dropout(0.4)(bn3)
    
    dense2 = layers.Dense(48, activation='relu')(dropout1)
    bn4 = layers.BatchNormalization()(dense2)
    dropout2 = layers.Dropout(0.3)(bn4)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(dropout2)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

input_shape = (config.MAX_FRAMES, config.NUM_LANDMARKS)
num_classes = len(sign_classes)

model = build_model(input_shape, num_classes)
optimizer = keras.optimizers.Adam(
    learning_rate=config.LEARNING_RATE,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-7
)
# Compile model - using only accuracy to avoid mixed precision issues
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
print(f"\n✅ Model: {model.count_params():,} parameters")

# ============================================================================
# TRAIN
# ============================================================================

print("\n" + "="*80)
print("🎓 TRAINING (This will take longer)")
print("="*80)

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=config.PATIENCE,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        f'{config.OUTPUT_DIR}/best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-7,
        verbose=1
    )
]

print(f"\n⚙️  Training for up to {config.EPOCHS} epochs")
print(f"   (Will stop early if no improvement)\n")

history = model.fit(
    X_train, y_train,
    validation_split=config.VALIDATION_SPLIT,
    epochs=config.EPOCHS,
    batch_size=config.BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# ============================================================================
# EVALUATE
# ============================================================================

print("\n" + "="*80)
print("📊 EVALUATION")
print("="*80)

model = keras.models.load_model(f'{config.OUTPUT_DIR}/best_model.h5')
eval_results = model.evaluate(X_test, y_test, verbose=0)

if isinstance(eval_results, list):
    test_loss = eval_results[0]
    test_acc = eval_results[1]
    test_top3_acc = eval_results[2] if len(eval_results) > 2 else None
else:
    test_loss = eval_results
    test_acc = None

print(f"\n📈 Test Results:")
if test_acc is not None:
    print(f"   Accuracy: {test_acc*100:.2f}%")
    if test_top3_acc is not None:
        print(f"   Top-3 Acc: {test_top3_acc*100:.2f}%")
print(f"   Loss:     {test_loss:.4f}")

y_pred = model.predict(X_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

print(f"\n📋 Per-Class Report:\n")
print(classification_report(y_true_classes, y_pred_classes, 
                           target_names=sign_classes, digits=3))

# ============================================================================
# VISUALIZE
# ============================================================================

print("\n" + "="*80)
print("📊 CREATING CHARTS")
print("="*80)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(history.history['accuracy'], label='Training', linewidth=2)
ax1.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
ax1.set_title('Accuracy', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(history.history['loss'], label='Training', linewidth=2)
ax2.plot(history.history['val_loss'], label='Validation', linewidth=2)
ax2.set_title('Loss', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{config.RESULTS_DIR}/training_history.png', dpi=150)
print("   ✅ training_history.png")

# Confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
           xticklabels=sign_classes, yticklabels=sign_classes)
plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(f'{config.RESULTS_DIR}/confusion_matrix.png', dpi=150)
print("   ✅ confusion_matrix.png")

# ============================================================================
# SAVE
# ============================================================================

model.save(f'{config.OUTPUT_DIR}/sign_language_model.h5')

model_info = {
    'sign_classes': sign_classes,
    'num_classes': len(sign_classes),
    'test_accuracy': float(test_acc) if test_acc is not None else 0.0,
    'max_frames': config.MAX_FRAMES,
    'num_landmarks': config.NUM_LANDMARKS,
    'total_parameters': model.count_params(),
    'epochs_trained': len(history.history['accuracy'])
}

with open(f'{config.OUTPUT_DIR}/model_info.pkl', 'wb') as f:
    pickle.dump(model_info, f)

print("\n" + "="*80)
print("🎉 IMPROVED TRAINING COMPLETE!")
print("="*80)

test_acc_str = f"{test_acc*100:.2f}%" if test_acc is not None else "N/A"
print(f"""
📊 Results:
   • Test Accuracy: {test_acc_str}
   • Classes: {len(sign_classes)}
   • Signs: {', '.join(sign_classes)}

🚀 Next:
   python test_webcam.py
""")
print("="*80)