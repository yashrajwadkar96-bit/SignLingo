#!/usr/bin/env python3
"""
Sign Language Recognition - Mac M1 Training Script
Optimized for Apple Silicon with TensorFlow Metal
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

# TensorFlow with Metal GPU support
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🤟 SIGN LANGUAGE RECOGNITION - MAC M1 OPTIMIZED")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Training configuration"""
    # Paths - ADJUST THIS TO YOUR DATASET PATH
    DATA_DIR = 'data/organized' # <-- Your dataset path
    OUTPUT_DIR = 'models'
    RESULTS_DIR = 'results'
    
    # Model parameters
    MAX_FRAMES = 45           # Increased sequence length for better temporal understanding
    NUM_LANDMARKS = 63        # 21 hand points × 3 coordinates
    
    # Training parameters
    EPOCHS = 80               # More epochs for better convergence
    BATCH_SIZE = 16           # Smaller batch for M1 Air
    LEARNING_RATE = 0.0005    # Lower learning rate for more stable training
    TEST_SIZE = 0.2
    VALIDATION_SPLIT = 0.15
    
    # Early stopping
    PATIENCE = 15             # More patience for better convergence
    MIN_DELTA = 0.0005        # More sensitive to improvements
    
    # Data filtering
    MIN_VIDEOS_PER_CLASS = 9   # Lower threshold to include more classes
    MAX_CLASSES = None         # No limit - use all available classes
    
    # Data augmentation
    USE_AUGMENTATION = True    # Enable data augmentation
    AUGMENTATION_PROB = 0.3    # Probability of applying augmentation
    
    # MediaPipe settings
    MIN_DETECTION_CONFIDENCE = 0.7
    MIN_TRACKING_CONFIDENCE = 0.7

config = Config()

# Create output directories
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
os.makedirs(config.RESULTS_DIR, exist_ok=True)

# ============================================================================
# CHECK SYSTEM CAPABILITIES
# ============================================================================

print("\n📊 System Information:")
print(f"   TensorFlow version: {tf.__version__}")
print(f"   Python version: {sys.version.split()[0]}")

# Check for Metal GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"   🎮 Metal GPU detected: {len(gpus)} device(s)")
    print(f"   ⚡ GPU acceleration: ENABLED")
else:
    print(f"   ⚠️  No GPU detected - using CPU (slower)")

# Set memory growth to avoid OOM errors
try:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except:
    pass

# ============================================================================
# DATA EXTRACTION
# ============================================================================

class LandmarkExtractor:
    """Extract hand landmarks from videos using MediaPipe"""
    
    def __init__(self, max_frames=30):
        self.max_frames = max_frames
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
    
    def extract_from_video(self, video_path):
        """Extract landmarks from a single video"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        landmarks_sequence = []
        frame_count = 0
        
        while cap.isOpened() and frame_count < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.hands.process(frame_rgb)
            
            # Extract landmarks
            if results.multi_hand_landmarks:
                # Take first detected hand
                hand_landmarks = results.multi_hand_landmarks[0]
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                landmarks_sequence.append(landmarks)
            else:
                # No hand detected - add zeros
                landmarks_sequence.append([0.0] * config.NUM_LANDMARKS)
            
            frame_count += 1
        
        cap.release()
        
        # Pad or truncate to fixed length
        while len(landmarks_sequence) < self.max_frames:
            landmarks_sequence.append([0.0] * config.NUM_LANDMARKS)
        
        return np.array(landmarks_sequence[:self.max_frames])
    
    def process_dataset(self, data_dir):
        """Process all videos in the dataset"""
        data_path = Path(data_dir)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        # Find all sign class folders
        class_folders = sorted([d for d in data_path.iterdir() if d.is_dir()])
        
        if not class_folders:
            raise ValueError(f"No class folders found in {data_dir}")
        
        print(f"\n📂 Found {len(class_folders)} sign classes")
        
        # Filter classes by video count
        valid_classes = []
        for class_folder in class_folders:
            video_files = list(class_folder.glob('*.mp4')) + list(class_folder.glob('*.avi')) + list(class_folder.glob('*.mov'))
            if len(video_files) >= config.MIN_VIDEOS_PER_CLASS:
                valid_classes.append(class_folder)
        
        print(f"✅ {len(valid_classes)} classes have ≥{config.MIN_VIDEOS_PER_CLASS} videos")
        
        # Limit number of classes if specified
        if config.MAX_CLASSES:
            valid_classes = valid_classes[:config.MAX_CLASSES]
            print(f"   Limiting to {len(valid_classes)} classes")
        else:
            print(f"   Using all {len(valid_classes)} valid classes")
        
        # Process each class
        X = []
        y = []
        sign_classes = []
        
        for class_folder in valid_classes:
            class_name = class_folder.name
            sign_classes.append(class_name)
            
            video_files = list(class_folder.glob('*.mp4')) + list(class_folder.glob('*.avi')) + list(class_folder.glob('*.mov'))
            
            print(f"\n📹 Processing {class_name}: {len(video_files)} videos")
            
            for video_file in tqdm(video_files, desc=f"   {class_name}", leave=False):
                try:
                    landmarks = self.extract_from_video(video_file)
                    X.append(landmarks)
                    y.append(class_name)
                except Exception as e:
                    print(f"\n   ⚠️  Error in {video_file.name}: {str(e)}")
        
        print(f"\n✅ Extraction complete!")
        print(f"   Total samples: {len(X)}")
        print(f"   Total classes: {len(sign_classes)}")
        
        return np.array(X), np.array(y), sign_classes

# ============================================================================
# EXTRACT FEATURES
# ============================================================================

print("\n" + "="*80)
print("🎯 STEP 1: EXTRACTING HAND LANDMARKS")
print("="*80)

extractor = LandmarkExtractor(max_frames=config.MAX_FRAMES)
X, y, sign_classes = extractor.process_dataset(config.DATA_DIR)

# Display class distribution
print("\n📊 Class Distribution:")
class_counts = Counter(y)
for sign, count in sorted(class_counts.items())[:20]:  # Show first 20
    bar = "█" * (count // 5)
    print(f"   {sign:20} : {bar} ({count} samples)")

# ============================================================================
# DATA AUGMENTATION
# ============================================================================

class DataAugmentation:
    """Data augmentation for landmark sequences"""
    
    @staticmethod
    def add_noise(sequence, noise_factor=0.01):
        """Add small random noise to landmarks"""
        noise = np.random.normal(0, noise_factor, sequence.shape)
        return sequence + noise
    
    @staticmethod
    def time_shift(sequence, max_shift=2):
        """Shift sequence in time"""
        shift = np.random.randint(-max_shift, max_shift + 1)
        if shift == 0:
            return sequence
        if shift > 0:
            return np.concatenate([sequence[shift:], sequence[:shift]], axis=0)
        else:
            return np.concatenate([sequence[shift:], sequence[:shift]], axis=0)
    
    @staticmethod
    def scale_landmarks(sequence, scale_range=(0.95, 1.05)):
        """Scale landmarks by a random factor"""
        scale = np.random.uniform(scale_range[0], scale_range[1])
        return sequence * scale
    
    @staticmethod
    def augment_sequence(sequence, prob=0.3):
        """Apply random augmentation to sequence"""
        if np.random.random() < prob:
            sequence = DataAugmentation.add_noise(sequence)
        if np.random.random() < prob:
            sequence = DataAugmentation.scale_landmarks(sequence)
        return sequence

# ============================================================================
# PREPARE DATA
# ============================================================================

print("\n" + "="*80)
print("🧮 STEP 2: PREPARING DATA FOR TRAINING")
print("="*80)

# Normalize features (per landmark, across all samples)
print("   Normalizing features...")
X_reshaped = X.reshape(-1, config.NUM_LANDMARKS)
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X_reshaped)
X_normalized = X_normalized.reshape(X.shape)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y_categorical,
    test_size=config.TEST_SIZE,
    random_state=42,
    stratify=y_encoded
)

# Apply data augmentation to training set if enabled
if config.USE_AUGMENTATION:
    print("   Applying data augmentation to training set...")
    X_train_augmented = []
    y_train_augmented = []
    
    for i in range(len(X_train)):
        X_train_augmented.append(X_train[i])
        y_train_augmented.append(y_train[i])
        
        # Augment 30% of samples
        if np.random.random() < config.AUGMENTATION_PROB:
            augmented = DataAugmentation.augment_sequence(
                X_train[i], 
                prob=0.5
            )
            X_train_augmented.append(augmented)
            y_train_augmented.append(y_train[i])
    
    original_count = len(X_train)
    X_train = np.array(X_train_augmented)
    y_train = np.array(y_train_augmented)
    print(f"   Augmented training set: {len(X_train)} samples (from {original_count} original)")

print(f"\n✅ Data split complete!")
print(f"   Training samples:   {len(X_train)}")
print(f"   Testing samples:    {len(X_test)}")
print(f"   Number of classes:  {len(sign_classes)}")
print(f"   Sequence shape:     {X_train.shape}")
print(f"   Data augmentation:  {'Enabled' if config.USE_AUGMENTATION else 'Disabled'}")

# Save encoders and scaler
with open(f'{config.OUTPUT_DIR}/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

with open(f'{config.OUTPUT_DIR}/sign_classes.pkl', 'wb') as f:
    pickle.dump(sign_classes, f)

with open(f'{config.OUTPUT_DIR}/feature_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# ============================================================================
# BUILD MODEL
# ============================================================================

print("\n" + "="*80)
print("🏗️  STEP 3: BUILDING IMPROVED LSTM MODEL WITH ATTENTION")
print("="*80)

def attention_layer(inputs):
    """Attention mechanism for temporal sequences"""
    # Simple attention mechanism
    attention = layers.Dense(1, activation='tanh')(inputs)
    attention = layers.Flatten()(attention)
    attention = layers.Activation('softmax')(attention)
    attention = layers.RepeatVector(inputs.shape[-1])(attention)
    attention = layers.Permute([2, 1])(attention)
    
    # Apply attention
    attended = layers.Multiply()([inputs, attention])
    return layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(attended)

def build_improved_lstm_model(input_shape, num_classes):
    """Build improved Bidirectional LSTM model with attention"""
    
    inputs = layers.Input(shape=input_shape)
    
    # Normalization layer
    normalized = layers.LayerNormalization()(inputs)
    
    # First Bidirectional LSTM layer
    lstm1 = layers.Bidirectional(
        layers.LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)
    )(normalized)
    bn1 = layers.BatchNormalization()(lstm1)
    
    # Second Bidirectional LSTM layer
    lstm2 = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)
    )(bn1)
    bn2 = layers.BatchNormalization()(lstm2)
    
    # Third Bidirectional LSTM layer
    lstm3 = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
    )(bn2)
    bn3 = layers.BatchNormalization()(lstm3)
    
    # Attention layer
    attended = attention_layer(bn3)
    
    # Dense layers
    dense1 = layers.Dense(128, activation='relu')(attended)
    bn4 = layers.BatchNormalization()(dense1)
    dropout1 = layers.Dropout(0.4)(bn4)
    
    dense2 = layers.Dense(64, activation='relu')(dropout1)
    bn5 = layers.BatchNormalization()(dense2)
    dropout2 = layers.Dropout(0.3)(bn5)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(dropout2)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

# Create model
input_shape = (config.MAX_FRAMES, config.NUM_LANDMARKS)
num_classes = len(sign_classes)

model = build_improved_lstm_model(input_shape, num_classes)

# Compile with Adam optimizer with learning rate decay
optimizer = keras.optimizers.Adam(
    learning_rate=config.LEARNING_RATE,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-7
)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
)

print("\n📊 Model Architecture:")
model.summary()

total_params = model.count_params()
print(f"\n✅ Model built: {total_params:,} parameters")

# ============================================================================
# TRAIN MODEL
# ============================================================================

print("\n" + "="*80)
print("🎓 STEP 4: TRAINING MODEL")
print("="*80)

print(f"\n⚙️  Configuration:")
print(f"   Epochs: {config.EPOCHS}")
print(f"   Batch size: {config.BATCH_SIZE}")
print(f"   Learning rate: {config.LEARNING_RATE}")
print(f"   Validation split: {config.VALIDATION_SPLIT}")

# Callbacks
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
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]

# Train
print("\n🚀 Starting training...\n")

history = model.fit(
    X_train, y_train,
    validation_split=config.VALIDATION_SPLIT,
    epochs=config.EPOCHS,
    batch_size=config.BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

print("\n✅ Training complete!")

# ============================================================================
# EVALUATE MODEL
# ============================================================================

print("\n" + "="*80)
print("📊 STEP 5: EVALUATING MODEL")
print("="*80)

# Load best model
model = keras.models.load_model(f'{config.OUTPUT_DIR}/best_model.h5')

# Evaluate on test set
eval_results = model.evaluate(X_test, y_test, verbose=0)
if isinstance(eval_results, list):
    test_loss = eval_results[0]
    test_acc = eval_results[1]
    test_top3_acc = eval_results[2] if len(eval_results) > 2 else None
else:
    test_loss = eval_results
    test_acc = None

print(f"\n📈 Test Set Performance:")
if test_acc is not None:
    print(f"   Accuracy:     {test_acc*100:.2f}%")
    if test_top3_acc is not None:
        print(f"   Top-3 Acc:    {test_top3_acc*100:.2f}%")
print(f"   Loss:         {test_loss:.4f}")

# Predictions
y_pred = model.predict(X_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Classification report
print(f"\n📋 Classification Report:\n")
print(classification_report(
    y_true_classes,
    y_pred_classes,
    target_names=sign_classes,
    digits=3
))

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("📊 STEP 6: CREATING VISUALIZATIONS")
print("="*80)

# Training history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Accuracy
ax1.plot(history.history['accuracy'], label='Training', linewidth=2)
ax1.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Loss
ax2.plot(history.history['loss'], label='Training', linewidth=2)
ax2.plot(history.history['val_loss'], label='Validation', linewidth=2)
ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{config.RESULTS_DIR}/training_history.png', dpi=150, bbox_inches='tight')
print("   ✅ Saved: training_history.png")

# Confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=sign_classes,
    yticklabels=sign_classes,
    cbar_kws={'label': 'Count'}
)
plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(f'{config.RESULTS_DIR}/confusion_matrix.png', dpi=150, bbox_inches='tight')
print("   ✅ Saved: confusion_matrix.png")

# ============================================================================
# SAVE MODEL AND INFO
# ============================================================================

print("\n" + "="*80)
print("💾 STEP 7: SAVING MODEL")
print("="*80)

# Save final model
model.save(f'{config.OUTPUT_DIR}/sign_language_model.h5')
print(f"   ✅ Model saved: {config.OUTPUT_DIR}/sign_language_model.h5")

# Save model info
model_info = {
    'sign_classes': sign_classes,
    'num_classes': len(sign_classes),
    'input_shape': input_shape,
    'total_parameters': total_params,
    'test_accuracy': float(test_acc) if test_acc is not None else 0.0,
    'epochs_trained': len(history.history['accuracy']),
    'max_frames': config.MAX_FRAMES,
    'num_landmarks': config.NUM_LANDMARKS
}

with open(f'{config.OUTPUT_DIR}/model_info.pkl', 'wb') as f:
    pickle.dump(model_info, f)
print(f"   ✅ Info saved: {config.OUTPUT_DIR}/model_info.pkl")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("🎉 TRAINING COMPLETE!")
print("="*80)

test_acc_str = f"{test_acc*100:.2f}%" if test_acc is not None else "N/A"
print(f"""
📊 Final Results:
   • Test Accuracy:  {test_acc_str}
   • Classes:        {len(sign_classes)}
   • Parameters:     {total_params:,}

📦 Saved Files:
   • {config.OUTPUT_DIR}/sign_language_model.h5
   • {config.OUTPUT_DIR}/best_model.h5
   • {config.OUTPUT_DIR}/label_encoder.pkl
   • {config.OUTPUT_DIR}/sign_classes.pkl
   • {config.OUTPUT_DIR}/model_info.pkl
   • {config.RESULTS_DIR}/training_history.png
   • {config.RESULTS_DIR}/confusion_matrix.png

🚀 Next Steps:
   1. Test with webcam: python test_webcam.py
   2. Review visualizations in results/
   3. Prepare presentation materials

Good luck! 🎓
""")

print("="*80)
