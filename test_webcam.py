#!/usr/bin/env python3
"""
Real-time Sign Language Recognition with Webcam
Mac M1 Optimized
"""

import cv2
import numpy as np
import mediapipe as mp
import pickle
from tensorflow import keras
from collections import deque
import time

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = 'models/sign_language_model.h5'
ENCODER_PATH = 'models/label_encoder.pkl'
CLASSES_PATH = 'models/sign_classes.pkl'
SCALER_PATH = 'models/feature_scaler.pkl'  # Feature normalization

# Recognition settings - Match training configuration
SEQUENCE_LENGTH = 30  # Updated to match training MAX_FRAMES
CONFIDENCE_THRESHOLD = 0.6  # Higher threshold to reduce false positives
SMOOTHING_WINDOW = 5  # Increased for more stable predictions
MIN_HAND_CONFIDENCE = 0.3  # Minimum confidence to accept hand detection

# Display settings
WINDOW_NAME = "Sign Language Recognition"
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.2
FONT_THICKNESS = 2

# Colors (BGR format)
COLOR_GREEN = (0, 255, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLUE = (255, 0, 0)

# ============================================================================
# LOAD MODEL AND ENCODERS
# ============================================================================

print("="*70)
print("🤟 SIGN LANGUAGE RECOGNITION - REAL-TIME TEST")
print("="*70)

print("\n📦 Loading model and encoders...")

try:
    # Load trained model
    model = keras.models.load_model(MODEL_PATH, safe_mode=False)
    print(f"   ✅ Model loaded: {MODEL_PATH}")
    
    # Load label encoder
    with open(ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
    print(f"   ✅ Encoder loaded: {ENCODER_PATH}")
    
    # Load sign classes
    with open(CLASSES_PATH, 'rb') as f:
        sign_classes = pickle.load(f)
    print(f"   ✅ Classes loaded: {len(sign_classes)} signs")
    
    # Load feature scaler if it exists (for normalization)
    scaler = None
    try:
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        print(f"   ✅ Feature scaler loaded: {SCALER_PATH}")
    except FileNotFoundError:
        print(f"   ⚠️  Feature scaler not found (normalization disabled)")
    
except Exception as e:
    print(f"\n❌ Error loading model: {e}")
    print("\nMake sure you've trained the model first!")
    exit(1)

# ============================================================================
# INITIALIZE MEDIAPIPE
# ============================================================================

print("\n🎥 Initializing MediaPipe Hands...")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # Enable both hands detection
    min_detection_confidence=0.5,  # Lower threshold for better detection
    min_tracking_confidence=0.5
)

print("   ✅ MediaPipe ready!")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_landmarks(hand_landmarks_list):
    """Extract and combine landmarks from one or both hands"""
    landmarks = []
    
    # Process up to 2 hands (model expects 63 landmarks = 21 points × 3 coords)
    # If two hands, combine them; if one hand, pad with zeros
    if len(hand_landmarks_list) == 2:
        # Two hands: combine both (21 points from each = 42 points, but model expects 21)
        # Take first hand only for now (model was trained with single hand)
        for lm in hand_landmarks_list[0].landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    elif len(hand_landmarks_list) == 1:
        # One hand: extract normally
        for lm in hand_landmarks_list[0].landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        # No hands: return zeros
        landmarks = [0.0] * 63
    
    return np.array(landmarks)

def smooth_predictions(prediction_history, window=5):
    """Average predictions over recent frames for stability"""
    if len(prediction_history) < window:
        return prediction_history[-1]
    
    # Average the probabilities
    recent_preds = np.array(list(prediction_history)[-window:])
    smoothed = np.mean(recent_preds, axis=0)
    return smoothed

def draw_info_panel(frame, prediction, confidence, fps, num_hands=0, seq_length=0):
    """Draw information overlay on frame"""
    h, w = frame.shape[:2]
    
    # Semi-transparent panel (taller to show more info)
    panel = frame.copy()
    cv2.rectangle(panel, (10, 10), (w-10, 150), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.7, frame, 0.3, 0, frame)
    
    # Prediction text
    if confidence >= CONFIDENCE_THRESHOLD:
        color = COLOR_GREEN
        status = "✓"
    elif confidence > 0:
        color = COLOR_YELLOW
        status = "?"
    else:
        color = COLOR_RED
        status = "✗"
    
    text = f"{status} {prediction}"
    cv2.putText(frame, text, (20, 50), FONT, FONT_SCALE, color, FONT_THICKNESS)
    
    # Confidence bar
    if confidence > 0:
        bar_width = int((w - 40) * confidence)
        cv2.rectangle(frame, (20, 70), (20 + bar_width, 90), color, -1)
    cv2.rectangle(frame, (20, 70), (w - 20, 90), COLOR_WHITE, 2)
    
    conf_text = f"{confidence*100:.1f}%"
    cv2.putText(frame, conf_text, (w - 120, 85), FONT, 0.6, COLOR_WHITE, 2)
    
    # Hands detected counter
    hands_text = f"Hands: {num_hands}"
    cv2.putText(frame, hands_text, (20, 110), FONT, 0.6, COLOR_WHITE, 1)
    
    # Sequence length indicator
    seq_text = f"Sequence: {seq_length}/{SEQUENCE_LENGTH}"
    cv2.putText(frame, seq_text, (20, 130), FONT, 0.5, COLOR_WHITE, 1)
    
    # FPS counter
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(frame, fps_text, (w - 120, 130), FONT, 0.5, COLOR_WHITE, 1)

def draw_instructions(frame):
    """Draw control instructions"""
    h, w = frame.shape[:2]
    
    instructions = [
        "Controls:",
        "S - Save Screenshot",
        "Q - Quit",
        "",
        f"Signs: {', '.join(sign_classes)}"  # Show available signs
    ]
    
    y_offset = h - 150
    for i, text in enumerate(instructions):
        cv2.putText(frame, text, (20, y_offset + i*25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)

# ============================================================================
# MAIN RECOGNITION LOOP
# ============================================================================

print("\n" + "="*70)
print("🎥 Starting webcam...")
print("="*70)

print("""
Controls:
   S - Save screenshot
   Q - Quit

Available signs: {', '.join(sign_classes)}
Position your hand in frame and start signing!
""")

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open webcam")
    exit(1)

# Set camera properties for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# State variables
sequence = deque(maxlen=SEQUENCE_LENGTH)
prediction_history = deque(maxlen=SMOOTHING_WINDOW)
frame_count = 0
start_time = time.time()

print("✅ Webcam started! Press 'Q' to quit.\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("⚠️  Failed to grab frame")
        break
    
    # Mirror the frame for more intuitive interaction
    frame = cv2.flip(frame, 1)
    
    # Convert to RGB for MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe
    results = hands.process(image_rgb)
    
    # Default prediction
    prediction = "No hand detected"
    confidence = 0.0
    
    # Count detected hands
    num_hands_detected = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
    
    # Extract landmarks if hand detected
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
        # Draw all detected hands
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
        
        # Extract landmarks from all detected hands
        landmarks = extract_landmarks(results.multi_hand_landmarks)
        
        # Check if we have valid landmarks (not all zeros)
        if np.sum(np.abs(landmarks)) > 0.1:  # Threshold to ensure hand is actually detected
            sequence.append(landmarks)
            
            # Make prediction when we have enough frames
            if len(sequence) == SEQUENCE_LENGTH:
                # Prepare input sequence
                input_sequence = np.array(list(sequence))
                
                # Normalize if scaler is available
                if scaler is not None:
                    # Reshape for normalization
                    original_shape = input_sequence.shape
                    input_reshaped = input_sequence.reshape(-1, 63)
                    input_normalized = scaler.transform(input_reshaped)
                    input_sequence = input_normalized.reshape(original_shape)
                
                # Expand dimensions for model input
                input_data = np.expand_dims(input_sequence, axis=0)
                
                # Predict
                pred_probs = model.predict(input_data, verbose=0)[0]
                prediction_history.append(pred_probs)
                
                # Smooth predictions
                smoothed_probs = smooth_predictions(prediction_history, SMOOTHING_WINDOW)
                
                # Get prediction
                pred_idx = np.argmax(smoothed_probs)
                confidence = smoothed_probs[pred_idx]
                
                # Get top 3 predictions for better feedback
                top3_indices = np.argsort(smoothed_probs)[-3:][::-1]
                
                if confidence >= CONFIDENCE_THRESHOLD:
                    prediction = label_encoder.classes_[pred_idx]
                else:
                    # Show top prediction even if below threshold
                    prediction = f"{label_encoder.classes_[pred_idx]} (low conf)"
        else:
            # Invalid landmarks, clear sequence
            if len(sequence) > 0:
                sequence.popleft()  # Remove oldest instead of clearing all
    else:
        # No hands detected - gradually clear sequence
        if len(sequence) > 0:
            sequence.popleft()  # Remove one frame at a time
        prediction = "No hand detected"
        confidence = 0.0
    
    # Calculate FPS
    frame_count += 1
    elapsed = time.time() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0
    
    # Draw UI elements
    draw_info_panel(frame, prediction, confidence, fps, num_hands_detected, len(sequence))
    draw_instructions(frame)
    
    # Display frame
    cv2.imshow(WINDOW_NAME, frame)
    
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q') or key == ord('Q'):
        print("\n👋 Quitting...")
        break
    
    elif key == ord('s') or key == ord('S'):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        cv2.imwrite(filename, frame)
        print(f"   📸 Screenshot saved: {filename}")
        
# Cleanup
cap.release()
cv2.destroyAllWindows()
hands.close()

print("\n" + "="*70)
print("✅ Session ended")
print(f"   Total frames processed: {frame_count}")
print(f"   Average FPS: {fps:.1f}")
print("="*70)