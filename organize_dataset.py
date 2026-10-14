#!/usr/bin/env python3
"""
Organize WLASL dataset into class folders
"""

import json
import os
import shutil
from pathlib import Path
from collections import defaultdict

print("="*70)
print("📂 ORGANIZING WLASL DATASET")
print("="*70)

# Paths
JSON_FILE = 'wlasl-processed/WLASL_v0.3.json'
VIDEOS_DIR = 'wlasl-processed/videos'
OUTPUT_DIR = 'data/organized'

# Load the JSON metadata
print(f"\n📖 Reading {JSON_FILE}...")
with open(JSON_FILE, 'r') as f:
    data = json.load(f)

print(f"   ✅ Loaded {len(data)} sign entries")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Map video IDs to glosses (sign names)
video_to_gloss = {}
gloss_counts = defaultdict(int)

for entry in data:
    gloss = entry['gloss']  # Sign name
    for instance in entry['instances']:
        video_id = instance['video_id']
        video_to_gloss[video_id] = gloss
        gloss_counts[gloss] += 1

print(f"\n📊 Dataset Statistics:")
print(f"   Total unique signs: {len(gloss_counts)}")
print(f"   Total video mappings: {len(video_to_gloss)}")

# Filter signs with enough videos
MIN_VIDEOS = 5
MAX_CLASSES = 50  # Limit for faster training

filtered_glosses = {k: v for k, v in gloss_counts.items() if v >= MIN_VIDEOS}
sorted_glosses = sorted(filtered_glosses.items(), key=lambda x: x[1], reverse=True)
selected_glosses = [g[0] for g in sorted_glosses[:MAX_CLASSES]]

print(f"\n✅ Selected {len(selected_glosses)} signs with ≥{MIN_VIDEOS} videos")
print(f"\nTop 20 signs by video count:")
for gloss, count in sorted_glosses[:20]:
    print(f"   {gloss:20} : {count} videos")

# Organize videos into folders
print(f"\n📁 Organizing videos into class folders...")

organized_count = 0
skipped_count = 0

for video_file in Path(VIDEOS_DIR).glob('*.mp4'):
    video_id = video_file.stem  # e.g., "00335" from "00335.mp4"
    
    if video_id in video_to_gloss:
        gloss = video_to_gloss[video_id]
        
        # Only organize selected glosses
        if gloss in selected_glosses:
            # Create class folder
            class_folder = Path(OUTPUT_DIR) / gloss
            class_folder.mkdir(exist_ok=True)
            
            # Copy video to class folder
            dest = class_folder / video_file.name
            if not dest.exists():
                shutil.copy2(video_file, dest)
                organized_count += 1
                
                if organized_count % 50 == 0:
                    print(f"   Processed {organized_count} videos...", end='\r')
        else:
            skipped_count += 1
    else:
        skipped_count += 1

print(f"\n\n✅ Organization complete!")
print(f"   Organized: {organized_count} videos")
print(f"   Skipped: {skipped_count} videos")
print(f"   Output directory: {OUTPUT_DIR}/")

# Show final structure
print(f"\n📂 Final Structure:")
class_folders = sorted(Path(OUTPUT_DIR).iterdir())
for folder in class_folders[:10]:
    if folder.is_dir():
        count = len(list(folder.glob('*.mp4')))
        print(f"   {folder.name:20} : {count} videos")
if len(class_folders) > 10:
    print(f"   ... and {len(class_folders) - 10} more classes")

print(f"\n🚀 Ready for training!")
print(f"   Update train_model.py:")
print(f"   DATA_DIR = 'data/organized'")
print("="*70)
