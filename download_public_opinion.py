"""
Download dataset dari berbagai sumber alternatif
"""

import pandas as pd
import requests
import os

print("="*70)
print("  Downloading Indonesian Emotion Dataset")
print("="*70)

# List of possible URLs
urls = [
    "https://raw.githubusercontent.com/Ricco48/Emotion-Dataset-from-Indonesian-Public-Opinion/main/data.csv",
    "https://raw.githubusercontent.com/Ricco48/Emotion-Dataset-from-Indonesian-Public-Opinion/master/data.csv",
    "https://raw.githubusercontent.com/Ricco48/Emotion-Dataset-from-Indonesian-Public-Opinion/main/dataset.csv",
    "https://raw.githubusercontent.com/Ricco48/Emotion-Dataset-from-Indonesian-Public-Opinion/master/dataset.csv",
    "https://raw.githubusercontent.com/Ricco48/Emotion-Dataset-from-Indonesian-Public-Opinion/main/emotion_dataset.csv",
]

df = None
success_url = None

print("\n🔍 Trying multiple URLs...\n")

for url in urls:
    print(f"📥 Trying: {url}")
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            print(f"✅ Success!")
            success_url = url
            
            # Save and load
            os.makedirs("datasets/raw", exist_ok=True)
            temp_path = "datasets/raw/temp_download.csv"
            
            with open(temp_path, 'wb') as f:
                f.write(response.content)
            
            # Try to load
            for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
                try:
                    df = pd.read_csv(temp_path, encoding=encoding)
                    print(f"   Loaded with encoding: {encoding}")
                    break
                except:
                    continue
            
            if df is not None:
                break
        else:
            print(f"   ❌ Status: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {str(e)[:50]}")

if df is None:
    print("\n⚠️ Could not download from GitHub automatically.")
    print("\n📋 Manual alternative:")
    print("   1. Visit: https://github.com/Ricco48/Emotion-Dataset-from-Indonesian-Public-Opinion")
    print("   2. Download the CSV file manually")
    print("   3. Save to: datasets/raw/public_opinion_manual.csv")
    print("   4. Run this script again")
    
    # Check if manual file exists
    manual_path = "datasets/raw/public_opinion_manual.csv"
    if os.path.exists(manual_path):
        print(f"\n✅ Found manual file: {manual_path}")
        df = pd.read_csv(manual_path)
        print(f"   Loaded {len(df)} rows")
    else:
        print("\n❌ Exiting. Please download manually first.")
        exit(1)

# Process the dataset
print(f"\n" + "="*70)
print("  Processing Dataset")
print("="*70)

print(f"\n📊 Dataset Info:")
print(f"   Shape: {df.shape}")
print(f"   Columns: {df.columns.tolist()}")

# Show first few rows
print(f"\n📝 Sample data:")
print(df.head())

# Detect columns
text_col = None
label_col = None

for col in df.columns:
    col_lower = str(col).lower()
    if any(word in col_lower for word in ['text', 'tweet', 'content', 'message']):
        text_col = col
    if any(word in col_lower for word in ['label', 'emotion', 'sentiment', 'class']):
        label_col = col

if text_col is None:
    text_col = df.columns[0]
if label_col is None:
    label_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

print(f"\n🔍 Detected columns:")
print(f"   Text: '{text_col}'")
print(f"   Label: '{label_col}'")

# Show label distribution
print(f"\n📈 Original label distribution:")
label_counts = df[label_col].value_counts()
print(label_counts)

# Map to 7 categories
emotion_mapping = {
    'anger': 'marah', 'Anger': 'marah', 'marah': 'marah', 'angry': 'marah',
    'fear': 'takut', 'Fear': 'takut', 'takut': 'takut', 'scared': 'takut',
    'joy': 'senang', 'Joy': 'senang', 'senang': 'senang', 'happy': 'senang',
    'happiness': 'senang', 'Happy': 'senang', 'Happiness': 'senang',
    'love': 'senang', 'Love': 'senang', 'cinta': 'senang',
    'sadness': 'sedih', 'Sadness': 'sedih', 'sedih': 'sedih', 'sad': 'sedih', 'Sad': 'sedih',
    'disgust': 'jijik', 'Disgust': 'jijik', 'jijik': 'jijik',
    'surprise': 'terkejut', 'Surprise': 'terkejut', 'terkejut': 'terkejut', 'surprised': 'terkejut',
    'neutral': 'netral', 'Neutral': 'netral', 'netral': 'netral',
}

df_clean = pd.DataFrame()
df_clean['text'] = df[text_col]
df_clean['emotion'] = df[label_col].map(emotion_mapping)

# Remove unmapped
before = len(df_clean)
df_clean = df_clean.dropna()
after = len(df_clean)

print(f"\n🔄 Mapping results:")
print(f"   Before: {before}")
print(f"   After: {after}")
print(f"   Dropped (unmapped): {before - after}")

print(f"\n📊 Mapped distribution:")
print(df_clean['emotion'].value_counts().sort_index())

# Save clean version
clean_path = "datasets/public_opinion_emotion.csv"
df_clean.to_csv(clean_path, index=False, encoding='utf-8')
print(f"\n💾 Saved: {clean_path}")

# Merge with existing
existing_path = "datasets/indonesian_emotion_dataset.csv"
if os.path.exists(existing_path):
    print(f"\n🔗 Merging with existing dataset...")
    df_existing = pd.read_csv(existing_path)
    print(f"   Existing: {len(df_existing)} samples")
    print(f"   New: {len(df_clean)} samples")
    
    df_combined = pd.concat([df_existing, df_clean], ignore_index=True)
    before_dedup = len(df_combined)
    df_combined = df_combined.drop_duplicates(subset=['text'], keep='first')
    after_dedup = len(df_combined)
    
    print(f"   Combined: {before_dedup}")
    print(f"   After dedup: {after_dedup}")
    print(f"   Removed: {before_dedup - after_dedup}")
    
    # Shuffle
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save
    merged_path = "datasets/indonesian_emotion_dataset_merged.csv"
    df_combined.to_csv(merged_path, index=False, encoding='utf-8')
    
    print(f"\n💾 Saved merged: {merged_path}")
    print(f"\n📊 Final distribution:")
    dist = df_combined['emotion'].value_counts().sort_index()
    print(dist)
    
    print(f"\n" + "="*70)
    print("✅ SUCCESS!")
    print("="*70)
    print(f"\n📂 Dataset created:")
    print(f"   {merged_path}")
    print(f"\n🎯 Total: {len(df_combined)} samples across 7 emotions")
    print(f"\n💡 Use this file for training!")
else:
    print(f"\n⚠️ No existing dataset found")
    print(f"   Using only new dataset: {clean_path}")
