"""
Script untuk training dan menyimpan model
Jalankan script ini untuk membuat pre-trained model yang digunakan oleh interface prediksi
"""

import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.ann_model import ANNClassifier
from backend.text_preprocessing import IndonesianTextPreprocessor
from backend.data_manager import DataManager
import pickle
import json

print("="*60)
print("  Training ANN Model untuk Emotion Classification")
print("="*60)

# Scan and load ALL CSV files from datasets folder
print("\n📂 Scanning datasets folder...")
import pandas as pd
import glob

datasets_folder = 'datasets'
csv_files = glob.glob(os.path.join(datasets_folder, '*.csv'))

# Exclude raw folder and non-dataset files
csv_files = [f for f in csv_files if 'raw' not in f.lower() and 'template' not in f.lower()]

if not csv_files:
    print(f"❌ No CSV files found in {datasets_folder}/")
    sys.exit(1)

print(f"✅ Found {len(csv_files)} dataset file(s):")
for i, file in enumerate(csv_files, 1):
    file_size = os.path.getsize(file) / 1024  # KB
    print(f"   {i}. {os.path.basename(file)} ({file_size:.1f} KB)")

# Load and merge all datasets
print("\n🔗 Loading and merging all datasets...")
all_dataframes = []

for csv_file in csv_files:
    try:
        df_temp = pd.read_csv(csv_file, encoding='utf-8')
        
        # Check if it has the right columns
        if 'text' in df_temp.columns and 'emotion' in df_temp.columns:
            all_dataframes.append(df_temp[['text', 'emotion']])
            print(f"   ✅ {os.path.basename(csv_file)}: {len(df_temp)} rows")
        else:
            print(f"   ⚠️  {os.path.basename(csv_file)}: Missing 'text' or 'emotion' columns, skipped")
    except Exception as e:
        print(f"   ❌ {os.path.basename(csv_file)}: Error - {str(e)[:50]}")

if not all_dataframes:
    print(f"❌ No valid datasets found")
    sys.exit(1)

# Merge all dataframes
df_merged = pd.concat(all_dataframes, ignore_index=True)
print(f"\n📊 Initial merge: {len(df_merged)} total rows")

# Remove duplicates
before_dedup = len(df_merged)
df_merged = df_merged.drop_duplicates(subset=['text'], keep='first')
after_dedup = len(df_merged)
print(f"   Removed {before_dedup - after_dedup} duplicate rows")

# Remove null values
before_null = len(df_merged)
df_merged = df_merged.dropna()
after_null = len(df_merged)
if before_null != after_null:
    print(f"   Removed {before_null - after_null} rows with null values")

# Save merged dataset
merged_output = os.path.join(datasets_folder, 'merged_all_datasets.csv')
df_merged.to_csv(merged_output, index=False, encoding='utf-8')
print(f"\n💾 Saved merged dataset: {merged_output}")

# Load using DataManager
print("\n📋 Processing merged dataset...")
data_manager = DataManager()

# Directly set the dataframe
data_manager.df = df_merged
data_manager.emotion_labels = sorted(df_merged['emotion'].unique().tolist())

print(f"✅ Merged dataset loaded: {len(data_manager.df)} samples")
print(f"   Labels: {data_manager.emotion_labels}")
print(f"\n📈 Distribution:")
dist = data_manager.df['emotion'].value_counts().sort_index()
for emotion, count in dist.items():
    percentage = (count / len(data_manager.df)) * 100
    print(f"   {emotion:10s}: {count:4d} samples ({percentage:.1f}%)")

# Split data
print("\n🔀 Splitting data...")
X_train, X_val, X_test, y_train, y_val, y_test = data_manager.split_data(
    test_size=0.2,
    val_size=0.1
)
print(f"   Train: {len(X_train)} samples")
print(f"   Val: {len(X_val) if X_val is not None else 0} samples")
print(f"   Test: {len(X_test)} samples")

# Preprocessing
print("\n🔧 Preprocessing text...")
preprocessor = IndonesianTextPreprocessor(
    max_features=1000,
    use_stemming=True,
    use_stopword_removal=True
)

X_train_tfidf = preprocessor.fit_transform(X_train)
X_val_tfidf = preprocessor.transform(X_val) if X_val is not None else None
X_test_tfidf = preprocessor.transform(X_test)
print(f"✅ TF-IDF features: {X_train_tfidf.shape[1]}")

# One-hot encode
y_train_onehot = data_manager.one_hot_encode(y_train)
y_val_onehot = data_manager.one_hot_encode(y_val) if y_val is not None else None
y_test_onehot = data_manager.one_hot_encode(y_test)

# Initialize model
print("\n🧠 Initializing ANN model...")
input_size = X_train_tfidf.shape[1]
output_size = len(data_manager.emotion_labels)

model = ANNClassifier(
    input_size=input_size,
    hidden_layers=[128, 64],
    output_size=output_size,
    learning_rate=0.01,
    activation='relu'
)
print(f"   Architecture: {input_size} -> [128, 64] -> {output_size}")

# Train
print("\n🚀 Training model...")
print("   This may take a few minutes...")
history = model.fit(
    X_train_tfidf, y_train_onehot,
    X_val_tfidf, y_val_onehot,
    epochs=150,
    batch_size=32,
    verbose=True
)

# Evaluate
print("\n📊 Evaluating on test set...")
from sklearn.metrics import accuracy_score, classification_report

y_test_pred = model.predict(X_test_tfidf)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"\n✅ Test Accuracy: {test_accuracy*100:.2f}%")
print("\nClassification Report:")
print(classification_report(
    y_test, y_test_pred,
    target_names=data_manager.emotion_labels
))

# Save model
print("\n💾 Saving model...")
os.makedirs('backend/models', exist_ok=True)

model_path = 'backend/models/pretrained_model.pkl'
preprocessor_path = 'backend/models/pretrained_preprocessor.pkl'
metadata_path = 'backend/models/pretrained_metadata.json'

model.save_model(model_path)
preprocessor.save_preprocessor(preprocessor_path)

metadata = {
    'emotion_labels': data_manager.emotion_labels,
    'input_size': input_size,
    'output_size': output_size,
    'test_accuracy': float(test_accuracy),
    'architecture': {
        'hidden_layers': [128, 64],
        'activation': 'relu',
        'learning_rate': 0.01
    }
}

with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Model saved:")
print(f"   - {model_path}")
print(f"   - {preprocessor_path}")
print(f"   - {metadata_path}")

print("\n" + "="*60)
print("✅ Training Complete!")
print("="*60)
print("\nSekarang Anda bisa menggunakan interface prediksi:")
print("1. Pastikan backend server running: python backend/main.py")
print("2. Buka: frontend/predict.html")
print("\nModel siap digunakan! 🚀")
