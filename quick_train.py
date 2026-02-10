"""
Quick script untuk train model dengan dataset yang sudah dikonversi
Versi fast dengan epochs lebih sedikit untuk testing
"""

import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.ann_model import ANNClassifier
from backend.text_preprocessing import IndonesianTextPreprocessor
from backend.data_manager import DataManager
import pandas as pd
import glob
import pickle
import json

print("="*60)
print("  Quick Training - Merged Dataset")
print("="*60)

# Check if merged dataset exists
merged_path = 'datasets/merged_all_datasets.csv'

if os.path.exists(merged_path):
    print(f"\n📂 Using existing merged dataset...")
    df = pd.read_csv(merged_path)
else:
    print(f"\n📂 Creating merged dataset...")
    # Scan for all converted CSV files
    csv_files = glob.glob('datasets/*_converted.csv')
    
    if not csv_files:
        print("❌ No converted files found. Run convert_datasets.py first")
        sys.exit(1)
    
    dfs = []
    for f in csv_files:
        dfs.append(pd.read_csv(f))
    
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset=['text'])
    df.to_csv(merged_path, index=False)
    print(f"   Saved to: {merged_path}")

print(f"✅ Dataset: {len(df)} samples")

# Filter to only 7 valid emotions
valid_emotions = ['senang', 'sedih', 'marah', 'takut', 'jijik', 'terkejut', 'netral']
df = df[df['emotion'].isin(valid_emotions)]

print(f"   After filtering: {len(df)} samples")
print(f"\n📊 Distribution:")
dist = df['emotion'].value_counts().sort_index()
for emotion, count in dist.items():
    print(f"   {emotion:10s}: {count:5d} ({count/len(df)*100:.1f}%)")

# Setup DataManager
data_manager = DataManager()
data_manager.df = df
data_manager.emotion_labels = sorted(df['emotion'].unique().tolist())

# Split data
print(f"\n🔀 Splitting data...")
X_train, X_val, X_test, y_train, y_val, y_test = data_manager.split_data(
    test_size=0.15,
    val_size=0.1
)
print(f"   Train: {len(X_train)}")
print(f"   Val: {len(X_val) if X_val is not None else 0}")
print(f"   Test: {len(X_test)}")

# Preprocessing
print(f"\n🔧 Preprocessing...")
preprocessor = IndonesianTextPreprocessor(
    max_features=2000,
    use_stemming=True,
    use_stopword_removal=True
)

X_train_tfidf = preprocessor.fit_transform(X_train)
X_val_tfidf = preprocessor.transform(X_val) if X_val is not None else None
X_test_tfidf = preprocessor.transform(X_test)
print(f"✅ TF-IDF: {X_train_tfidf.shape[1]} features")

# One-hot encode
y_train_onehot = data_manager.one_hot_encode(y_train)
y_val_onehot = data_manager.one_hot_encode(y_val) if y_val is not None else None
y_test_onehot = data_manager.one_hot_encode(y_test)

# Model
print(f"\n🧠 Training ANN...")
model = ANNClassifier(
    input_size=X_train_tfidf.shape[1],
    hidden_layers=[256, 128],
    output_size=len(data_manager.emotion_labels),
    learning_rate=0.01,
    activation='relu'
)

# Train with fewer epochs for quick test
history = model.fit(
    X_train_tfidf, y_train_onehot,
    X_val_tfidf, y_val_onehot,
    epochs=100,  # Reduced for faster training
    batch_size=64,
    verbose=True
)

# Evaluate
print(f"\n📊 Evaluating...")
from sklearn.metrics import accuracy_score, classification_report

y_test_pred = model.predict(X_test_tfidf)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"\n✅ Test Accuracy: {test_accuracy*100:.2f}%")
print(f"\nClassification Report:")
print(classification_report(
    y_test, y_test_pred,
    target_names=data_manager.emotion_labels,
    zero_division=0
))

# Save model
print(f"\n💾 Saving model...")
os.makedirs('backend/models', exist_ok=True)

model.save_model('backend/models/pretrained_model.pkl')
preprocessor.save_preprocessor('backend/models/pretrained_preprocessor.pkl')

metadata = {
    'emotion_labels': data_manager.emotion_labels,
    'input_size': X_train_tfidf.shape[1],
    'output_size': len(data_manager.emotion_labels),
    'test_accuracy': float(test_accuracy),
    'training_samples': len(X_train),
    'architecture': {
        'hidden_layers': [256, 128],
        'activation': 'relu',
        'learning_rate': 0.01
    }
}

with open('backend/models/pretrained_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Model saved!")
print(f"\n" + "="*60)
print(f"✅ Training Complete!")
print(f"="*60)
print(f"\nRestart backend untuk load model baru:")
print(f"   1. Stop: Ctrl+C di terminal backend")
print(f"   2. Start: python backend/main.py")
