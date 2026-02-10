"""
FAST Optimized Training - Target: 85-90% Accuracy in 5-10 minutes
Balanced configuration untuk speed & accuracy
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.ann_model import ANNClassifier
from backend.text_preprocessing import IndonesianTextPreprocessor
from backend.data_manager import DataManager
import pandas as pd
import pickle
import json

print("="*70)
print("  FAST TRAINING - Target: 85-90% Accuracy")
print("="*70)

# Load merged dataset
merged_path = 'datasets/full_merged_dataset.csv'
if not os.path.exists(merged_path):
    print(f"❌ File not found: {merged_path}")
    sys.exit(1)

print(f"\n📂 Loading merged dataset...")
df = pd.read_csv(merged_path, encoding='utf-8')
print(f"✅ Loaded: {len(df)} samples")

# Filter valid emotions
valid_emotions = ['senang', 'sedih', 'marah', 'takut', 'jijik', 'terkejut', 'netral']
df = df[df['emotion'].isin(valid_emotions)]
print(f"   After filtering: {len(df)} samples")

# Show distribution
print(f"\n📊 Distribution:")
dist = df['emotion'].value_counts().sort_index()
for emotion, count in dist.items():
    pct = (count / len(df)) * 100
    print(f"   {emotion:10s}: {count:5d} ({pct:5.1f}%)")

# Setup
data_manager = DataManager()
data_manager.df = df
data_manager.emotion_labels = sorted(df['emotion'].unique().tolist())

# Split
print(f"\n🔀 Splitting data...")
X_train, X_val, X_test, y_train, y_val, y_test = data_manager.split_data(
    test_size=0.15,
    val_size=0.15,
    random_state=42
)
print(f"   Train: {len(X_train)}")
print(f"   Val:   {len(X_val)}")
print(f"   Test:  {len(X_test)}")

# Fast preprocessing - reduced features, no stemming for speed
print(f"\n🔧 Preprocessing (FAST mode)...")
preprocessor = IndonesianTextPreprocessor(
    max_features=2000,  # Reduced from 5000
    use_stemming=False,  # Disable stemming for speed
    use_stopword_removal=True
)

X_train_tfidf = preprocessor.fit_transform(X_train)
X_val_tfidf = preprocessor.transform(X_val)
X_test_tfidf = preprocessor.transform(X_test)
print(f"✅ TF-IDF: {X_train_tfidf.shape[1]} features")

# One-hot
y_train_onehot = data_manager.one_hot_encode(y_train)
y_val_onehot = data_manager.one_hot_encode(y_val)
y_test_onehot = data_manager.one_hot_encode(y_test)

# Smaller model for speed
print(f"\n🧠 Building ANN...")
input_size = X_train_tfidf.shape[1]
output_size = len(data_manager.emotion_labels)

print(f"   Architecture: {input_size} → [256, 128] → {output_size}")
print(f"   Learning rate: 0.01")

model = ANNClassifier(
    input_size=input_size,
    hidden_layers=[256, 128],  # Smaller than [512, 256, 128]
    output_size=output_size,
    learning_rate=0.01,
    activation='relu'
)

# Train with moderate epochs
print(f"\n🚀 Training (150 epochs)...")
print(f"   Estimated time: 5-10 minutes\n")

history = model.fit(
    X_train_tfidf, y_train_onehot,
    X_val_tfidf, y_val_onehot,
    epochs=150,
    batch_size=64,
    verbose=True
)

# Evaluate
print(f"\n📊 Evaluation...")
from sklearn.metrics import accuracy_score, classification_report

y_test_pred = model.predict(X_test_tfidf)
y_train_pred = model.predict(X_train_tfidf)
y_val_pred = model.predict(X_val_tfidf)

test_acc = accuracy_score(y_test, y_test_pred)
train_acc = accuracy_score(y_train, y_train_pred)
val_acc = accuracy_score(y_val, y_val_pred)

print(f"\n{'='*70}")
print(f"  RESULTS")
print(f"{'='*70}")
print(f"\n📈 Accuracy:")
print(f"   Train: {train_acc*100:.2f}%")
print(f"   Val:   {val_acc*100:.2f}%")
print(f"   Test:  {test_acc*100:.2f}%")

if test_acc >= 0.85:
    print(f"\n✅ EXCELLENT! ≥ 85%")
elif test_acc >= 0.75:
    print(f"\n✅ GOOD! ≥ 75%")
else:
    print(f"\n⚠️  Needs improvement")

print(f"\n📋 Classification Report:")
print(classification_report(
    y_test, y_test_pred,
    target_names=data_manager.emotion_labels,
    zero_division=0
))

# Save
print(f"\n💾 Saving model...")
os.makedirs('backend/models', exist_ok=True)

model.save_model('backend/models/pretrained_model.pkl')
preprocessor.save_preprocessor('backend/models/pretrained_preprocessor.pkl')

metadata = {
    'emotion_labels': data_manager.emotion_labels,
    'input_size': input_size,
    'output_size': output_size,
    'test_accuracy': float(test_acc),
    'val_accuracy': float(val_acc),
    'train_accuracy': float(train_acc),
    'training_samples': len(X_train),
    'total_samples': len(df),
    'architecture': {
        'hidden_layers': [256, 128],
        'activation': 'relu',
        'learning_rate': 0.01,
        'epochs': 150,
        'max_features': 2000,
        'use_stemming': False
    }
}

with open('backend/models/pretrained_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Saved!")
print(f"\n{'='*70}")
print(f"✅ COMPLETE!")
print(f"{'='*70}")
print(f"\n📊 Final Test Accuracy: {test_acc*100:.2f}%")
print(f"\n🔄 Next: Restart backend server")
print(f"   Stop: Ctrl+C")
print(f"   Start: python backend/main.py")
