"""
Optimized Training Script - Target: 90% Accuracy
Menggunakan semua dataset converted dan hyperparameter optimization
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.ann_model import ANNClassifier
from backend.text_preprocessing import IndonesianTextPreprocessor
from backend.data_manager import DataManager
import pandas as pd
import glob
import pickle
import json

print("="*70)
print("  OPTIMIZED TRAINING - Target: 90% Accuracy")
print("="*70)

# Step 1: Merge ALL converted datasets
print("\n📂 Step 1: Merging all converted datasets...")
converted_files = glob.glob('datasets/*_converted.csv')

if not converted_files:
    print("❌ No converted files found!")
    print("   Run: python convert_datasets.py")
    sys.exit(1)

print(f"   Found {len(converted_files)} converted files")

all_dfs = []
total_rows = 0

for f in converted_files:
    df_temp = pd.read_csv(f, encoding='utf-8')
    all_dfs.append(df_temp)
    total_rows += len(df_temp)
    print(f"   ✅ {os.path.basename(f)}: {len(df_temp)} rows")

# Merge
df = pd.concat(all_dfs, ignore_index=True)
print(f"\n📊 Initial merge: {len(df)} rows")

# Filter valid emotions
valid_emotions = ['senang', 'sedih', 'marah', 'takut', 'jijik', 'terkejut', 'netral']
df = df[df['emotion'].isin(valid_emotions)]
print(f"   After filtering: {len(df)} rows")

# Deduplicate
before = len(df)
df = df.drop_duplicates(subset=['text'], keep='first')
print(f"   After dedup: {len(df)} rows (removed {before - len(df)})")

# Remove nulls
df = df.dropna()

# Save merged
merged_path = 'datasets/full_merged_dataset.csv'
df.to_csv(merged_path, index=False, encoding='utf-8')
print(f"\n💾 Saved: {merged_path}")

# Show distribution
print(f"\n📈 Emotion Distribution:")
dist = df['emotion'].value_counts().sort_index()
for emotion, count in dist.items():
    pct = (count / len(df)) * 100
    bar = '█' * int(pct / 2)
    print(f"   {emotion:10s}: {count:5d} ({pct:5.1f}%) {bar}")

# Step 2: Setup DataManager
print(f"\n🔧 Step 2: Setting up data manager...")
data_manager = DataManager()
data_manager.df = df
data_manager.emotion_labels = sorted(df['emotion'].unique().tolist())

# Step 3: Split data with optimal ratio
print(f"\n🔀 Step 3: Splitting data (70/15/15)...")
X_train, X_val, X_test, y_train, y_val, y_test = data_manager.split_data(
    test_size=0.15,
    val_size=0.15,
    random_state=42
)
print(f"   Train: {len(X_train)} samples")
print(f"   Val:   {len(X_val)} samples")
print(f"   Test:  {len(X_test)} samples")

# Step 4: Advanced preprocessing
print(f"\n🔧 Step 4: Advanced text preprocessing...")
preprocessor = IndonesianTextPreprocessor(
    max_features=5000,  # Increased for better feature representation
    use_stemming=True,
    use_stopword_removal=True
)

X_train_tfidf = preprocessor.fit_transform(X_train)
X_val_tfidf = preprocessor.transform(X_val)
X_test_tfidf = preprocessor.transform(X_test)
print(f"✅ TF-IDF features: {X_train_tfidf.shape[1]}")

# One-hot encode
y_train_onehot = data_manager.one_hot_encode(y_train)
y_val_onehot = data_manager.one_hot_encode(y_val)
y_test_onehot = data_manager.one_hot_encode(y_test)

# Step 5: Build optimized model
print(f"\n🧠 Step 5: Building optimized ANN architecture...")
input_size = X_train_tfidf.shape[1]
output_size = len(data_manager.emotion_labels)

print(f"   Architecture: {input_size} → [512, 256, 128] → {output_size}")
print(f"   Activation: ReLU")
print(f"   Learning rate: 0.005 (optimized)")

model = ANNClassifier(
    input_size=input_size,
    hidden_layers=[512, 256, 128],  # Deeper network
    output_size=output_size,
    learning_rate=0.005,  # Lower learning rate for better convergence
    activation='relu'
)

# Step 6: Train with more epochs
print(f"\n🚀 Step 6: Training (200 epochs)...")
print(f"   This will take 5-10 minutes for optimal results\n")

history = model.fit(
    X_train_tfidf, y_train_onehot,
    X_val_tfidf, y_val_onehot,
    epochs=200,  # More epochs for better learning
    batch_size=64,
    verbose=True
)

# Step 7: Evaluate
print(f"\n📊 Step 7: Evaluation...")
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Test set evaluation
y_test_pred = model.predict(X_test_tfidf)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Training set evaluation
y_train_pred = model.predict(X_train_tfidf)
train_accuracy = accuracy_score(y_train, y_train_pred)

# Validation set evaluation
y_val_pred = model.predict(X_val_tfidf)
val_accuracy = accuracy_score(y_val, y_val_pred)

print(f"\n{'='*70}")
print(f"  RESULTS")
print(f"{'='*70}")
print(f"\n📈 Accuracy Scores:")
print(f"   Training Set:   {train_accuracy*100:.2f}%")
print(f"   Validation Set: {val_accuracy*100:.2f}%")
print(f"   Test Set:       {test_accuracy*100:.2f}%")

if test_accuracy >= 0.90:
    print(f"\n✅ TARGET ACHIEVED! Accuracy ≥ 90%")
elif test_accuracy >= 0.80:
    print(f"\n✅ Good! Accuracy ≥ 80%")
elif test_accuracy >= 0.70:
    print(f"\n⚠️  Decent. Accuracy ≥ 70%")
else:
    print(f"\n⚠️  Below target. Consider more data or tuning.")

print(f"\n📋 Detailed Classification Report:")
print(classification_report(
    y_test, y_test_pred,
    target_names=data_manager.emotion_labels,
    zero_division=0
))

# Confusion matrix
print(f"\n🔲 Confusion Matrix:")
cm = confusion_matrix(y_test, y_test_pred)
print("       ", "  ".join([f"{e[:4]:>4s}" for e in data_manager.emotion_labels]))
for i, emotion in enumerate(data_manager.emotion_labels):
    print(f"{emotion[:4]:>4s}: ", "  ".join([f"{cm[i][j]:4d}" for j in range(len(data_manager.emotion_labels))]))

# Step 8: Save model
print(f"\n💾 Step 8: Saving model...")
os.makedirs('backend/models', exist_ok=True)

model.save_model('backend/models/pretrained_model.pkl')
preprocessor.save_preprocessor('backend/models/pretrained_preprocessor.pkl')

metadata = {
    'emotion_labels': data_manager.emotion_labels,
    'input_size': input_size,
    'output_size': output_size,
    'test_accuracy': float(test_accuracy),
    'val_accuracy': float(val_accuracy),
    'train_accuracy': float(train_accuracy),
    'training_samples': len(X_train),
    'total_samples': len(df),
    'architecture': {
        'hidden_layers': [512, 256, 128],
        'activation': 'relu',
        'learning_rate': 0.005,
        'epochs': 200,
        'max_features': 5000
    }
}

with open('backend/models/pretrained_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Model saved!")
print(f"   - backend/models/pretrained_model.pkl")
print(f"   - backend/models/pretrained_preprocessor.pkl")
print(f"   - backend/models/pretrained_metadata.json")

print(f"\n{'='*70}")
print(f"✅ TRAINING COMPLETE!")
print(f"{'='*70}")
print(f"\n📊 Final Test Accuracy: {test_accuracy*100:.2f}%")
print(f"\n🔄 Next Steps:")
print(f"   1. Restart backend server:")
print(f"      - Stop: Ctrl+C")
print(f"      - Start: python backend/main.py")
print(f"   2. Test prediksi di aplikasi web")
print(f"   3. Enjoy accurate predictions! 🎉")
