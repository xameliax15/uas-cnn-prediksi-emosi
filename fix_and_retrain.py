"""
Fix dataset - merge 'Sad' into 'sedih' dan re-train
"""
import pandas as pd
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.ann_model import ANNClassifier
from backend.text_preprocessing import IndonesianTextPreprocessor
from backend.data_manager import DataManager
import json

print("="*70)
print("  Fixing Dataset & Re-training")
print("="*70)

# Load dataset
df = pd.read_csv('datasets/full_merged_dataset.csv')
print(f"\nOriginal emotions: {df['emotion'].unique()}")
print(f"Original counts:\n{df['emotion'].value_counts()}")

# Replace 'Sad' with 'sedih'
df['emotion'] = df['emotion'].replace({'Sad': 'sedih'})

# Remove any other non-Indonesian labels
valid_emotions = ['senang', 'sedih', 'marah', 'takut', 'jijik', 'terkejut', 'netral']
df = df[df['emotion'].isin(valid_emotions)]

print(f"\nFixed emotions: {sorted(df['emotion'].unique())}")
print(f"\nFixed counts:")
for emotion in sorted(df['emotion'].unique()):
    count = len(df[df['emotion'] == emotion])
    print(f"  {emotion:10s}: {count}")

# Save cleaned dataset
df.to_csv('datasets/full_merged_dataset_clean.csv', index=False)
print(f"\n✅ Saved: datasets/full_merged_dataset_clean.csv")

# Quick training
print(f"\n🚀 Training with clean dataset...")
data_manager = DataManager()
data_manager.df = df
data_manager.emotion_labels = sorted(df['emotion'].unique().tolist())

X_train, X_val, X_test, y_train, y_val, y_test = data_manager.split_data(
    test_size=0.15, val_size=0.15, random_state=42
)

preprocessor = IndonesianTextPreprocessor(max_features=2000, use_stemming=False, use_stopword_removal=True)
X_train_tfidf = preprocessor.fit_transform(X_train)
X_val_tfidf = preprocessor.transform(X_val)
X_test_tfidf = preprocessor.transform(X_test)

y_train_onehot = data_manager.one_hot_encode(y_train)
y_val_onehot = data_manager.one_hot_encode(y_val)
y_test_onehot = data_manager.one_hot_encode(y_test)

model = ANNClassifier(
    input_size=X_train_tfidf.shape[1],
    hidden_layers=[256, 128],
    output_size=len(data_manager.emotion_labels),
    learning_rate=0.01,
    activation='relu'
)

print(f"Training {len(X_train)} samples, {len(data_manager.emotion_labels)} emotions...")
history = model.fit(X_train_tfidf, y_train_onehot, X_val_tfidf, y_val_onehot, epochs=150, batch_size=64, verbose=True)

from sklearn.metrics import accuracy_score
y_test_pred = model.predict(X_test_tfidf)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"\n✅ Test Accuracy: {test_acc*100:.2f}%")

# Save
os.makedirs('backend/models', exist_ok=True)
model.save_model('backend/models/pretrained_model.pkl')
preprocessor.save_preprocessor('backend/models/pretrained_preprocessor.pkl')

metadata = {
    'emotion_labels': data_manager.emotion_labels,
    'input_size': X_train_tfidf.shape[1],
    'output_size': len(data_manager.emotion_labels),
    'test_accuracy': float(test_acc),
    'total_samples': len(df),
    'architecture': {'hidden_layers': [256, 128], 'activation': 'relu', 'learning_rate': 0.01, 'epochs': 150}
}

with open('backend/models/pretrained_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n✅ Model saved! Restart backend untuk load model baru.")
print(f"Emotions: {data_manager.emotion_labels}")
