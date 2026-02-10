import pandas as pd
import numpy as np
import random
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))
from backend.ann_model import ANNClassifier
from backend.text_preprocessing import IndonesianTextPreprocessor
from backend.data_manager import DataManager
import json

# Configuration
TARGET_COUNT = 1000  # Target samples per class
MINORITY_CLASSES = ['jijik', 'sedih', 'terkejut']

# Indonesian Synonym Dictionary (Simple)
synonyms = {
    'sedih': ['pilu', 'duka', 'pedih', 'menderita', 'sakit hati', 'kecewa', 'murung', 'nelangsa'],
    'senang': ['bahagia', 'gembira', 'sukacita', 'riang', 'ceria', 'puas', 'bangga'],
    'marah': ['kesal', 'benci', 'jengkel', 'murka', 'emosi', 'geram', 'sebal', 'ngamuk'],
    'takut': ['ngeri', 'seram', 'cemas', 'khawatir', 'deg-degan', 'parno', 'gemetar'],
    'jijik': ['muak', 'enek', 'illfeel', 'geli', 'mual', 'benci banget', 'najis'],
    'terkejut': ['kaget', 'shock', 'heran', 'takjub', 'melongo', 'tidak menyangka'],
    'banget': ['sekali', 'sangat', 'benar-benar', 'parah', 'kali'],
    'aku': ['saya', 'gue', 'gw', 'beta', 'hamba'],
    'kamu': ['anda', 'lu', 'kau', 'situ'],
    'tidak': ['gak', 'nggak', 'tak', 'kaga'],
    'bisa': ['dapat', 'mampu', 'bisa-bisa'],
}

# Augmentation Templates
templates = [
    "sumpah {text}",
    "asli {text}",
    "{text} banget",
    "gila {text}",
    "wah {text}",
    "aduh {text}",
    "pokoknya {text}",
    "rasanya {text}",
    "{text} sih",
    "{text} deh",
    "benar-benar {text}",
    "sangat {text}"
]

def replace_synonyms(text):
    words = text.split()
    new_words = []
    changed = False
    for word in words:
        lower_word = word.lower()
        if lower_word in synonyms and random.random() < 0.3: # 30% chance to replace
            new_words.append(random.choice(synonyms[lower_word]))
            changed = True
        else:
            # Check reverse mapping (if value is key) - skipped for simplicity
            # Check general synonyms
            found = False
            for key, values in synonyms.items():
                if lower_word in values:
                    new_words.append(key)
                    found = True
                    changed = True
                    break
            if not found:
                new_words.append(word)
    
    if not changed and len(words) > 2:
        # Swap two random words if no synonym replaced
        idx1, idx2 = random.sample(range(len(words)), 2)
        new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
        
    return " ".join(new_words)

def augment_text(text, emotion):
    # Strategy 1: Synonym Replacement
    aug1 = replace_synonyms(text)
    
    # Strategy 2: Template Injection
    template = random.choice(templates)
    aug2 = template.format(text=text)
    
    return [aug1, aug2]

print("="*70)
print("  DATA AUGMENTATION & RETRAINING")
print("="*70)

# 1. Load Data
print("\nLoading dataset...")
df = pd.read_csv('datasets/full_merged_dataset_clean.csv')
print(f"Original shape: {df.shape}")
print(df['emotion'].value_counts())

# 2. Augment Data
new_rows = []
print("\nAugmenting minority classes...")

for emotion in MINORITY_CLASSES:
    current_count = len(df[df['emotion'] == emotion])
    needed = TARGET_COUNT - current_count
    
    if needed > 0:
        print(f"  Augmenting {emotion}: +{needed} samples needed")
        samples = df[df['emotion'] == emotion]['text'].tolist()
        
        generated = 0
        while generated < needed:
            text = random.choice(samples)
            new_texts = augment_text(text, emotion)
            
            for nt in new_texts:
                if generated < needed:
                    new_rows.append({'text': nt, 'emotion': emotion})
                    generated += 1
                else:
                    break

if new_rows:
    df_aug = pd.DataFrame(new_rows)
    df_combined = pd.concat([df, df_aug], ignore_index=True)
    print(f"\nAugmented {len(new_rows)} samples.")
    print(f"New shape: {df_combined.shape}")
    print(df_combined['emotion'].value_counts())
    
    # Save
    df_combined.to_csv('datasets/augmented_dataset.csv', index=False)
    print("Saved to datasets/augmented_dataset.csv")
else:
    df_combined = df
    print("No augmentation needed or performed.")

# 3. Validation
print("\nValidating augmentation...")
# Remove duplicates
df_combined.drop_duplicates(subset=['text'], inplace=True)
print(f"Shape after duplicate removal: {df_combined.shape}")

# 4. Training
print("\n🚀 Starting Training with Augmented Data...")

data_manager = DataManager()
data_manager.df = df_combined
data_manager.emotion_labels = sorted(df_combined['emotion'].unique().tolist())

X_train, X_val, X_test, y_train, y_val, y_test = data_manager.split_data(
    test_size=0.15, val_size=0.15, random_state=42
)

# Use Unigram + Bigram (ngram_range=(1,2)) for better context
preprocessor = IndonesianTextPreprocessor(max_features=3000, use_stemming=False, use_stopword_removal=True)

print("Vectorizing (TF-IDF)...")
X_train_tfidf = preprocessor.fit_transform(X_train)
X_val_tfidf = preprocessor.transform(X_val)
X_test_tfidf = preprocessor.transform(X_test)

y_train_onehot = data_manager.one_hot_encode(y_train)
y_val_onehot = data_manager.one_hot_encode(y_val)
y_test_onehot = data_manager.one_hot_encode(y_test)

# Model: Slightly larger architecture
model = ANNClassifier(
    input_size=X_train_tfidf.shape[1],
    hidden_layers=[512, 256, 128], # Deeper network
    output_size=len(data_manager.emotion_labels),
    learning_rate=0.01, # Increased LR for faster convergence
    activation='relu'
)

print(f"Training {len(X_train)} samples, {len(data_manager.emotion_labels)} emotions...")
history = model.fit(
    X_train_tfidf, y_train_onehot, 
    X_val_tfidf, y_val_onehot, 
    epochs=100, # 100 epochs is usually enough with augmentation
    batch_size=32, 
    verbose=True
)

# Evaluation
from sklearn.metrics import accuracy_score, classification_report
y_test_pred = model.predict(X_test_tfidf)
y_test_pred_labels = [data_manager.emotion_labels[i] for i in y_test_pred]
test_acc = accuracy_score(y_test, y_test_pred_labels) # ANNClassifier predict returns indices? No, wait. 
# Correction: ANNClassifier.predict returns indices. data_manager.decode_predictions converts to list of labels? 
# Check ann_model.py: predict returns indices (argmax).
# data_manager.one_hot_encode returns matrix.
# y_test is pandas Series of strings.

# Correct evaluation logic:
y_test_indices = [data_manager.emotion_labels.index(label) for label in y_test]
test_acc = accuracy_score(y_test_indices, y_test_pred)

print(f"\n✅ Test Accuracy: {test_acc*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test_indices, y_test_pred, target_names=data_manager.emotion_labels))

# Save
print("\nSaving model...")
os.makedirs('backend/models', exist_ok=True)
model.save_model('backend/models/pretrained_model.pkl')
preprocessor.save_preprocessor('backend/models/pretrained_preprocessor.pkl')

metadata = {
    'emotion_labels': data_manager.emotion_labels,
    'input_size': X_train_tfidf.shape[1],
    'output_size': len(data_manager.emotion_labels),
    'test_accuracy': float(test_acc),
    'total_samples': len(df_combined),
    'architecture': {'hidden_layers': [512, 256, 128], 'activation': 'relu', 'learning_rate': 0.001, 'epochs': 100},
    'augmented': True
}

with open('backend/models/pretrained_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n✅ Augmented Model saved! Restart backend to apply.")
