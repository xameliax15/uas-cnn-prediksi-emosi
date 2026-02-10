"""
Script untuk mengkonversi dataset yang gagal di-load
- TSV files (tab-separated)
- Files dengan nama kolom berbeda
"""

import pandas as pd
import os
import glob

print("="*70)
print("  Dataset Format Converter")
print("="*70)

datasets_folder = 'datasets'
converted_folder = os.path.join(datasets_folder, 'converted')
os.makedirs(converted_folder, exist_ok=True)

# List file yang mau dikonversi
all_files = glob.glob(os.path.join(datasets_folder, '*.csv'))

print(f"\n📂 Found {len(all_files)} files in {datasets_folder}/\n")

converted_count = 0
failed_count = 0

for filepath in all_files:
    filename = os.path.basename(filepath)
    
    # Skip files yang sudah converted atau merged
    if 'converted' in filename.lower() or 'merged' in filename.lower() or 'template' in filename.lower():
        continue
    
    print(f"📄 Processing: {filename}")
    
    try:
        # Try different separators and encodings
        loaded = False
        df = None
        
        # Try 1: CSV comma-separated UTF-8
        try:
            df = pd.read_csv(filepath, encoding='utf-8', sep=',')
            if len(df.columns) > 1:
                print(f"   ✅ Loaded as CSV (comma)")
                loaded = True
        except:
            pass
        
        # Try 2: TSV tab-separated UTF-8
        if not loaded:
            try:
                df = pd.read_csv(filepath, encoding='utf-8', sep='\t')
                if len(df.columns) > 1:
                    print(f"   ✅ Loaded as TSV (tab)")
                    loaded = True
            except:
                pass
        
        # Try 3: CSV with latin-1 encoding
        if not loaded:
            try:
                df = pd.read_csv(filepath, encoding='latin-1', sep=',')
                if len(df.columns) > 1:
                    print(f"   ✅ Loaded with latin-1 encoding")
                    loaded = True
            except:
                pass
        
        # Try 4: TSV with latin-1 encoding
        if not loaded:
            try:
                df = pd.read_csv(filepath, encoding='latin-1', sep='\t')
                if len(df.columns) > 1:
                    print(f"   ✅ Loaded as TSV with latin-1")
                    loaded = True
            except:
                pass
        
        if not loaded or df is None:
            print(f"   ❌ Failed to load")
            failed_count += 1
            continue
        
        print(f"   📊 Shape: {df.shape}")
        print(f"   🔤 Columns: {df.columns.tolist()}")
        
        # Detect text and emotion columns
        text_col = None
        emotion_col = None
        
        # Common text column names
        text_keywords = ['text', 'tweet', 'content', 'message', 'sentence', 'data']
        # Common emotion column names  
        emotion_keywords = ['emotion', 'label', 'sentiment', 'class', 'feeling']
        
        for col in df.columns:
            col_lower = str(col).lower().strip()
            
            # Check for text column
            if text_col is None:
                for keyword in text_keywords:
                    if keyword in col_lower:
                        text_col = col
                        break
            
            # Check for emotion column
            if emotion_col is None:
                for keyword in emotion_keywords:
                    if keyword in col_lower:
                        emotion_col = col
                        break
        
        # If not found, use first two columns
        if text_col is None and len(df.columns) >= 1:
            text_col = df.columns[0]
            print(f"   ⚠️  Using first column as text: '{text_col}'")
        
        if emotion_col is None and len(df.columns) >= 2:
            emotion_col = df.columns[1]
            print(f"   ⚠️  Using second column as emotion: '{emotion_col}'")
        
        if text_col is None or emotion_col is None:
            print(f"   ❌ Cannot identify text/emotion columns")
            failed_count += 1
            continue
        
        print(f"   🎯 Mapped: text='{text_col}', emotion='{emotion_col}'")
        
        # Create clean dataframe
        df_clean = pd.DataFrame()
        df_clean['text'] = df[text_col].astype(str)
        df_clean['emotion'] = df[emotion_col].astype(str)
        
        # Remove null and empty
        before = len(df_clean)
        df_clean = df_clean.dropna()
        df_clean = df_clean[df_clean['text'].str.strip() != '']
        df_clean = df_clean[df_clean['emotion'].str.strip() != '']
        after = len(df_clean)
        
        if before != after:
            print(f"   🧹 Cleaned: {before} → {after} rows")
        
        if len(df_clean) == 0:
            print(f"   ❌ No valid data after cleaning")
            failed_count += 1
            continue
        
        # Map common emotion labels to Indonesian
        emotion_mapping = {
            'anger': 'marah', 'Anger': 'marah', 'angry': 'marah', 'ANGER': 'marah',
            'fear': 'takut', 'Fear': 'takut', 'scared': 'takut', 'FEAR': 'takut',
            'joy': 'senang', 'Joy': 'senang', 'happy': 'senang', 'JOY': 'senang',
            'happiness': 'senang', 'Happy': 'senang',
            'love': 'senang', 'Love': 'senang', 'LOVE': 'senang',
            'sadness': 'sedih', 'Sadness': 'sedih', 'sad': 'sedih', 'SAD': 'sedih',
            'disgust': 'jijik', 'Disgust': 'jijik', 'DISGUST': 'jijik',
            'surprise': 'terkejut', 'Surprise': 'terkejut', 'SURPRISE': 'terkejut',
            'neutral': 'netral', 'Neutral': 'netral', 'NEUTRAL': 'netral',
        }
        
        # Apply mapping if possible
        df_clean['emotion'] = df_clean['emotion'].replace(emotion_mapping)
        
        # Show emotion distribution
        print(f"   📈 Emotions: {df_clean['emotion'].unique().tolist()[:10]}")
        
        # Save converted file
        output_filename = filename.replace('.csv', '_converted.csv')
        output_path = os.path.join(converted_folder, output_filename)
        df_clean.to_csv(output_path, index=False, encoding='utf-8')
        
        print(f"   💾 Saved: converted/{output_filename}")
        print(f"   ✅ Success: {len(df_clean)} rows\n")
        
        converted_count += 1
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)[:100]}\n")
        failed_count += 1

print("="*70)
print(f"✅ Conversion Complete!")
print("="*70)
print(f"   Converted: {converted_count} files")
print(f"   Failed: {failed_count} files")
print(f"\n📂 Converted files saved to: {converted_folder}/")

if converted_count > 0:
    print(f"\n💡 Next steps:")
    print(f"   1. Review converted files in {converted_folder}/")
    print(f"   2. Move valid files to {datasets_folder}/")
    print(f"   3. Re-run: python train_model.py")
