"""
Script untuk download dan konversi dataset emosi Indonesia
dari sumber publik (GitHub)
"""

import pandas as pd
import requests
import os
from io import StringIO

def download_github_csv(url, output_path):
    """
    Download CSV file dari GitHub
    """
    print(f"Downloading from: {url}")
    response = requests.get(url)
    
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f"✅ Downloaded to: {output_path}")
        return True
    else:
        print(f"❌ Failed to download. Status code: {response.status_code}")
        return False


def download_indonesian_twitter_emotion_dataset():
    """
    Download dataset dari GitHub: Indonesian Twitter Emotion Dataset
    4,403 tweets dengan 5 emosi: love, anger, sadness, joy, fear
    """
    print("\n" + "="*60)
    print("📥 Downloading Indonesian Twitter Emotion Dataset")
    print("="*60)
    
    # URL raw GitHub
    url = "https://raw.githubusercontent.com/meisaputri21/Indonesian-Twitter-Emotion-Dataset/master/dataset.csv"
    output_path = "datasets/raw_twitter_emotion.csv"
    
    os.makedirs("datasets", exist_ok=True)
    
    if download_github_csv(url, output_path):
        df = pd.read_csv(output_path, sep=';')
        print(f"\n📊 Dataset Info:")
        print(f"   Total samples: {len(df)}")
        print(f"\n📈 Label distribution:")
        print(df['label'].value_counts())
        return df
    
    return None


def download_public_opinion_emotion_dataset():
    """
    Download dataset dari GitHub: Emotion Dataset from Indonesian Public Opinion
    7,080 tweets dengan 6 emosi: anger, fear, joy, love, sad, neutral
    """
    print("\n" + "="*60)
    print("📥 Downloading Public Opinion Emotion Dataset")
    print("="*60)
    
    # URL raw GitHub
    url = "https://raw.githubusercontent.com/Ricco48/Emotion-Dataset-from-Indonesian-Public-Opinion/main/data.csv"
    output_path = "datasets/raw_public_opinion.csv"
    
    os.makedirs("datasets", exist_ok=True)
    
    if download_github_csv(url, output_path):
        df = pd.read_csv(output_path)
        print(f"\n📊 Dataset Info:")
        print(f"   Total samples: {len(df)}")
        print(f"\n📈 Label distribution:")
        print(df.iloc[:, 1].value_counts())  # Assuming second column is label
        return df
    
    return None


def map_emotions_to_7_categories(df, source_column='label', text_column='text'):
    """
    Mapping emosi dari berbagai dataset ke 7 kategori standar:
    - marah (anger)
    - jijik (disgust)
    - takut (fear)
    - senang (joy/love/happy)
    - sedih (sadness)
    - terkejut (surprise)
    - netral (neutral)
    """
    
    # Mapping dictionary
    emotion_mapping = {
        # English to Indonesian
        'anger': 'marah',
        'fear': 'takut',
        'joy': 'senang',
        'happy': 'senang',
        'happiness': 'senang',
        'love': 'senang',  # Love biasanya positif, masuk ke senang
        'sadness': 'sedih',
        'sad': 'sedih',
        'disgust': 'jijik',
        'surprise': 'terkejut',
        'neutral': 'netral',
        
        # Case variations
        'Anger': 'marah',
        'Fear': 'takut',
        'Joy': 'senang',
        'Happy': 'senang',
        'Love': 'senang',
        'Sadness': 'sedih',
        'Sad': 'sedih',
        'Disgust': 'jijik',
        'Surprise': 'terkejut',
        'Neutral': 'netral',
    }
    
    # Create new dataframe
    df_mapped = pd.DataFrame()
    df_mapped['text'] = df[text_column]
    df_mapped['emotion'] = df[source_column].map(emotion_mapping)
    
    # Remove unmapped emotions
    df_mapped = df_mapped.dropna(subset=['emotion'])
    
    print(f"\n🔄 Emotion Mapping Results:")
    print(f"   Original samples: {len(df)}")
    print(f"   Mapped samples: {len(df_mapped)}")
    print(f"   Dropped: {len(df) - len(df_mapped)}")
    print(f"\n📊 New distribution:")
    print(df_mapped['emotion'].value_counts())
    
    return df_mapped


def create_combined_dataset():
    """
    Download dan gabungkan semua dataset
    """
    print("\n" + "="*60)
    print("🎯 Creating Combined Indonesian Emotion Dataset")
    print("="*60)
    
    all_datasets = []
    
    # Dataset 1: Twitter Emotion
    df1 = download_indonesian_twitter_emotion_dataset()
    if df1 is not None:
        # Cek nama kolom
        print(f"\n📝 Columns in Twitter dataset: {df1.columns.tolist()}")
        
        # Adjust column names berdasarkan format sebenarnya
        if 'tweet' in df1.columns and 'label' in df1.columns:
            df1_mapped = map_emotions_to_7_categories(df1, source_column='label', text_column='tweet')
        elif len(df1.columns) >= 2:
            # Use first two columns
            df1_mapped = map_emotions_to_7_categories(df1, source_column=df1.columns[1], text_column=df1.columns[0])
        
        all_datasets.append(df1_mapped)
    
    # Dataset 2: Public Opinion
    try:
        df2 = download_public_opinion_emotion_dataset()
        if df2 is not None:
            print(f"\n📝 Columns in Public Opinion dataset: {df2.columns.tolist()}")
            
            # Adjust based on actual format
            if 'text' in df2.columns and 'emotion' in df2.columns:
                df2_mapped = map_emotions_to_7_categories(df2, source_column='emotion', text_column='text')
            elif len(df2.columns) >= 2:
                df2_mapped = map_emotions_to_7_categories(df2, source_column=df2.columns[1], text_column=df2.columns[0])
            
            all_datasets.append(df2_mapped)
    except Exception as e:
        print(f"⚠️  Could not download Public Opinion dataset: {e}")
    
    # Combine all datasets
    if all_datasets:
        combined_df = pd.concat(all_datasets, ignore_index=True)
        
        # Remove duplicates
        original_count = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=['text'])
        print(f"\n🔍 Removed {original_count - len(combined_df)} duplicate texts")
        
        # Shuffle
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save
        output_path = "datasets/indonesian_emotion_7categories.csv"
        combined_df.to_csv(output_path, index=False)
        
        print(f"\n" + "="*60)
        print(f"✅ COMBINED DATASET CREATED")
        print(f"="*60)
        print(f"📂 Saved to: {output_path}")
        print(f"📊 Total samples: {len(combined_df)}")
        print(f"\n📈 Final distribution:")
        print(combined_df['emotion'].value_counts())
        
        # Check which emotions are missing
        target_emotions = {'marah', 'jijik', 'takut', 'senang', 'sedih', 'terkejut', 'netral'}
        available_emotions = set(combined_df['emotion'].unique())
        missing_emotions = target_emotions - available_emotions
        
        if missing_emotions:
            print(f"\n⚠️  Missing emotions: {missing_emotions}")
            print(f"   Note: Anda mungkin perlu menambahkan data untuk emosi ini secara manual")
        
        return combined_df
    
    return None


def create_sample_for_missing_emotions(df, target_count=50):
    """
    Buat sample data untuk emosi yang hilang (jijik, terkejut)
    """
    print("\n" + "="*60)
    print("📝 Creating Sample Data for Missing Emotions")
    print("="*60)
    
    # Sample data untuk jijik
    jijik_samples = [
        "Menjijikkan sekali perilaku seperti itu",
        "Ih najis banget deh enggak suka",
        "Jorok banget ya ampun bikin ilfeel",
        "Geli dan jijik lihat yang begitu",
        "Enek banget deh lihatnya",
        "Bau dan menjijikkan sekali",
        "Mual lihat pemandangan seperti itu",
        "Menjijikkan tidak bisa ditolerir",
        "Kotor dan menjijikkan",
        "Ih gross banget",
        "Jorok ih menjijikkan banget",
        "Muak banget ngeliatnya",
        "Jijay banget deh",
        "Gimana sih kotor banget",
        "Ih serem jorok banget",
    ]
    
    # Sample data untuk terkejut
    terkejut_samples = [
        "Wow! Saya tidak menyangka hal ini",
        "Astaga! Sungguh mengejutkan",
        "Lho kok bisa? Tidak terduga sama sekali",
        "Ya ampun kaget! Mendadak banget",
        "Wah gila! Beneran kejadian",
        "Astagfirullah kaget banget tiba-tiba",
        "Oh my god! Tidak percaya ini nyata",
        "Heboh banget kabar mengejutkan ini",
        "Kagum sekaligus terkejut",
        "Wah gak nyangka banget",
        "Kaget poll mendadak gini",
        "Surprise banget sih ini",
        "Unexpected banget deh",
        "Shocking news banget",
        "Tiba-tiba banget bikin kaget",
    ]
    
    # Create DataFrames
    df_jijik = pd.DataFrame({
        'text': jijik_samples,
        'emotion': 'jijik'
    })
    
    df_terkejut = pd.DataFrame({
        'text': terkejut_samples,
        'emotion': 'terkejut'
    })
    
    # Combine with main dataset
    df_enhanced = pd.concat([df, df_jijik, df_terkejut], ignore_index=True)
    df_enhanced = df_enhanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"✅ Added {len(jijik_samples)} samples for 'jijik'")
    print(f"✅ Added {len(terkejut_samples)} samples for 'terkejut'")
    print(f"\n📊 New total: {len(df_enhanced)} samples")
    print(f"\n📈 Updated distribution:")
    print(df_enhanced['emotion'].value_counts())
    
    # Save enhanced dataset
    output_path = "datasets/indonesian_emotion_7categories_enhanced.csv"
    df_enhanced.to_csv(output_path, index=False)
    print(f"\n💾 Saved enhanced dataset to: {output_path}")
    
    return df_enhanced


if __name__ == "__main__":
    print("\n" + "🚀 "*30)
    print("Indonesian Emotion Dataset Downloader & Converter")
    print("🚀 "*30)
    
    # Download and combine datasets
    df = create_combined_dataset()
    
    if df is not None:
        # Enhance with missing emotions
        df_final = create_sample_for_missing_emotions(df)
        
        print("\n" + "="*60)
        print("✅ DATASET READY!")
        print("="*60)
        print(f"📂 Final dataset: datasets/indonesian_emotion_7categories_enhanced.csv")
        print(f"📊 Total samples: {len(df_final)}")
        print(f"\n🎯 You can now upload this dataset to the web application!")
        print("="*60)
    else:
        print("\n❌ Failed to create dataset. Please check your internet connection.")
