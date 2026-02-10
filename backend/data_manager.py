"""
Data Management Utilities
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json
import os


class DataManager:
    """
    Manager untuk dataset klasifikasi emosi
    """
    
    def __init__(self):
        self.df = None
        self.label_encoder = LabelEncoder()
        self.emotion_labels = []
        self.is_encoded = False
    
    def load_dataset(self, filepath, text_column='text', label_column='emotion'):
        """
        Load dataset from file (CSV, Excel, or JSON)
        
        Args:
            filepath: Path to dataset file
            text_column: Name of text column
            label_column: Name of label/emotion column
            
        Returns:
            Success status and message
        """
        try:
            # Detect file type
            _, ext = os.path.splitext(filepath)
            
            if ext.lower() in ['.csv', '.txt']:
                self.df = pd.read_csv(filepath)
            elif ext.lower() in ['.xlsx', '.xls']:
                self.df = pd.read_excel(filepath)
            elif ext.lower() == '.json':
                self.df = pd.read_json(filepath)
            else:
                return False, f"Format file tidak didukung: {ext}"
            
            # Validate columns
            if text_column not in self.df.columns:
                return False, f"Kolom '{text_column}' tidak ditemukan"
            
            if label_column not in self.df.columns:
                return False, f"Kolom '{label_column}' tidak ditemukan"
            
            # Rename columns to standard names
            self.df = self.df.rename(columns={text_column: 'text', label_column: 'emotion'})
            
            # Remove rows with missing values
            original_count = len(self.df)
            self.df = self.df.dropna(subset=['text', 'emotion'])
            removed_count = original_count - len(self.df)
            
            if len(self.df) == 0:
                return False, "Dataset kosong setelah menghapus nilai yang hilang"
            
            # Get unique emotions
            self.emotion_labels = sorted(self.df['emotion'].unique().tolist())
            
            message = f"Dataset berhasil dimuat: {len(self.df)} data"
            if removed_count > 0:
                message += f" ({removed_count} baris dengan nilai hilang dihapus)"
            
            return True, message
            
        except Exception as e:
            return False, f"Error saat memuat dataset: {str(e)}"
    
    def get_statistics(self):
        """
        Get dataset statistics
        
        Returns:
            Dictionary of statistics
        """
        if self.df is None:
            return None
        
        stats = {
            'total_samples': int(len(self.df)),  # Convert to native Python int
            'emotion_labels': self.emotion_labels,
            'emotion_distribution': {
                str(k): int(v)  # Convert both key and value to native Python types
                for k, v in self.df['emotion'].value_counts().to_dict().items()
            },
            'avg_text_length': float(self.df['text'].str.len().mean()),
            'min_text_length': int(self.df['text'].str.len().min()),
            'max_text_length': int(self.df['text'].str.len().max()),
        }
        
        return stats
    
    def encode_labels(self):
        """
        Encode emotion labels to integers
        
        Returns:
            Encoded labels
        """
        if self.df is None:
            raise ValueError("Dataset belum dimuat")
        
        encoded = self.label_encoder.fit_transform(self.df['emotion'])
        self.is_encoded = True
        
        return encoded
    
    def decode_labels(self, encoded_labels):
        """
        Decode integer labels to emotion names
        
        Args:
            encoded_labels: Array of encoded labels
            
        Returns:
            Array of emotion names
        """
        if not self.is_encoded:
            raise ValueError("Label encoder belum di-fit")
        
        return self.label_encoder.inverse_transform(encoded_labels)
    
    def one_hot_encode(self, labels):
        """
        One-hot encode labels
        
        Args:
            labels: Array of integer labels
            
        Returns:
            One-hot encoded array
        """
        n_classes = len(self.emotion_labels)
        one_hot = np.zeros((len(labels), n_classes))
        one_hot[np.arange(len(labels)), labels] = 1
        
        return one_hot
    
    def split_data(self, test_size=0.2, val_size=0.1, random_state=42):
        """
        Split dataset into train, validation, and test sets
        
        Args:
            test_size: Proportion of test set
            val_size: Proportion of validation set
            random_state: Random seed
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        if self.df is None:
            raise ValueError("Dataset belum dimuat")
        
        # Get texts and labels
        texts = self.df['text'].values
        labels = self.encode_labels()
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        # Second split: train vs val
        if val_size > 0:
            val_ratio = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
            )
        else:
            X_train, y_train = X_temp, y_temp
            X_val, y_val = None, None
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_label_mapping(self):
        """
        Get label to index mapping
        
        Returns:
            Dictionary mapping emotion names to indices
        """
        if not self.is_encoded:
            return None
        
        return {emotion: idx for idx, emotion in enumerate(self.label_encoder.classes_)}
    
    def export_sample(self, filepath, n_samples=10):
        """
        Export sample data to file (untuk template)
        
        Args:
            filepath: Output filepath
            n_samples: Number of samples to export
        """
        if self.df is None:
            # Create sample template
            sample_data = {
                'text': [
                    'Hari ini sangat menyenangkan sekali!',
                    'Saya merasa sedih dan kecewa',
                    'Ini membuat saya sangat marah',
                    'Saya takut akan hal ini',
                    'Menjijikkan sekali perilaku seperti itu',
                    'Wow! Saya tidak menyangka hal ini',
                    'Biasa saja, tidak ada yang spesial'
                ],
                'emotion': ['senang', 'sedih', 'marah', 'takut', 'jijik', 'terkejut', 'netral']
            }
            df_sample = pd.DataFrame(sample_data)
        else:
            df_sample = self.df.head(n_samples)
        
        # Save based on extension
        _, ext = os.path.splitext(filepath)
        
        if ext.lower() == '.csv':
            df_sample.to_csv(filepath, index=False)
        elif ext.lower() in ['.xlsx', '.xls']:
            df_sample.to_excel(filepath, index=False)
        elif ext.lower() == '.json':
            df_sample.to_json(filepath, orient='records', indent=2)
    
    def validate_prediction_input(self, text):
        """
        Validate input text for prediction
        
        Args:
            text: Input text
            
        Returns:
            is_valid, message
        """
        if not isinstance(text, str):
            return False, "Input harus berupa teks"
        
        if len(text.strip()) == 0:
            return False, "Teks tidak boleh kosong"
        
        if len(text) < 3:
            return False, "Teks terlalu pendek"
        
        return True, "Valid"
