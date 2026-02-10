"""
Text Preprocessing untuk Teks Bahasa Indonesia
"""

import re
import pandas as pd
import numpy as np
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


class IndonesianTextPreprocessor:
    """
    Preprocessor untuk teks media sosial berbahasa Indonesia
    """
    
    def __init__(self, max_features=1000, use_stemming=True, use_stopword_removal=True):
        """
        Initialize preprocessor
        
        Args:
            max_features: Maksimum fitur untuk TF-IDF vectorization
            use_stemming: Gunakan stemming atau tidak
            use_stopword_removal: Gunakan stopword removal atau tidak
        """
        self.max_features = max_features
        self.use_stemming = use_stemming
        self.use_stopword_removal = use_stopword_removal
        
        # Initialize Sastrawi
        if use_stemming:
            factory = StemmerFactory()
            self.stemmer = factory.create_stemmer()
        
        if use_stopword_removal:
            factory = StopWordRemoverFactory()
            self.stopword_remover = factory.create_stop_word_remover()
        
        # TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(max_features=max_features, lowercase=True)
        
        self.is_fitted = False
    
    def clean_text(self, text):
        """
        Clean raw text
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions (@username)
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags (keep the word)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove angka
        text = re.sub(r'\d+', '', text)
        
        # Remove special characters and punctuation (keep only letters and spaces)
        text = re.sub(r'[^a-z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def normalize_slang(self, text):
        """
        Normalize Indonesian slang words (basic normalization)
        Bisa diperluas dengan dictionary yang lebih lengkap
        """
        slang_dict = {
            'gak': 'tidak',
            'ga': 'tidak',
            'ngga': 'tidak',
            'nggak': 'tidak',
            'gue': 'saya',
            'gw': 'saya',
            'lo': 'kamu',
            'lu': 'kamu',
            'udah': 'sudah',
            'udh': 'sudah',
            'blm': 'belum',
            'blom': 'belum',
            'bgt': 'banget',
            'banget': 'sangat',
            'bener': 'benar',
            'emang': 'memang',
            'emg': 'memang',
            'gmn': 'bagaimana',
            'gimana': 'bagaimana',
            'knp': 'kenapa',
            'knapa': 'kenapa',
            'sm': 'sama',
            'dgn': 'dengan',
            'tdk': 'tidak',
            'yg': 'yang',
            'krn': 'karena',
            'utk': 'untuk',
            'sdh': 'sudah',
            'pd': 'pada',
            'trs': 'terus',
            'tau': 'tahu',
            'dong': '',
            'sih': '',
            'kok': '',
            'deh': '',
            'nih': '',
            'wkwk': '',
            'haha': '',
            'hehe': '',
            'wkwkwk': '',
            'hahaha': '',
            'waw': 'wow',
        }
        
        words = text.split()
        normalized_words = [slang_dict.get(word, word) for word in words]
        return ' '.join(normalized_words)
    
    def preprocess_text(self, text):
        """
        Complete preprocessing pipeline for single text
        
        Args:
            text: Raw text
            
        Returns:
            Preprocessed text
        """
        # Clean text
        text = self.clean_text(text)
        
        # Normalize slang
        text = self.normalize_slang(text)
        
        # Remove stopwords
        if self.use_stopword_removal and text:
            text = self.stopword_remover.remove(text)
        
        # Stemming
        if self.use_stemming and text:
            text = self.stemmer.stem(text)
        
        return text
    
    def fit_transform(self, texts):
        """
        Fit vectorizer and transform texts to TF-IDF features
        
        Args:
            texts: List of raw text strings
            
        Returns:
            TF-IDF feature matrix
        """
        # Preprocess all texts
        preprocessed_texts = [self.preprocess_text(text) for text in texts]
        
        # Fit and transform TF-IDF
        tfidf_matrix = self.vectorizer.fit_transform(preprocessed_texts)
        
        self.is_fitted = True
        
        return tfidf_matrix.toarray()
    
    def transform(self, texts):
        """
        Transform texts to TF-IDF features (must call fit_transform first)
        
        Args:
            texts: List of raw text strings
            
        Returns:
            TF-IDF feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer belum di-fit. Gunakan fit_transform terlebih dahulu.")
        
        # Preprocess all texts
        preprocessed_texts = [self.preprocess_text(text) for text in texts]
        
        # Transform to TF-IDF
        tfidf_matrix = self.vectorizer.transform(preprocessed_texts)
        
        return tfidf_matrix.toarray()
    
    def get_feature_names(self):
        """
        Get TF-IDF feature names (words)
        """
        if not self.is_fitted:
            return []
        
        return self.vectorizer.get_feature_names_out().tolist()
    
    def get_vocabulary_size(self):
        """
        Get vocabulary size
        """
        if not self.is_fitted:
            return 0
        
        return len(self.vectorizer.vocabulary_)
    
    def save_preprocessor(self, filepath):
        """
        Save preprocessor to file
        """
        preprocessor_data = {
            'max_features': self.max_features,
            'use_stemming': self.use_stemming,
            'use_stopword_removal': self.use_stopword_removal,
            'vectorizer': self.vectorizer,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(preprocessor_data, f)
    
    def load_preprocessor(self, filepath):
        """
        Load preprocessor from file
        """
        with open(filepath, 'rb') as f:
            preprocessor_data = pickle.load(f)
        
        self.max_features = preprocessor_data['max_features']
        self.use_stemming = preprocessor_data['use_stemming']
        self.use_stopword_removal = preprocessor_data['use_stopword_removal']
        self.vectorizer = preprocessor_data['vectorizer']
        self.is_fitted = preprocessor_data['is_fitted']
        
        # Re-initialize Sastrawi components
        if self.use_stemming:
            factory = StemmerFactory()
            self.stemmer = factory.create_stemmer()
        
        if self.use_stopword_removal:
            factory = StopWordRemoverFactory()
            self.stopword_remover = factory.create_stop_word_remover()


def analyze_text_statistics(texts, labels=None):
    """
    Analyze text statistics
    
    Args:
        texts: List of texts
        labels: List of labels (optional)
        
    Returns:
        Dictionary of statistics
    """
    stats = {}
    
    # Basic statistics
    stats['total_texts'] = len(texts)
    stats['avg_length'] = np.mean([len(text) for text in texts])
    stats['min_length'] = min([len(text) for text in texts])
    stats['max_length'] = max([len(text) for text in texts])
    
    # Word count statistics
    word_counts = [len(text.split()) for text in texts]
    stats['avg_word_count'] = np.mean(word_counts)
    stats['min_word_count'] = min(word_counts)
    stats['max_word_count'] = max(word_counts)
    
    # Label distribution
    if labels is not None:
        unique, counts = np.unique(labels, return_counts=True)
        stats['label_distribution'] = dict(zip(unique.tolist(), counts.tolist()))
    
    return stats
