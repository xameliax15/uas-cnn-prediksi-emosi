"""
Flask API for Emotion Classification Application
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import json
import numpy as np
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from ann_model import ANNClassifier
from text_preprocessing import IndonesianTextPreprocessor, analyze_text_statistics
from data_manager import DataManager

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
MODEL_FOLDER = os.path.join(BASE_DIR, 'models')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Global variables
data_manager = DataManager()
preprocessor = None
model = None
current_dataset_stats = None

# Auto-load pretrained model if exists
def load_pretrained_on_startup():
    global model, preprocessor, data_manager
    
    model_path = os.path.join(MODEL_FOLDER, 'pretrained_model.pkl')
    preprocessor_path = os.path.join(MODEL_FOLDER, 'pretrained_preprocessor.pkl')  
    metadata_path = os.path.join(MODEL_FOLDER, 'pretrained_metadata.json')
    
    if os.path.exists(model_path) and os.path.exists(preprocessor_path):
        try:
            print("\n🔄 Loading pretrained model...")
            
            # Load model
            model = ANNClassifier(input_size=1, hidden_layers=[1], output_size=1)
            model.load_model(model_path)
            
            # Load preprocessor
            preprocessor = IndonesianTextPreprocessor()
            preprocessor.load_preprocessor(preprocessor_path)
            
            # Load metadata if exists
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    data_manager.emotion_labels = metadata.get('emotion_labels', [])
                    print(f"✅ Pretrained model loaded!")
                    print(f"   Accuracy: {metadata.get('test_accuracy', 0)*100:.2f}%")
                    print(f"   Emotions: {data_manager.emotion_labels}")
            else:
                print("✅ Model & preprocessor loaded")
                
        except Exception as e:
            print(f"⚠️  Failed to load pretrained model: {str(e)}")

# Load pretrained model
load_pretrained_on_startup()


@app.route('/')
def index():
    """API status"""
    return jsonify({
        'status': 'running',
        'message': 'Emotion Classification API',
        'version': '1.0.0'
    })


@app.route('/api/upload', methods=['POST'])
def upload_dataset():
    """
    Upload dataset file
    """
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'Tidak ada file yang diupload'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'success': False, 'message': 'Nama file kosong'}), 400
        
        # Save file
        filename = f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Load dataset
        text_column = request.form.get('text_column', 'text')
        label_column = request.form.get('label_column', 'emotion')
        
        success, message = data_manager.load_dataset(filepath, text_column, label_column)
        
        if not success:
            return jsonify({'success': False, 'message': message}), 400
        
        # Get statistics
        global current_dataset_stats
        current_dataset_stats = data_manager.get_statistics()
        
        return jsonify({
            'success': True,
            'message': message,
            'statistics': current_dataset_stats
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500


@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """
    Get current dataset statistics
    """
    if current_dataset_stats is None:
        return jsonify({'success': False, 'message': 'Dataset belum dimuat'}), 400
    
    return jsonify({
        'success': True,
        'statistics': current_dataset_stats
    })


@app.route('/api/preprocess', methods=['POST'])
def preprocess_data():
    """
    Preprocess dataset
    """
    try:
        global preprocessor
        
        # Get preprocessing parameters
        params = request.json
        max_features = params.get('max_features', 1000)
        use_stemming = params.get('use_stemming', True)
        use_stopword_removal = params.get('use_stopword_removal', True)
        
        # Initialize preprocessor
        preprocessor = IndonesianTextPreprocessor(
            max_features=max_features,
            use_stemming=use_stemming,
            use_stopword_removal=use_stopword_removal
        )
        
        # Sample preprocessing (just to show preview)
        sample_texts = data_manager.df['text'].head(5).tolist()
        preprocessed_samples = [preprocessor.preprocess_text(text) for text in sample_texts]
        
        preview = [
            {'original': orig, 'preprocessed': prep}
            for orig, prep in zip(sample_texts, preprocessed_samples)
        ]
        
        return jsonify({
            'success': True,
            'message': 'Preprocessing berhasil dikonfigurasi',
            'preview': preview,
            'config': {
                'max_features': max_features,
                'use_stemming': use_stemming,
                'use_stopword_removal': use_stopword_removal
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500


@app.route('/api/train', methods=['POST'])
def train_model():
    """
    Train ANN model
    """
    try:
        global model, preprocessor
        
        if data_manager.df is None:
            return jsonify({'success': False, 'message': 'Dataset belum dimuat'}), 400
        
        if preprocessor is None:
            # Use default preprocessor
            preprocessor = IndonesianTextPreprocessor()
        
        # Get training parameters
        params = request.json
        hidden_layers = params.get('hidden_layers', [64, 32])
        learning_rate = params.get('learning_rate', 0.01)
        epochs = params.get('epochs', 100)
        batch_size = params.get('batch_size', 32)
        test_size = params.get('test_size', 0.2)
        val_size = params.get('val_size', 0.1)
        activation = params.get('activation', 'sigmoid')
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = data_manager.split_data(
            test_size=test_size,
            val_size=val_size
        )
        
        # Preprocess texts
        X_train_tfidf = preprocessor.fit_transform(X_train)
        X_val_tfidf = preprocessor.transform(X_val) if X_val is not None else None
        X_test_tfidf = preprocessor.transform(X_test)
        
        # One-hot encode labels
        y_train_onehot = data_manager.one_hot_encode(y_train)
        y_val_onehot = data_manager.one_hot_encode(y_val) if y_val is not None else None
        y_test_onehot = data_manager.one_hot_encode(y_test)
        
        # Initialize model
        input_size = X_train_tfidf.shape[1]
        output_size = len(data_manager.emotion_labels)
        
        model = ANNClassifier(
            input_size=input_size,
            hidden_layers=hidden_layers,
            output_size=output_size,
            learning_rate=learning_rate,
            activation=activation
        )
        
        # Train model
        history = model.fit(
            X_train_tfidf, y_train_onehot,
            X_val_tfidf, y_val_onehot,
            epochs=epochs,
            batch_size=batch_size,
            verbose=False
        )
        
        # Evaluate on test set
        y_test_pred = model.predict(X_test_tfidf)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        
        # Classification report
        report = classification_report(
            y_test, y_test_pred,
            target_names=data_manager.emotion_labels,
            output_dict=True
        )
        
        return jsonify({
            'success': True,
            'message': 'Training selesai',
            'history': {
                'loss': history['loss'],
                'accuracy': history['accuracy'],
                'val_loss': history['val_loss'],
                'val_accuracy': history['val_accuracy']
            },
            'evaluation': {
                'test_accuracy': float(test_accuracy),
                'confusion_matrix': cm.tolist(),
                'classification_report': report
            },
            'model_config': model.get_config(),
            'data_split': {
                'train_samples': len(X_train),
                'val_samples': len(X_val) if X_val is not None else 0,
                'test_samples': len(X_test)
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict emotion for text
    """
    try:
        global model, preprocessor
        
        if model is None:
            return jsonify({'success': False, 'message': 'Model belum dilatih'}), 400
        
        if preprocessor is None:
            return jsonify({'success': False, 'message': 'Preprocessor belum diinisialisasi'}), 400
        
        # Get input text
        data = request.json
        text = data.get('text', '')
        
        # Validate input
        is_valid, message = data_manager.validate_prediction_input(text)
        if not is_valid:
            return jsonify({'success': False, 'message': message}), 400
        
        # Preprocess and predict
        X_tfidf = preprocessor.transform([text])
        prediction = model.predict(X_tfidf)[0]
        probabilities = model.predict_proba(X_tfidf)[0]
        
        # Get emotion label
        emotion = data_manager.emotion_labels[prediction]
        
        # Create probability distribution
        prob_dist = {
            label: float(prob)
            for label, prob in zip(data_manager.emotion_labels, probabilities)
        }
        
        return jsonify({
            'success': True,
            'prediction': emotion,
            'confidence': float(probabilities[prediction]),
            'probabilities': prob_dist,
            'preprocessed_text': preprocessor.preprocess_text(text)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500


@app.route('/api/predict_batch', methods=['POST'])
def predict_batch():
    """
    Predict emotions for multiple texts
    """
    try:
        if model is None:
            return jsonify({'success': False, 'message': 'Model belum dilatih'}), 400
        
        # Get input texts
        data = request.json
        texts = data.get('texts', [])
        
        if not texts:
            return jsonify({'success': False, 'message': 'Tidak ada teks untuk diprediksi'}), 400
        
        # Preprocess and predict
        X_tfidf = preprocessor.transform(texts)
        predictions = model.predict(X_tfidf)
        probabilities = model.predict_proba(X_tfidf)
        
        # Format results
        results = []
        for i, text in enumerate(texts):
            emotion = data_manager.emotion_labels[predictions[i]]
            prob_dist = {
                label: float(prob)
                for label, prob in zip(data_manager.emotion_labels, probabilities[i])
            }
            
            results.append({
                'text': text,
                'prediction': emotion,
                'confidence': float(probabilities[i][predictions[i]]),
                'probabilities': prob_dist
            })
        
        return jsonify({
            'success': True,
            'results': results,
            'total': len(results)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500


@app.route('/api/save_model', methods=['POST'])
def save_model():
    """
    Save trained model and preprocessor
    """
    try:
        if model is None:
            return jsonify({'success': False, 'message': 'Model belum dilatih'}), 400
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f'model_{timestamp}.pkl'
        preprocessor_filename = f'preprocessor_{timestamp}.pkl'
        metadata_filename = f'metadata_{timestamp}.json'
        
        model_path = os.path.join(MODEL_FOLDER, model_filename)
        preprocessor_path = os.path.join(MODEL_FOLDER, preprocessor_filename)
        metadata_path = os.path.join(MODEL_FOLDER, metadata_filename)
        
        # Save model
        model.save_model(model_path)
        
        # Save preprocessor
        preprocessor.save_preprocessor(preprocessor_path)
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'emotion_labels': data_manager.emotion_labels,
            'label_mapping': data_manager.get_label_mapping(),
            'model_config': model.get_config(),
            'model_file': model_filename,
            'preprocessor_file': preprocessor_filename
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return jsonify({
            'success': True,
            'message': 'Model berhasil disimpan',
            'files': {
                'model': model_filename,
                'preprocessor': preprocessor_filename,
                'metadata': metadata_filename
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500


@app.route('/api/load_model', methods=['POST'])
def load_model():
    """
    Load saved model and preprocessor
    """
    try:
        global model, preprocessor, data_manager
        
        data = request.json
        metadata_filename = data.get('metadata_file', '')
        
        metadata_path = os.path.join(MODEL_FOLDER, metadata_filename)
        
        if not os.path.exists(metadata_path):
            return jsonify({'success': False, 'message': 'File metadata tidak ditemukan'}), 400
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load model
        model_path = os.path.join(MODEL_FOLDER, metadata['model_file'])
        model = ANNClassifier(input_size=1, hidden_layers=[1], output_size=1)  # Dummy init
        model.load_model(model_path)
        
        # Load preprocessor
        preprocessor_path = os.path.join(MODEL_FOLDER, metadata['preprocessor_file'])
        preprocessor = IndonesianTextPreprocessor()
        preprocessor.load_preprocessor(preprocessor_path)
        
        # Update data manager with emotion labels
        data_manager.emotion_labels = metadata['emotion_labels']
        
        return jsonify({
            'success': True,
            'message': 'Model berhasil dimuat',
            'metadata': metadata
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500


@app.route('/api/list_models', methods=['GET'])
def list_models():
    """
    List all saved models
    """
    try:
        models = []
        
        for filename in os.listdir(MODEL_FOLDER):
            if filename.startswith('metadata_') and filename.endswith('.json'):
                metadata_path = os.path.join(MODEL_FOLDER, filename)
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                models.append({
                    'filename': filename,
                    'timestamp': metadata.get('timestamp'),
                    'emotion_labels': metadata.get('emotion_labels'),
                    'model_config': metadata.get('model_config')
                })
        
        # Sort by timestamp (newest first)
        models.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return jsonify({
            'success': True,
            'models': models
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500


@app.route('/api/download_template', methods=['GET'])
def download_template():
    """
    Download sample dataset template
    """
    try:
        temp_manager = DataManager()
        template_path = os.path.join(UPLOAD_FOLDER, 'template_dataset.csv')
        temp_manager.export_sample(template_path, n_samples=5)
        
        return send_file(template_path, as_attachment=True, download_name='template_dataset.csv')
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'}), 500


if __name__ == '__main__':
    print("=" * 50)
    print("Emotion Classification API Server")
    print("=" * 50)
    print("Server running on http://localhost:5000")
    print("=" * 50)
    app.run(debug=False, host='0.0.0.0', port=5000)
