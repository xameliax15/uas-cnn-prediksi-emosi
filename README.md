# Klasifikasi Emosi pada Teks Media Sosial Berbahasa Indonesia

## Deskripsi

Aplikasi web untuk klasifikasi emosi pada teks media sosial berbahasa Indonesia menggunakan **Artificial Neural Network (ANN)** dengan algoritma **Backpropagation**.

## Fitur

- 📁 Upload dataset (CSV, Excel, JSON)
- 🔧 Preprocessing teks Indonesia (Stemming dengan Sastrawi, Stopword removal)
- 🎯 Training model ANN dengan konfigurasi custom
- 📊 Visualisasi training (Loss & Accuracy charts)
- 🔮 Prediksi emosi untuk teks baru
- 💾 Save/Load trained model
- 📈 Evaluasi model (Confusion Matrix, Classification Report)

## Teknologi

### Backend
- Python 3.x
- Flask (Web Framework)
- NumPy (ANN implementation from scratch)
- Sastrawi (Indonesian NLP)
- Scikit-learn (Metrics & Vectorization)
- Pandas (Data manipulation)

### Frontend
- HTML5, CSS3, JavaScript
- Chart.js (Visualization)
- Modern Glassmorphism Design

## Instalasi

### 1. Install Dependencies Backend

```bash
cd backend
pip install -r requirements.txt
```

### 2. Jalankan Backend Server

```bash
python main.py
```

Server akan berjalan di `http://localhost:5000`

### 3. Jalankan Frontend

Buka `frontend/index.html` di browser, atau gunakan live server:

```bash
cd frontend
# Gunakan live server favorit Anda, misalnya:
# python -m http.server 8000
```

## Cara Penggunaan

### 1. Upload Dataset
- Siapkan dataset dengan kolom `text` (teks) dan `emotion` (label emosi)
- Upload file CSV/Excel/JSON
- Lihat statistik dataset

### 2. Preprocessing
- Konfigurasi parameter preprocessing (TF-IDF, Stemming, Stopwords)
- Jalankan preprocessing
- Review hasil preprocessing

### 3. Training Model
- Konfigurasi arsitektur ANN:
  - Hidden layers (contoh: 64,32)
  - Learning rate
  - Epochs
  - Batch size
  - Activation function
- Mulai training
- Monitor progress dan visualisasi

### 4. Evaluasi
- Review test accuracy
- Lihat confusion matrix
- Analisis classification report

### 5. Prediksi
- Input teks baru
- Dapatkan prediksi emosi dengan confidence score
- Lihat distribusi probabilitas

## Struktur Dataset

Format CSV contoh:

```csv
text,emotion
"Hari ini sangat menyenangkan!",senang
"Saya merasa sedih dan kecewa",sedih
"Ini membuat saya sangat marah",marah
"Saya takut akan hal ini",takut
"Menjijikkan sekali perilaku seperti itu",jijik
"Wow! Saya tidak menyangka hal ini",terkejut
"Biasa saja tidak ada yang spesial",netral
```

## Kategori Emosi

Aplikasi ini mendukung 7 kategori emosi:
1. **Marah** (Anger) - Perasaan kesal, dongkol, murka
2. **Jijik** (Disgust) - Perasaan muak, menjijikkan, tidak suka
3. **Takut** (Fear) - Perasaan cemas, khawatir, ngeri
4. **Senang** (Joy) - Perasaan bahagia, gembira, sukacita
5. **Sedih** (Sadness) - Perasaan kecewa, hancur, terpukul
6. **Terkejut** (Surprise) - Perasaan kaget, tidak terduga
7. **Netral** (Neutral) - Tidak ada emosi khusus, biasa saja

## Arsitektur ANN

Model menggunakan:
- **Input Layer**: TF-IDF features
- **Hidden Layers**: Configurable (default: [64, 32])
- **Output Layer**: Softmax untuk multi-class classification
- **Activation**: Sigmoid/ReLU/Tanh
- **Training**: Backpropagation dengan mini-batch gradient descent
- **Loss**: Cross-entropy

## API Endpoints

- `POST /api/upload` - Upload dataset
- `POST /api/preprocess` - Preprocess data
- `POST /api/train` - Train model
- `POST /api/predict` - Predict single text
- `POST /api/predict_batch` - Predict multiple texts
- `POST /api/save_model` - Save trained model
- `POST /api/load_model` - Load saved model
- `GET /api/list_models` - List all saved models
- `GET /api/download_template` - Download dataset template

## Lisensi

MIT License

## Kontributor

Dibuat untuk penelitian klasifikasi emosi pada teks media sosial berbahasa Indonesia.
