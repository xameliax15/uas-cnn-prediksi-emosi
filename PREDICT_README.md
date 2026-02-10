# 🎯 Aplikasi Prediksi Emosi - Versi Simplified

Aplikasi web sederhana untuk analisis emosi teks berbahasa Indonesia, **tanpa bagian training**. Interface bersih dan modern seperti [Junia.ai Text Tone Analyzer](https://www.junia.ai/tools/text-tone-analyzer).

## 📂 File yang Tersedia

### Versi Lengkap (Development)
- `frontend/index.html` - Full version dengan upload, preprocessing, training, dan prediksi
- Untuk penelitian, development, dan training model

### Versi Simplified (Production) ⭐ **BARU!**
- `frontend/predict.html` - **Prediction-only interface**
- Hanya input teks dan hasil prediksi
- Clean, simple, production-ready

## 🚀 Cara Menggunakan Versi Simplified

### Step 1: Train Model (One-Time Setup)

Model harus di-train terlebih dahulu sebelum menggunakan interface prediksi:

```bash
python train_model.py
```

Script ini akan:
- ✅ Load dataset (`datasets/indonesian_emotion_dataset.csv`)
- ✅ Preprocessing dengan TF-IDF + Sastrawi
- ✅ Train ANN model (150 epochs)
- ✅ Evaluasi accuracy
- ✅ Save model ke `backend/models/pretrained_model.pkl`

**Note**: Training memakan waktu 2-5 menit tergantung CPU.

### Step 2: Jalankan Backend Server

```bash
cd backend
python main.py
```

Backend akan running di `http://localhost:5000`

### Step 3: Buka Interface Prediksi

Pilih salah satu cara:

**Opsi A: Double-click**
- Buka File Explorer
- Navigate ke `frontend/`
- Double-click `predict.html`

**Opsi B: PowerShell**
```powershell
Start-Process "d:\Documents\uas cnn\frontend\predict.html"
```

**Opsi C: Browser**
- Buka browser
- File → Open File
- Pilih `frontend/predict.html`

## 🎨 Fitur Interface Simplified

### Yang Ditampilkan:
✅ Input area untuk teks  
✅ Tombol "Analisis Emosi"  
✅ Quick example chips (senang, sedih, marah, terkejut)  
✅ Hasil prediksi dengan emoji besar  
✅ Confidence score dengan color-coded badge  
✅ Probability distribution bars untuk semua 7 emosi  
✅ Preprocessed text details (collapsible)  

### Yang TIDAK Ditampilkan:
❌ Upload dataset  
❌ Preprocessing configuration  
❌ Training section  
❌ Charts & visualizations  
❌ Model configuration  

## 📊 Hasil yang Ditampilkan

Ketika Anda analisis teks, akan muncul:

```
Emosi Terdeteksi: Senang
😊
95.2% Confidence

Distribusi Probabilitas:
Senang   ████████████████████ 95.2%
Netral   ██ 2.1%
Terkejut █ 1.5%
...
```

## 🎯 Cara Menggunakan Interface

1. **Ketik/paste teks** di text area (minimal 3 karakter)
2. **Klik "Analisis Emosi"** atau tekan `Ctrl+Enter`
3. **Lihat hasil**:
   - Emosi terdeteksi (senang, sedih, marah, dll)
   - Confidence score (0-100%)
   - Emoji visualization
   - Probability bars untuk semua emosi
4. **Klik "Hapus"** untuk clear dan input teks baru

## 🎨 Design Features

- **Glassmorphism style** dengan frosted glass effect
- **Purple/pink gradient** background dengan animasi
- **Smooth animations** untuk loading dan results
- **Color-coded emotions** (hijau = confident, orange = medium, merah = low)
- **Responsive design** untuk mobile & desktop
- **Emoji visualization** untuk setiap emosi
- **Dark theme** modern dan eye-friendly

## 🔧 Perbandingan Dua Versi

| Fitur | index.html (Full) | predict.html (Simple) |
|-------|-------------------|----------------------|
| Upload Dataset | ✅ | ❌ |
| Preprocessing Config | ✅ | ❌ |
| Model Training | ✅ | ❌ |
| Training Charts | ✅ | ❌ |
| Prediction | ✅ | ✅ |
| Emoji Display | ❌ | ✅ |
| Example Prompts | ❌ | ✅ |
| Use Case | Research/Dev | Production |

## ⚙️ Technical Stack

### Frontend
- HTML5 + CSS3 (Glassmorphism)
- Vanilla JavaScript (no framework)
- Google Fonts (Inter)
- Fetch API for AJAX calls

### Backend (sama untuk kedua versi)
- Python 3.x
- Flask API
- Custom ANN (NumPy)
- Sastrawi (Indonesian stemmer)
- TF-IDF Vectorization

## 📋 Requirements

Backend harus tetap running untuk kedua versi:

```bash
# Install dependencies
pip install flask flask-cors numpy pandas scikit-learn sastrawi

# Run server
python backend/main.py
```

## 🐛 Troubleshooting

**Error: "Gagal terhubung ke server"**
- Pastikan backend running di `http://localhost:5000`
- Check terminal ada log "Running on http://0.0.0.0:5000"

**Error: "Model belum dilatih"**
- Jalankan `python train_model.py` dulu
- Pastikan file `backend/models/pretrained_model.pkl` ada

**Hasil prediksi aneh**
- Check apakah model sudah di-train dengan dataset yang benar
- Training ulang dengan epochs lebih banyak

## 🎓 Untuk Penelitian

Gunakan **index.html** (full version) untuk:
- Eksperimen dengan dataset berbeda
- Training dengan hyperparameter berbeda
- Analisis performa model
- Visualisasi training progress

Gunakan **predict.html** (simple version) untuk:
- Demo ke dosen/penguji
- Production deployment
- User testing
- Clean presentation

## 📸 Screenshot

### Interface Prediksi (predict.html)
- Clean input area
- Beautiful gradient background
- Emoji-based result display
- Color-coded confidence scores
- Smooth animations

## 🚀 Deploy to Production

Untuk production (web hosting):

1. **Train model** di local dengan dataset lengkap
2. **Copy backend/** folder ke server
3. **Setup Python environment** di server
4. **Run Flask** dengan Gunicorn/uWSGI
5. **Deploy frontend** (`predict.html` + CSS + JS)
6. **Update API_URL** di `app-simple.js` ke production URL

---

Sekarang Anda punya **2 versi aplikasi**:
- 📊 **Full version** untuk development & research
- 🎯 **Simple version** untuk production & demo

Pilih sesuai kebutuhan! 🚀
