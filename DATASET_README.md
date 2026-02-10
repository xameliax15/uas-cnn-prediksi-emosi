# Dataset Download & Conversion Tools

## 📥 Download Dataset Otomatis

Script ini akan men-download dataset emosi bahasa Indonesia dari sumber publik dan mengkonversinya ke format yang sesuai dengan aplikasi.

## 🎯 Dataset yang Didownload

1. **Indonesian Twitter Emotion Dataset** (4,403 tweets)
   - Sumber: GitHub @meisaputri21
   - Label: love, anger, sadness, joy, fear

2. **Public Opinion Emotion Dataset** (7,080 tweets)
   - Sumber: GitHub @Ricco48
   - Label: anger, fear, joy, love, sad, neutral

## 🔄 Mapping Label Emosi

Dataset akan di-mapping ke 7 kategori standar:

| Label Asli | Label Baru |
|------------|------------|
| anger | marah |
| fear | takut |
| joy, happy, happiness, love | senang |
| sadness, sad | sedih |
| disgust | jijik |
| surprise | terkejut |
| neutral | netral |

## 📦 Instalasi Dependencies

```bash
pip install requests pandas
```

## 🚀 Cara Menggunakan

### 1. Jalankan Script Download

```bash
python download_dataset.py
```

Script akan:
- ✅ Download dataset dari GitHub
- ✅ Mapping label emosi ke 7 kategori
- ✅ Menggabungkan semua dataset
- ✅ Menghapus duplikat
- ✅ Menambahkan sample untuk emosi yang hilang (jijik, terkejut)
- ✅ Menyimpan hasil akhir ke `datasets/indonesian_emotion_7categories_enhanced.csv`

### 2. Output

File yang dihasilkan:
```
datasets/
├── raw_twitter_emotion.csv              # Dataset mentah 1
├── raw_public_opinion.csv               # Dataset mentah 2
├── indonesian_emotion_7categories.csv   # Dataset gabungan
└── indonesian_emotion_7categories_enhanced.csv  # Dataset final + sample tambahan
```

### 3. Upload ke Aplikasi

1. Buka aplikasi web emotion classification
2. Upload file: `datasets/indonesian_emotion_7categories_enhanced.csv`
3. Kolom teks: `text`
4. Kolom label: `emotion`
5. Klik "Upload Dataset"

## 📊 Statistik Dataset Final

Setelah download selesai, Anda akan mendapatkan:
- **Total**: ~11,000+ samples (tergantung duplikat yang dihapus)
- **Format**: CSV dengan 2 kolom (`text`, `emotion`)
- **7 Kategori**: marah, jijik, takut, senang, sedih, terkejut, netral

## ⚠️ Catatan

- **Internet diperlukan** untuk download
- Jika ada dataset yang gagal didownload, script akan tetap melanjutkan dengan dataset yang berhasil
- Emosi `jijik` dan `terkejut` akan ditambahkan dengan sample manual karena jarang ada di dataset publik
- Dataset sudah di-shuffle (acak) untuk training yang lebih baik

## 🔧 Troubleshooting

**Error: Failed to download**
- Cek koneksi internet
- Coba akses GitHub URL secara manual
- Beberapa repository mungkin sudah dihapus/dipindah

**Missing emotions**
- Normal jika `jijik` dan `terkejut` tidak ada di dataset asli
- Script akan otomatis menambahkan sample untuk emosi ini

**Duplicate removal**
- Script otomatis menghapus teks yang duplikat
- Jumlah akhir mungkin berbeda dari jumlah asli

## 📚 Sumber Dataset

- [Indonesian Twitter Emotion Dataset](https://github.com/meisaputri21/Indonesian-Twitter-Emotion-Dataset)
- [Emotion Dataset from Indonesian Public Opinion](https://github.com/Ricco48/Emotion-Dataset-from-Indonesian-Public-Opinion)

## 🎓 Untuk Penelitian

Dataset yang dihasilkan cocok untuk:
- Training model klasifikasi emosi
- Penelitian sentiment analysis bahasa Indonesia
- Skripsi/tesis NLP
- Jurnal ilmiah

Pastikan untuk **mengutip sumber dataset asli** dalam penelitian Anda!
