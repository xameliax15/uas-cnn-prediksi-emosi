# 📊 Dataset Klasifikasi Emosi Bahasa Indonesia

## Dataset yang Tersedia

### 1. Indonesian Emotion Dataset (210 samples) - RECOMMENDED ⭐

**Lokasi**: `datasets/indonesian_emotion_dataset.csv`

**Spesifikasi:**
- 📊 **Total**: 210 teks berbahasa Indonesia
- 🎯 **Kategori**: 7 emosi dasar
- ⚖️ **Distribusi**: Seimbang (30 samples per kategori)
- 🌐 **Bahasa**: Formal dan informal Indonesian
- 📝 **Format**: CSV (text, emotion)

**Distribusi Detail:**
```
jijik (disgust)    : 30 samples
marah (anger)      : 30 samples  
netral (neutral)   : 30 samples
sedih (sadness)    : 30 samples
senang (joy)       : 30 samples
takut (fear)       : 30 samples
terkejut (surprise): 30 samples
-----------------------------------
TOTAL              : 210 samples
```

**Karakteristik:**
- ✅ Bahasa Indonesia autentik (formal & gaul)
- ✅ Variasi intensitas emosi (mild to extreme)
- ✅ Real-world expressions
- ✅ Balanced untuk training

### 2. Sample Dataset (63 samples)

**Lokasi**: `sample_dataset.csv`

Dataset kecil untuk quick testing.

## 🎯 Kategori Emosi

| Emosi | Label | Deskripsi | Contoh |
|-------|-------|-----------|---------|
| Senang | `senang` | Bahagia, gembira, sukacita | "Alhamdulillah semua berjalan lancar" |
| Sedih | `sedih` | Kecewa, duka, nelangsa | "Hatiku hancur mendengar berita itu" |
| Marah | `marah` | Kesal, murka, dongkol | "Kenapa harus begini aku kesal" |
| Takut | `takut` | Cemas, khawatir, ngeri | "Deg-degan banget nunggu hasilnya" |
| Jijik | `jijik` | Muak, menjijikkan, geli | "Ih najis banget deh enggak suka" |
| Terkejut | `terkejut` | Kaget, tidak terduga | "Wow! Saya tidak menyangka hal ini" |
| Netral | `netral` | Tidak ada emosi khusus | "Biasa saja tidak ada yang spesial" |

## 📥 Cara Menggunakan

### Di Aplikasi Web

1. Buka aplikasi emotion classification
2. Di bagian "Upload Dataset"
3. Klik "Pilih file"
4. Pilih: `datasets/indonesian_emotion_dataset.csv`
5. Kolom teks: `text` (default)
6. Kolom label: `emotion` (default)
7. Klik "Upload Dataset"

### Lihat Statistik

```bash
python -c "import pandas as pd; df = pd.read_csv('datasets/indonesian_emotion_dataset.csv'); print(df['emotion'].value_counts())"
```

## 🔄 Download Dataset Publik (Opsional)

Untuk dataset yang lebih besar, gunakan script `download_dataset.py`:

```bash
python download_dataset.py
```

Script akan mencoba download dari:
- Indonesian Twitter Emotion Dataset (4,403 tweets)
- Public Opinion Emotion Dataset (7,080 tweets)

**Note**: Memerlukan koneksi internet dan akses ke GitHub.

## 📚 Sumber & Referensi

Dataset ini dibuat berdasarkan:
- Penelitian emosi dasar Ekman
- Indonesian sentiment analysis best practices
- Real Indonesian social media expressions

Untuk penelitian, Anda dapat mengutip:
- Dataset custom untuk klasifikasi emosi bahasa Indonesia
- 7 kategori emosi: marah, jijik, takut, senang, sedih, terkejut, netral

## 💡 Tips Training

**Untuk hasil optimal:**
- Gunakan dataset 210 samples untuk training
- Set test size 20% (42 samples untuk testing)
- Gunakan validation 10% (21 samples untuk validasi)
- Epochs: 100-200
- Learning rate: 0.01
- Hidden layers: [64, 32] atau [128, 64]

**Expected Results:**
- Training accuracy: 85-95%
- Test accuracy: 75-85% (tergantung arsitektur)
- Balanced precision/recall untuk semua kategori

## 📊 Statistics

```
Dataset: indonesian_emotion_dataset.csv
Total samples: 210
Categories: 7
Distribution: Perfectly balanced (30 each)
Language: Indonesian (ID)
Format: CSV
Encoding: UTF-8
```

Silakan gunakan dataset ini untuk penelitian klasifikasi emosi Anda! 🚀
