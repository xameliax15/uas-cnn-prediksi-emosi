# 📊 Dataset Auto-Merge Training

## ✅ Fitur Baru: Auto-Load Semua Dataset

Script `train_model.py` sekarang **otomatis** membaca dan menggabungkan **semua file CSV** di folder `datasets/`.

## 🔍 Cara Kerja

### 1. Scan Folder
Script akan scan folder `datasets/` dan mencari semua file `.csv`:

```python
datasets_folder = 'datasets'
csv_files = glob.glob(os.path.join(datasets_folder, '*.csv'))
```

### 2. Filter File
Script akan **exclude** file yang tidak relevan:
- File di subfolder `raw/`
- File template
- File yang tidak punya kolom `text` dan `emotion`

### 3. Load & Merge
Setiap file CSV yang valid akan:
1. Di-load dengan pandas
2. Divalidasi (harus punya kolom `text` dan `emotion`)
3. Digabung dengan dataset lain

### 4. Deduplication
Setelah merge:
- Hapus baris duplikat (berdasarkan kolom `text`)
- Hapus baris dengan nilai null
- Shuffle data secara random

### 5. Save Merged Dataset
Hasil merge disimpan ke:
```
datasets/merged_all_datasets.csv
```

File ini bisa digunakan langsung untuk training berikutnya.

## 📂 File yang Terdeteksi

Dari folder `datasets/`, script menemukan:

| File | Size | Status | Rows |
|------|------|--------|------|
| AngerData.csv | 122.5 KB | ❌ Format error | - |
| codeswitch_emotion.csv | 95.0 KB | ❌ Format error | - |
| EmoTweetID-Human.csv | 254.5 KB | ⚠️ Missing columns | - |
| EmoTweetID-Lexicon.csv | 256.4 KB | ⚠️ Missing columns | - |
| FearData.csv | 71.8 KB | ❌ Format error | - |
| **indonesian_emotion_dataset.csv** | 6.5 KB | ✅ **Loaded** | **210** |
| **indonesian_emotion_expanded.csv** | 8.4 KB | ✅ **Loaded** | **280** |
| JoyData.csv | 107.1 KB | ❌ Format error | - |
| LoveData.csv | 100.3 KB | ❌ Format error | - |
| NeutralData.csv | 240.0 KB | ❌ Format error | - |
| SadData.csv | 99.9 KB | ❌ Format error | - |

**Total berhasil di-load**: 2 file ✅  
**Total rows setelah merge & dedup**: **280 samples**

## 📊 Hasil Merge

```
Initial merge: 490 rows (210 + 280)
After deduplication: 280 rows
Duplicates removed: 210 rows
```

**Distribusi Final (Perfectly Balanced):**
```
jijik     : 40 samples (14.3%)
marah     : 40 samples (14.3%)
netral    : 40 samples (14.3%)
sedih     : 40 samples (14.3%)
senang    : 40 samples (14.3%)
takut     : 40 samples (14.3%)
terkejut  : 40 samples (14.3%)
```

## ⚠️ File yang Gagal Load

Beberapa file gagal di-load karena:

### Format Error (TSV format, bukan CSV):
- AngerData.csv
- FearData.csv  
- JoyData.csv
- LoveData.csv
- NeutralData.csv
- SadData.csv

**Solusi**: File-file ini kemungkinan dalam format TSV (tab-separated), perlu dikonversi atau dibaca dengan parameter berbeda.

### Missing Columns:
- EmoTweetID-Human.csv (kolom berbeda)
- EmoTweetID-Lexicon.csv (kolom berbeda)
- codeswitch_emotion.csv (kolom berbeda)

**Solusi**: File ini perlu mapping kolom manual.

## 🎯 Cara Menggunakan

### Jalankan Training
```bash
python train_model.py
```

Script akan otomatis:
1. ✅ Scan folder `datasets/`
2. ✅ Load semua CSV valid
3. ✅ Merge & deduplicate
4. ✅ Save ke `merged_all_datasets.csv`
5. ✅ Train model dengan ANN
6. ✅ Save model ke `backend/models/`

### Tambah Dataset Baru

Cukup **drop file CSV** baru ke folder `datasets/`:

```
datasets/
├── your_new_dataset.csv  ← Tambahkan file baru di sini
```

**Syarat:**
- Format: CSV (comma-separated)
- Kolom wajib: `text`, `emotion`
- Encoding: UTF-8

Script akan **otomatis** mendeteksi dan merge!

## 🔧 Kustomisasi

Jika ingin exclude file tertentu, edit di `train_model.py`:

```python
# Exclude file tertentu
csv_files = [f for f in csv_files 
             if 'raw' not in f.lower() 
             and 'template' not in f.lower()
             and 'old' not in f.lower()]  # Tambah exclude di sini
```

## 📝 Output Files

Training akan menghasilkan:

```
datasets/merged_all_datasets.csv  ← Merged dataset
backend/models/
├── pretrained_model.pkl          ← Trained model
├── pretrained_preprocessor.pkl   ← TF-IDF vectorizer  
└── pretrained_metadata.json      ← Model info
```

## ✅ Keuntungan Auto-Merge

1. **Flexible**: Tinggal tambah file CSV baru
2. **Automatic**: Tidak perlu edit code
3. **Safe**: Auto-deduplication & validation
4. **Traceable**: Merged file tersimpan untuk audit
5. **Scalable**: Bisa handle banyak dataset sekaligus

---

**Current Status**: Training model dengan 280 samples across 7 emotions (perfectly balanced) 🚀
