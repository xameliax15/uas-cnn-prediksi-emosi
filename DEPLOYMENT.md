# Deployment Guide: Emotion Classification Application

Aplikasi ini dapat di-deploy ke berbagai platform cloud (Render, Railway, Heroku, AWS, Google Cloud). Berikut adalah panduan deployment paling mudah menggunakan **Render** (Gratis).

## Persiapan Sebelum Deployment

1.  **Pastikan `requirements.txt` lengkap**:
    Sudah ditambahkan `gunicorn` dan dependencies lainnya:
    - `flask`
    - `flask-cors`
    - `numpy`
    - `pandas`
    - `scikit-learn`
    - `Sastrawi`
    - `openpyxl`
    - `gunicorn`

2.  **Procfile**:
    File `Procfile` sudah dibuat di root directory dengan isi:
    ```
    web: gunicorn backend.main:app
    ```
    Ini memberi tahu server cara menjalankan aplikasi.

3.  **Project Structure**:
    Pastikan struktur folder seperti ini (sudah sesuai):
    ```
    / (root)
      ├── backend/
      ├── frontend/
      ├── Procfile
      ├── requirements.txt
      └── ...
    ```

## Langkah Deployment di Render.com (Gratis & Mudah)

1.  **Push ke GitHub**:
    Pastikan semua kode sudah di-push ke repository GitHub Anda.

2.  **Buat Akun di Render**:
    Buka [render.com](https://render.com) dan login/signup dengan GitHub.

3.  **Buat Web Service Baru**:
    - Klik "New" -> "Web Service".
    - Pilih repository GitHub project ini.

4.  **Konfigurasi**:
    - **Name**: `emotion-classification-app` (atau nama lain)
    - **Region**: Singapore (paling dekat dengan Indonesia)
    - **Branch**: `main` (atau branch yang aktif)
    - **Root Directory**: `.` (biarkan kosong/default)
    - **Runtime**: Python 3
    - **Build Command**: `pip install -r backend/requirements.txt`
    - **Start Command**: `gunicorn backend.main:app`

5.  **Environment Variables**:
    Tambahkan variabel berikut (optional tapi recommended):
    - `PYTHON_VERSION`: `3.10.0` (atau versi Python yang Anda gunakan)

6.  **Deploy**:
    Klik "Create Web Service". Render akan mulai build dan deploy aplikasi Anda.

*> Catatan: Render "Free Tier" mungkin melakukan spin-down (tidur) jika tidak diakses selama 15 menit. Akses pertama akan lambat (cold start).*

---

## 🐍 Alternatif: PythonAnywhere (100% Gratis & Stabil)

Jika Anda kesulitan dengan Render, **PythonAnywhere** adalah pilihan terbaik khusus Python.

1.  **Daftar Akun**:
    Buka `www.pythonanywhere.com` dan buat akun "Beginner" (Gratis).

2.  **Upload Code**:
    -   Buka tab **Files**.
    -   Upload file `emotion_classification_app.zip` (yang sudah saya buatkan).
    -   Unzip di console bash: `unzip emotion_classification_app.zip`.

3.  **Install Dependencies**:
    Buka **Bash Console** dan jalankan:
    ```bash
    pip install flask flask-cors numpy pandas scikit-learn Sastrawi openpyxl
    ```

4.  **Setup Web App**:
    -   Buka tab **Web**.
    -   Klik "Add a new web app".
    -   Pilih **Flask** -> **Python 3.10** (atau terbaru).
    -   Path: Sesuaikan dengan folder tempat Anda unzip (misal: `/home/username/mysite/backend/main.py`).

5.  **Reload**:
    Klik tombol hijau "Reload". Website Anda akan live di `username.pythonanywhere.com`!

---

## Akses Aplikasi

Setelah deployment selesai (biasanya 2-5 menit), Render akan memberikan URL public, misal:
`https://emotion-classification-app.onrender.com`

Anda bisa menggunakan URL ini untuk mengakses API backend.

## Deployment Frontend

Untuk frontend (`index.html`, `predict.html`), cara termudah adalah:

**Opsi 1: Serve Frontend dari Backend (Termudah)**

Ubah `backend/main.py` sedikit agar Flask juga menyajikan file HTML statis.
(Saya bisa bantu setup ini jika Anda mau).

**Opsi 2: Deploy Frontend Terpisah (Vercel/Netlify)**

1.  Upload folder `frontend` ke GitHub terpisah atau drag & drop ke [Netlify](https://netlify.com).
2.  Ubah `API_URL` di `frontend/app.js` agar mengarah ke URL backend yang sudah di-deploy di Render (bukan `localhost:5000`).

---

## 🌩️ Alternatif: Deployment dengan Cloudflare Tunnel (Mudah & Gratis)

Jika Anda ingin menjalankan aplikasi di **laptop sendiri** tetapi bisa diakses orang lain lewat internet (tanpa perlu setting IP Public Router), gunakan **Cloudflare Tunnel**.

### Kelebihan:
-   **Gratis** selamanya.
-   **Tidak perlu upload** file ke server (coding tetap di laptop).
-   **HTTPS** otomatis (Secure).
-   Cocok untuk **demo tugas/skripsi**.

### Langkah-langkah:

1.  **Download Cloudflared**:
    -   Download `cloudflared-windows-amd64.exe` dari [Cloudflare Downloads](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/).
    -   Rename file menjad `cloudflared.exe` dan taruh di folder project ini.

2.  **Jalankan Backend**:
    Pastikan backend Flask berjalan di terminal lain:
    ```bash
    python backend/main.py
    ```

3.  **Jalankan Tunnel (Quick Tunnel)**:
    Buka terminal baru di folder project dan ketik:
    ```powershell
    .\cloudflared.exe tunnel --url http://localhost:5000
    ```

4.  **Dapatkan URL**:
    Cloudflare akan memberikan link acak, contoh:
    ```
    https://random-name-123.trycloudflare.com
    ```
    Copy link tersebut dan berikan ke teman/dosen Anda. Aplikasi Anda sekarang online!

*> Catatan: Laptop harus tetap menyala dan terhubung internet agar link bisa diakses.*

---

## Catatan Penting

- **Data Persistance**: Di Render versi gratis, file yang di-upload atau model yang di-train **akan hilang** setiap kali restart/re-deploy.
- Untuk aplikasi production yang serius, gunakan database (PostgreSQL) dan object storage (AWS S3) untuk menyimpan dataset dan model.
- Aplikasi ini cocok untuk **Demo** dan **Portfolio**.

---

**Butuh bantuan setup "Serve Frontend dari Backend"?**
Beri tahu saya, dan saya akan konfigurasi agar Backend Flask juga bisa menampilkan Frontend, jadi cuma perlu deploy 1 service saja di Render!
