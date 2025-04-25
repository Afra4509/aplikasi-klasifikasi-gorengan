# Aplikasi Klasifikasi Gorengan

Aplikasi berbasis web untuk mengklasifikasikan jenis gorengan (makanan ringan Indonesia) dan menampilkan harganya menggunakan TensorFlow Lite dan Streamlit.

![Aplikasi Klasifikasi Gorengan](https://via.placeholder.com/800x400?text=Aplikasi+Klasifikasi+Gorengan)

## 📝 Deskripsi

Aplikasi ini memanfaatkan model machine learning untuk mengklasifikasikan gambar gorengan ke dalam tiga kategori:
- Es Teh
- Risol
- Maryam

Setelah klasifikasi, aplikasi akan menampilkan jenis gorengan yang terdeteksi beserta harganya.

## ✨ Fitur

- Unggah dan klasifikasikan gambar gorengan
- Tampilan hasil klasifikasi dengan tingkat keyakinan
- Tampilan harga produk berdasarkan klasifikasi
- Fleksibilitas untuk menggunakan model default atau mengunggah model kustom
- Antarmuka pengguna yang responsif dan intuitif

## 🔧 Teknologi yang Digunakan

- Python 3.x
- Streamlit
- TensorFlow Lite
- NumPy
- OpenCV
- PIL (Python Imaging Library)

## 🚀 Instalasi dan Penggunaan

### Prasyarat

Pastikan Anda telah menginstal Python versi 3.6 atau yang lebih baru. Kemudian install semua dependensi yang diperlukan:

```bash
pip install -r requirements.txt
```

### Menjalankan Aplikasi

1. Clone repositori ini:
```bash
git clone https://github.com/username/aplikasi-klasifikasi-gorengan.git
cd aplikasi-klasifikasi-gorengan
```

2. Install dependensi:
```bash
pip install -r requirements.txt
```

3. Jalankan aplikasi:
```bash
streamlit run app.py
```

4. Buka browser dan akses `http://localhost:8501`

### Penggunaan

1. Pilih opsi model di sidebar (gunakan model default atau unggah model Anda sendiri)
2. Unggah gambar gorengan yang ingin diklasifikasikan
3. Klik tombol "Klasifikasi Gambar"
4. Lihat hasil klasifikasi beserta harga dan tingkat keyakinan

## 📂 Struktur Proyek

```
aplikasi-klasifikasi-gorengan/
├── app.py                  # Kode utama aplikasi Streamlit
├── model.tflite            # Model default TensorFlow Lite (tidak disertakan di repo)
├── requirements.txt        # Daftar dependensi
├── .gitignore              # File konfigurasi Git ignore
└── README.md               # Dokumentasi proyek
```

## 📋 Requirements

Buat file `requirements.txt` dengan konten berikut:

```
streamlit>=1.15.0
numpy>=1.19.5
opencv-python>=4.5.0
tensorflow>=2.7.0
pillow>=8.0.0
```

## 🔄 Model

Aplikasi ini mendukung model klasifikasi TensorFlow Lite (.tflite) dengan input gambar berukuran 224x224 piksel. Model default diharapkan dapat mendeteksi tiga kelas:
- Es Teh
- Risol
- Maryam

Jika Anda ingin menggunakan model kustom, pastikan model tersebut:
1. Dalam format TensorFlow Lite (.tflite)
2. Menerima input gambar dengan dimensi 224x224 piksel
3. Menghasilkan output probabilitas untuk kelas-kelas yang sesuai

## 🤝 Kontribusi

Kontribusi selalu diterima! Silakan buat issue atau pull request jika Anda memiliki perbaikan atau fitur tambahan.
