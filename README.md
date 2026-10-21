# DL-RA_studio2-klasifikasi-sampah
Proyek kompetisi klasifikasi lima jenis sampah menggunakan Convolutional Neural Network (CNN) from scratch dan fine-tuning model pretrained untuk membandingkan performa berdasarkan Macro-F1
# Studio Proyek 2 — Eksperimen Deep Learning untuk Klasifikasi Sampah

**Mata Kuliah:** Deep Learning (IF25-40401)  
**Program Studi:** Teknik Informatika, Institut Teknologi Sumatera  
**Semester:** Ganjil 2026/2027

## 1. Identitas Kelompok
**Nama Kelompok:** Manula

| No. | Nama Anggota | NIM |
|---|---|---|
| 1 | Aprililianti | 123140041 |
| 2 | Fanisa Aulia Safitri | 123140121 |

## 2. Gambaran Proyek

Model klasifikasi citra yang menerima satu foto sampah dan memprediksi satu dari lima kelas, yaitu:
| Indeks | Kelas|
|---|---|
| 0 | botol_plastik |
| 1 | gelas_plastik |
| 2 | kertas_kardus |
| 3 | saset_kemasan |
| 4 | styrofoam |

Penilaian memakai Macro-F1 pada validation set tersembunyi milik tim pengajar

## 3. Rencana Eksperimen

| Komponen | Rencana |
|---|---|
| Arsitektur CNN | Eksperimen CNN dari awal dan beberapa backbone pretrained |
| Backbone yang dipertimbangkan | AlexNet, VGG, ResNet, MobileNet, dan ConvNeXT |
| Strategi pelatihan | Penggantian layer klasifikasi, freezing layer, linear probing, dan full fine-tuning |
| Pengolahan citra | Resize, normalisasi, dan augmentasi sesuai kebutuhan |
| Evaluasi | Macro-F1 dan efisiensi model |

Konfigurasi akhir akan ditentukan berdasarkan hasil eksperimen yang dilakukan selama proyek berlangsung.

## 4. Pemantauan Hasil Kompetisi

Evaluasi dilakukan melalui tiga putaran preview dan satu hasil final yang menjadi dasar penilaian performa model.

| Tahap Evaluasi | Jadwal | Macro-F1 |
|---|---|---|
| Preview 1 | 14 Oktober 2026 | Belum tersedia |
| Preview 2 | 21 Oktober 2026 | Belum tersedia |
| Preview 3 / Final | 25 Oktober 2026 | Belum tersedia |

## 5. Struktur Repository

- `train.py` — berkas untuk menjalankan proses pelatihan model.
- `evaluate.py` — berkas untuk mendukung proses evaluasi model.
- `submission/model.py` — definisi arsitektur model dan transformasi citra.
- `submission/requirements.txt` — daftar pustaka yang diperlukan.
- `submission/model.pth` — bobot model yang dihasilkan setelah pelatihan.
- `submission/README.md` — petunjuk penggunaan dan informasi berkas submission.

