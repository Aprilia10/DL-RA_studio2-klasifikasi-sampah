# Studio Proyek 2 — Eksperimen Deep Learning untuk Klasifikasi Sampah

**Mata Kuliah:** Deep Learning (IF25-40401)  
**Program Studi:** Teknik Informatika, Institut Teknologi Sumatera  
**Semester:** Ganjil 2026/2027

Proyek kompetisi klasifikasi lima jenis sampah menggunakan Convolutional Neural Network (CNN) *from scratch* dan *fine-tuning* model *pretrained* untuk membandingkan performa berdasarkan Macro-F1.

---

## 1. Identitas Kelompok

**Nama Kelompok:** Manula

| No. | Nama Anggota | NIM |
|:---:|:---|:---:|
| 1 | Aprililianti | 123140041 |
| 2 | Fanisa Aulia Safitri | 123140121 |

---

## 2. Gambaran Proyek

Model klasifikasi citra menerima satu foto sampah dan memprediksi satu dari lima kelas target:

| Indeks | Kelas | Keterangan |
|:---:|:---|:---|
| 0 | `botol_plastik` | Botol minuman plastik bening/berwarna |
| 1 | `gelas_plastik` | Cup atau gelas kemasan minuman plastik |
| 2 | `kertas_kardus` | Kardus, kertas kemasan, dan lembaran kertas |
| 3 | `saset_kemasan` | Bungkus saset plastik/alumunium foil (makanan/minuman) |
| 4 | `styrofoam` | Wadah makanan berbahan busa polistirena |

Metrik penilaian utama yang digunakan adalah **Macro-F1** pada set validasi tersembunyi milik tim pengajar.

---

## 3. Rencana & Ringkasan Pendekatan

| Komponen | Rencana & Implementasi |
|---|---|
| **Arsitektur Model** | Baseline ResNet-18 (*torchvision*), ConvNeXt-Tiny (*timm*), dan EfficientNet-B2 (*timm*) |
| **Strategi Pelatihan** | Fine-tuning model pretrained ImageNet dengan pemisahan learning rate antara *backbone* dan *classification head* |
| **Augmentasi Citra** | Random Resized Crop, Horizontal Flip, Color Jitter, Random Rotation, TrivialAugmentWide, dan Random Erasing |
| **Regularisasi & Loss** | Label Smoothing Cross-Entropy Loss (`smooth=0.1`), AdamW optimizer, dan Cosine Annealing Learning Rate Scheduler |
| **Protokol Validasi** | Local split 15% dari data latih (`my_val/`) dengan seed terkunci (42) untuk perbandingan adil dan pencegahan *data leakage* |

---

## 4. Pemantauan Hasil Kompetisi

Evaluasi dilakukan melalui dua putaran preview dan satu evaluasi final yang menjadi penentu akhir penilaian performa model.

| Tahap Evaluasi | Jadwal | Macro-F1 | Status / Model |
|:---|:---:|:---:|:---|
| Preview 1 | 14 Oktober 2026 | - | Belum tersedia |
| Preview 2 | 21 Oktober 2026 | **0.9901** | ResNet-18 (Pretrained) |
| Final | 25 Oktober 2026 | *Menunggu evaluasi* | Model terbaik kelompok Manula |

---

## 5. Hasil Evaluasi & Perbandingan Model

Evaluasi model pada subset validasi lokal (`my_val`) yang sama:

| Model | Ukuran Input | Macro-F1 (`my_val`) | Akurasi | Keterangan |
|---|:---:|:---:|:---:|---|
| **ResNet-18 (Baseline)** | 224×224 | *TBD* | *TBD* | Model submission Preview 2 (44.8 MB) |
| **ConvNeXt-Tiny** | 224×224 | *TBD* | *TBD* | Eksperimen Anggota 1 (`sub_a`) |
| **EfficientNet-B2** | 288×288 | *TBD* | *TBD* | Eksperimen Anggota 2 (`sub_b`) |

*Aturan adopsi submission final: Model baru hanya menggantikan ResNet-18 jika Macro-F1 lokal lebih tinggi minimal +0.005.*

---

## 6. Data & Data Tambahan

1. **Dataset Utama:**
   - Folder `train/` berisi citra 5 kelas sampah yang disediakan oleh tim pengajar.
2. **Data Tambahan Eksternal:**
   - Tidak menggunakan data eksternal tambahan dari luar dataset kompetisi untuk menjaga konsistensi domain distribusi dan memastikan kepatuhan penuh terhadap ketentuan kompetisi.
3. **Pembagian Data Validasi Lokal (`my_val/`):**
   - Dibuat menggunakan stratified random sampling sebesar 15% dari dataset `train/` menggunakan random seed `42`.
   - Subset ini diisolasi dari proses pelatihan model eksplorasi (`train.py`) guna menghasilkan skor Macro-F1 yang obyektif dan sebanding.
   - Integritas data validasi dicocokkan antar-anggota dengan verifikasi checksum MD5.

---

## 7. Cara Reproduksi

Berikut adalah panduan lengkap untuk mereproduksi hasil pelatihan dan evaluasi model:

> [!IMPORTANT]
> **Dataset `train/` tidak disertakan dalam repository ini** (diabaikan via `.gitignore`) karena ukurannya terlalu besar untuk GitHub. Unduh dataset terpisah dari portal/link yang diberikan oleh tim pengajar, lalu letakkan folder `train/` di root repository (sejajar dengan `train.py`).

### 1. Persiapan Lingkungan (Environment)

Pastikan Python 3.10+ telah terpasang, lalu pasang dependensi yang dibutuhkan:

```bash
pip install -r submission/requirements.txt
pip install timm numpy
```

### 2. Penyiapan Subset Validasi Lokal (`my_val/`)

Setelah dataset `train/` tersedia, buat subset `my_val` (15%) dan `my_train` (85%):

```bash
python prepare_split.py
```

Verifikasi konsistensi checksum data validasi di terminal:

```bash
# Linux/macOS
find my_val -type f | sort | md5sum

# Windows (PowerShell)
Get-ChildItem -Recurse my_val -File | Sort-Object FullName | Get-FileHash -Algorithm MD5
```

### 3. Pelatihan Model

- **Melatih Baseline ResNet-18:**
  ```bash
  python train_baseline.py --data train/ --epochs 8
  ```

- **Melatih Model ConvNeXt-Tiny (Anggota 1):**
  ```bash
  python train.py --data train/ --arch convnext_tiny --out sub_a
  ```

- **Melatih Model EfficientNet-B2 (Anggota 2):**
  ```bash
  python train.py --data train/ --arch efficientnet_b2 --img_size 288 --out sub_b
  ```

- **Pelatihan Final Full-Train (Jika model baru menang):**
  ```bash
  python train.py --data train/ --arch <nama_arsitektur_terbaik> --out submission_new --full
  ```

### 4. Evaluasi Model

Jalankan evaluasi pada model submission atau model eksperimen menggunakan `evaluate.py`:

```bash
# Evaluasi submission ResNet-18 saat ini
python evaluate.py --submission ./submission --data ./my_val

# Evaluasi model eksperimen EfficientNet-B2
python evaluate.py --submission ./sub_b --data ./my_val
```

---

## 8. Struktur Repository

- `train.py` — Skrip pelatihan fine-tuning backbone modern (*timm*) dengan proteksi validasi.
- `train_baseline.py` — Skrip pelatihan baseline ResNet-18 serta pembuat split `my_val`.
- `evaluate.py` — Skrip evaluasi performa submission (Macro-F1, per-class F1, matriks konfusi).
- `submission/`
  - `model.py` — Definisi arsitektur dan transformasi inferensi.
  - `model.pth` — Bobot terlatih model submission.
  - `requirements.txt` — Dependensi minimal untuk inferensi.
  - `README.md` — Panduan teknis berkas submission.
