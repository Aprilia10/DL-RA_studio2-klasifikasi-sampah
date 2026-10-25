# Studio Proyek 2 — Eksperimen Deep Learning untuk Klasifikasi Sampah

**Mata Kuliah:** Deep Learning (IF25-40401)  
**Program Studi:** Teknik Informatika, Institut Teknologi Sumatera  
**Semester:** Ganjil 2026/2027

Proyek kompetisi klasifikasi lima jenis sampah menggunakan *fine-tuning* model *pretrained* ImageNet untuk membandingkan performa berdasarkan Macro-F1.

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
| 3 | `saset_kemasan` | Bungkus saset plastik/aluminium foil (makanan/minuman) |
| 4 | `styrofoam` | Wadah makanan berbahan busa polistirena |

Metrik penilaian utama adalah **Macro-F1** pada set validasi tersembunyi milik tim pengajar.

---

## 3. Ringkasan Pendekatan

**Model submission final: ConvNeXt-Tiny (*timm*), fine-tuning penuh dari bobot pretrained ImageNet, input 224×224.**

| Komponen | Implementasi |
|---|---|
| **Arsitektur Model** | Baseline ResNet-18 (*torchvision*) dan ConvNeXt-Tiny (*timm*); last layer diganti menjadi 5 kelas |
| **Strategi Pelatihan** | Full fine-tuning dengan pemisahan learning rate: *head* 3e-4 dan *backbone* 3e-5 (`bb_mult` 0,1); AdamW (weight decay 0,05); 12 epoch |
| **Augmentasi Citra** | Random Resized Crop, Horizontal Flip, Color Jitter, Random Rotation, TrivialAugmentWide, dan Random Erasing |
| **Regularisasi & Loss** | Label Smoothing Cross-Entropy (`smooth=0.1`, membantu menghadapi label noise), AdamW, dan Cosine Annealing LR Scheduler |
| **Protokol Validasi** | Split lokal 15% dari data latih (`my_val/`, 410 gambar) dengan seed tetap; gambar `my_val` dikeluarkan dari latihan agar skor tidak bocor (*data leakage*) |
| **Run Final** | ConvNeXt-Tiny dilatih ulang pada seluruh `train/` (`--full`), epoch terakhir disimpan |
| **Ukuran Bobot** | Disimpan dalam fp16 (±54 MB) agar di bawah batas 100 MB GitHub; dimuat kembali ke fp32 oleh `load_state_dict` |

---

## 4. Pemantauan Hasil Kompetisi

Evaluasi dilakukan dua putaran preview dan satu evaluasi Final yang menentukan nilai.

| Tahap Evaluasi | Jadwal | Macro-F1 | Status / Model |
|:---|:---:|:---:|:---|
| Preview 1 | 14 Oktober 2026 | - | |
| Preview 2 | 21 Oktober 2026 | 0.9901 | ResNet-18 (Pretrained) |
| Final | 25 Oktober 2026 | 0.9722 | ConvNeXt-Tiny (fp16) |

---

## 5. Hasil Evaluasi & Perbandingan Model

Skor lokal dihitung dengan `python evaluate.py --submission <folder> --data ./my_val` pada `my_val` yang sama (410 gambar).

### 5.1 Hasil pada `my_val`

| Model | Input | Macro-F1 | Akurasi | Keterangan |
|---|:---:|:---:|:---:|---|
| ResNet-18 (Baseline) | 224×224 | 0,9378 | 0,9415 | `train_baseline.py`, 8 epoch; submission Preview 2 (±44 MB) |
| **ConvNeXt-Tiny** | 224×224 | **0,9703** | **0,9732** | `train.py`, 12 epoch, `my_val` tidak ikut dilatih |

**F1 per kelas:**

| Model | botol_plastik | gelas_plastik | kertas_kardus | saset_kemasan | styrofoam |
|---|:---:|:---:|:---:|:---:|:---:|
| ResNet-18 | 0,9610 | 0,9620 | 0,8846 | 0,9442 | 0,9372 |
| ConvNeXt-Tiny | 0,9873 | 0,9873 | 0,9320 | 0,9852 | 0,9596 |

**Confusion matrix** (baris = label asli, kolom = prediksi):

| ResNet-18 | botol | gelas | kertas | saset | styrofoam |
|---|:---:|:---:|:---:|:---:|:---:|
| botol_plastik | 74 | 1 | 0 | 2 | 1 |
| gelas_plastik | 0 | 76 | 1 | 0 | 1 |
| kertas_kardus | 0 | 2 | 46 | 1 | 5 |
| saset_kemasan | 2 | 0 | 2 | 93 | 3 |
| styrofoam | 0 | 1 | 1 | 1 | 97 |

| ConvNeXt-Tiny | botol | gelas | kertas | saset | styrofoam |
|---|:---:|:---:|:---:|:---:|:---:|
| botol_plastik | 78 | 0 | 0 | 0 | 0 |
| gelas_plastik | 0 | 78 | 0 | 0 | 0 |
| kertas_kardus | 1 | 1 | 48 | 1 | 3 |
| saset_kemasan | 0 | 0 | 0 | 100 | 0 |
| styrofoam | 1 | 1 | 1 | 2 | 95 |

### 5.2 Analisis singkat

- ConvNeXt-Tiny mencapai Macro-F1 0,9703 pada `my_val` (410 gambar), di atas baseline ResNet-18 (0,9378). Selisih +0,0325 jauh di atas ambang adopsi +0,005, sehingga ConvNeXt-Tiny dipilih sebagai submission final.
- `kertas_kardus` adalah kelas tersulit pada kedua model (F1 0,8846 pada ResNet-18 dan 0,9320 pada ConvNeXt-Tiny). Kesalahan terbanyaknya adalah `kertas_kardus` yang terprediksi sebagai `styrofoam` (5 gambar pada ResNet-18, 3 gambar pada ConvNeXt-Tiny).
- ConvNeXt-Tiny memperbaiki kelima kelas dibanding ResNet-18, dengan kenaikan terbesar pada `saset_kemasan` (+0,0410) dan `kertas_kardus` (+0,0474).
- Validasi lokal hanya 410 gambar, sehingga selisih kecil (di bawah ±1 poin) belum tentu signifikan.
- Eksperimen EfficientNet-B2 tidak dilaporkan karena log evaluasi yang dapat diverifikasi tidak tersedia.

> **Catatan:** bobot yang dikumpulkan di `submission/` berasal dari run `--full` (seluruh `train/`, termasuk `my_val`), sehingga skor `my_val` pada bobot tersebut (±1,0) **tidak bermakna** dan hanya membuktikan submission dapat dijalankan. Skor 0,9703 berasal dari run terpisah yang tidak melatih `my_val`.

*Aturan adopsi submission final: model baru menggantikan ResNet-18 hanya jika Macro-F1 lokal lebih tinggi minimal +0,005.*

---

## 6. Data & Data Tambahan

1. **Dataset Utama:** folder `train/` berisi ±2.750 citra 5 kelas dari tim pengajar.
2. **Data Tambahan Eksternal:** tidak menggunakan data eksternal dari luar dataset kompetisi, untuk menjaga konsistensi domain dan kepatuhan terhadap ketentuan kompetisi.
3. **Validasi Lokal (`my_val/`):**
   - Dibuat dengan `prepare_split.py` (15% dari `train/`, seed tetap), menghasilkan 410 gambar.
   - `train.py` mengeluarkan gambar `my_val` dari latihan (kecuali run `--full`) agar skor Macro-F1 objektif dan sebanding.
   - Integritas data validasi dicocokkan antar-anggota dengan checksum MD5.

---

## 7. Cara Reproduksi

> **Dataset `train/` tidak disertakan dalam repository ini** (diabaikan via `.gitignore`). Unduh dari tautan tim pengajar, lalu letakkan folder `train/` di root repository (sejajar dengan `train.py`).

### 1. Persiapan Lingkungan

```bash
pip install -r submission/requirements.txt   # timm
pip install numpy
```

### 2. Penyiapan Validasi Lokal

```bash
python prepare_split.py

# Verifikasi checksum my_val
find my_val -type f | sort | md5sum                                                # Linux/macOS
Get-ChildItem -Recurse my_val -File | Sort-Object FullName | Get-FileHash -Algorithm MD5   # PowerShell
```

### 3. Pelatihan

```bash
# Baseline ResNet-18 (lihat skrip untuk argumen)
python train_baseline.py

# ConvNeXt-Tiny (my_val tidak ikut dilatih)
python train.py --data train/ --arch convnext_tiny --out sub_a

# Run final pada seluruh train/
python train.py --data train/ --arch convnext_tiny --out submission_new --full
```

### 4. Konversi Bobot ke fp16 dan Pasang ke `submission/`

```bash
python - <<'PY'
import torch
sd = torch.load("submission_new/model.pth", map_location="cpu")
torch.save({k: v.half() if v.is_floating_point() else v for k, v in sd.items()}, "submission/model.pth")
PY
cp submission_new/model.py submission/model.py
cp submission_new/requirements.txt submission/requirements.txt
```

### 5. Evaluasi

```bash
python evaluate.py --submission ./sub_a --data ./my_val
python evaluate.py --submission ./submission --data ./my_val   # skor ±1,0 karena run --full
```

---

## 8. Struktur Repository

- `train.py` — pelatihan fine-tuning backbone *timm*; membuat `model.py` kontrak submission.
- `train_baseline.py` — baseline ResNet-18.
- `prepare_split.py` — membuat `my_val/` (15%) dan `my_train/` (85%) dari `train/`.
- `evaluate.py` — evaluasi (Macro-F1, F1 per kelas, confusion matrix).
- `submission/`
  - `model.py` — `get_model()` (arsitektur saja) dan `get_transform()`.
  - `model.pth` — bobot ConvNeXt-Tiny (fp16, ±54 MB).
  - `requirements.txt` — dependensi inferensi (`timm`).
  - `README.md` — catatan berkas submission.
