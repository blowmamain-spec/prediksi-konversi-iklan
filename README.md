# Prediksi Konversi Iklan

Aplikasi web interaktif untuk memprediksi apakah seorang pengguna media sosial akan membeli produk yang diiklankan — dibangun dengan **Streamlit** dan tiga algoritma Machine Learning.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Tentang Proyek

Dataset yang digunakan adalah **Social Network Ads** — berisi data demografis pengguna media sosial beserta label apakah mereka melakukan pembelian setelah melihat iklan. Tujuan proyek ini adalah membandingkan performa tiga algoritma klasifikasi dalam memprediksi konversi iklan.

---

## Fitur Aplikasi

| Tab | Deskripsi |
|---|---|
| **Eksplorasi Data** | Preview dataset dan visualisasi distribusi kelas (Beli / Tidak Beli) |
| **Kinerja Model** | Evaluasi akurasi & *classification report* setiap model secara interaktif |
| **Coba Prediksi** | Input data audiens baru dan dapatkan hasil prediksi secara langsung |

---

## Algoritma yang Digunakan

| Model | Hyperparameter Tuning | Keterangan |
|---|---|---|
| K-Nearest Neighbors (KNN) | GridSearchCV (`n_neighbors`: 5, 7) | Berbasis jarak antar titik data |
| Decision Tree | GridSearchCV (`max_depth`: 3, 5) | Model berbasis pohon keputusan |
| Naïve Bayes (Gaussian) | — | Berbasis probabilitas Bayes |

> Ketidakseimbangan kelas ditangani menggunakan **SMOTE** (*Synthetic Minority Over-sampling Technique*).

---

## Struktur Dataset

**File:** `Social_Network_Ads.csv` — 400 baris data

| Kolom | Tipe | Keterangan |
|---|---|---|
| `User ID` | int | ID unik pengguna |
| `Gender` | str | Jenis kelamin (`Male` / `Female`) |
| `Age` | int | Usia pengguna (18–60 tahun) |
| `EstimatedSalary` | int | Estimasi gaji tahunan (USD) |
| `Purchased` | int | Label target: `1` = Beli, `0` = Tidak Beli |

---

## Instalasi & Cara Menjalankan

**1. Clone repository**
```bash
git clone https://github.com/blowmamain-spec/prediksi-konversi-iklan.git
cd prediksi-konversi-iklan
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Jalankan aplikasi**
```bash
streamlit run app2.py
```

## Requirements

```
streamlit>=1.30.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
imbalanced-learn>=0.11.0
```

## Struktur Proyek

```
prediksi-konversi-iklan/
├── app2.py                  # Aplikasi utama Streamlit
├── Social_Network_Ads.csv   # Dataset
├── requirements.txt         # Daftar dependensi
└── README.md
```

---

> Dibuat sebagai tugas mata kuliah **Sistem Cerdas**  
> Program Teknologi Informasi · Telkom University
