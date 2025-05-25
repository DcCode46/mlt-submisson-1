# Laporan Proyek Machine Learning - Dwi NurCahyo Purbonegoro

## Domain Proyek: Prediksi Harga Mobil Bekas
![image](https://github.com/user-attachments/assets/8991718b-7ea3-4fbc-a2ed-802a9ee3b983)

### Latar Belakang
Harga mobil bekas sangat dipengaruhi oleh beberapa faktor utama seperti **merk mobil**, **jarak tempuh**, dan **umur mobil**. Dalam banyak kasus, mobil dengan **jarak tempuh rendah**, **merk premium**, atau **mobil klasik** dengan jumlah terbatas memiliki harga jual yang lebih tinggi.

Melalui proyek ini, dibangun sistem prediksi harga mobil berbasis machine learning untuk:
- Memberikan estimasi harga berdasarkan data spesifik mobil
- Membantu pembeli/penjual menentukan harga yang wajar
- Meningkatkan efisiensi dalam jual beli mobil bekas

---

## Business Understanding

### Problem Statements
1. Bagaimana memprediksi harga mobil berdasarkan merk, jarak tempuh, dan umur mobil?
2. Bagaimana mengidentifikasi pengaruh kombinasi fitur tersebut terhadap nilai jual kendaraan?

### Goals
1. Membangun model prediksi harga mobil bekas yang akurat.
2. Menerapkan preprocessing dan rekayasa fitur untuk menangkap insight penting.
3. Mengoptimalkan model regresi berbasis pohon seperti Random Forest dan XGBoost.

---

## Data Understanding

Dataset berisi informasi spesifikasi mobil bekas. Fitur penting yang digunakan:
- `Manufacturer` → untuk menentukan merk
- `Mileage` → total jarak tempuh (km)
- `Prod. year` → tahun produksi
- `Fuel type`, `Gear box type`, `Airbags`, `Category`, dll.

Beberapa fitur direkayasa:
- `Car_Age` = 2025 - `Prod. year`
- `Mileage_per_Year` = `Mileage` / `Car_Age`
- `Is_Luxury` → flag jika merk termasuk BMW, AUDI, TESLA, dll.
- `Is_Classic` → mobil dengan usia lebih dari 20 tahun

---

## Data Preparation

### Langkah-langkah yang dilakukan:
- **Handling Missing Value** pada `Levy` menggunakan median.
- **Pembersihan string numerik** seperti `Mileage`, `Levy`, dan `Engine volume`.
- **Encoding kategorikal**: 
  - Label Encoding untuk `brand`, `model`, dan `Fuel type`
  - One-Hot Encoding hanya untuk beberapa kategori besar
- **Feature Scaling** menggunakan StandardScaler pada `Mileage`, `Car_Age`, `Mileage_per_Year`
- **Transformasi target `Price`** dengan `log1p` untuk mengatasi distribusi skewed

---

## Modeling

### Model yang digunakan:
- **Random Forest Regressor**: model pohon acak yang tahan outlier
- **XGBoost Regressor**: model boosting populer untuk prediksi tabular

### Evaluasi:
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **R²**: Koefisien determinasi

### Tuning:
GridSearchCV ringan dilakukan dengan kombinasi kecil dari:
- `n_estimators`, `max_depth` untuk RandomForest
- `learning_rate`, `colsample_bytree`, `max_depth` untuk XGBoost

---

## Hasil Evaluasi

| Model                  | MAE     | RMSE    | R² Score |
|------------------------|---------|---------|----------|
|             Linear Regression           | ~0.946485           | ~1.367330        | ~0.285271         |
| Random Forest Regressor| ~0.442549   | ~0.893156   | ~0.695035    |
| XGBoost Regressor      | ~0.594116   | ~0.988063   | ~0.626781    |

Model **XGBoost** memberikan hasil terbaik dan digunakan sebagai model final.

---

## Kesimpulan

Model prediksi harga mobil berhasil dibuat berdasarkan fitur-fitur kunci: **merk mobil**, **jarak tempuh**, dan **usia kendaraan**. Feature engineering memainkan peran penting untuk meningkatkan akurasi prediksi.

### Rekomendasi:
- Gunakan dataset yang lebih besar dengan fitur tambahan seperti kondisi kendaraan.
- Bangun API prediksi untuk digunakan pada platform jual beli mobil.
- Lakukan integrasi scraping data real-time dari situs jual-beli kendaraan.
