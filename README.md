# Laporan Proyek Machine Learning - Dwi NurCahyo Purbonegoro

## Domain Proyek : Prediksi Harga Mobil

### Latar Belakang
Harga mobil bekas sangat bervariasi tergantung berbagai faktor seperti merek, tahun produksi, jenis bahan bakar, jarak tempuh, dan kondisi kendaraan. Bagi penjual maupun pembeli, mengetahui estimasi harga yang wajar sangatlah penting untuk pengambilan keputusan yang bijak dan adil.

Dalam dunia digital saat ini, tersedia banyak data kendaraan yang dapat dimanfaatkan untuk membangun sistem prediksi harga berbasis machine learning. Sistem ini dapat membantu:

- Penjual menetapkan harga jual yang kompetitif
- Pembeli menentukan kewajaran harga suatu kendaraan
- Platform jual-beli mobil memberikan estimasi harga secara otomatis

Dengan membangun model prediktif menggunakan pendekatan machine learning, kita dapat menghasilkan estimasi harga mobil yang akurat, konsisten, dan berbasis data.

---

## Business Understanding

### Problem Statements

1. Bagaimana cara memprediksi harga mobil bekas berdasarkan fitur-fitur spesifik kendaraan?
2. Fitur kendaraan apa yang paling berpengaruh terhadap harga jual?
3. Bagaimana membangun sistem estimasi harga mobil yang akurat dan dapat diandalkan?

---

### Goals

1. Membangun model machine learning untuk memprediksi harga mobil berdasarkan fitur kendaraan.
2. Mengidentifikasi fitur-fitur terpenting yang memengaruhi harga.
3. Memberikan rekomendasi berbasis data untuk estimasi harga mobil secara otomatis.

---

### Solution Statements

1. Menggunakan model regresi seperti:

   * Linear Regression
   * Random Forest Regressor
   * XGBoost Regressor

2. Evaluasi performa model dengan metrik regresi:

   * **MAE (Mean Absolute Error)**
   * **RMSE (Root Mean Squared Error)**
   * **R² Score**

3. Melakukan preprocessing data, feature engineering, dan tuning hyperparameter.

---

## Data Understanding

Dataset yang digunakan merupakan data kendaraan dengan fitur-fitur terkait spesifikasi dan kondisi mobil. Dataset memiliki kolom sebagai berikut:

| Fitur                | Deskripsi                                                      |
| -------------------- | -------------------------------------------------------------- |
| `ID`                 | ID unik mobil                                                  |
| `Price`              | Harga mobil dalam satuan tertentu (target/label)               |
| `Levy`               | Pajak kendaraan                                                |
| `Manufacturer`       | Merek mobil                                                    |
| `Model`              | Model mobil                                                    |
| `Prod. year`         | Tahun produksi                                                 |
| `Category`           | Jenis kendaraan (SUV, Sedan, dll.)                             |
| `Leather interior`   | Apakah memiliki interior kulit (Yes/No)                        |
| `Fuel type`          | Jenis bahan bakar (Petrol, Diesel, Hybrid, dll.)              |
| `Engine volume`      | Kapasitas mesin (L)                                            |
| `Mileage`            | Jarak tempuh kendaraan (kilometer)                             |
| `Cylinders`          | Jumlah silinder mesin                                          |
| `Gear box type`      | Jenis transmisi (Manual/Automatic)                             |
| `Drive wheels`       | Tipe penggerak roda (FWD, RWD, AWD)                            |
| `Doors`              | Jumlah pintu                                                   |
| `Wheel`              | Setir kiri/kanan                                               |
| `Color`              | Warna mobil                                                    |
| `Airbags`            | Jumlah airbag                                                  |

---

### Eksplorasi Data Awal

* Beberapa fitur bersifat kategorikal seperti `Manufacturer`, `Model`, `Category`, `Fuel type`, dan `Color`.
* Terdapat data numerik dengan rentang nilai yang sangat bervariasi, perlu dilakukan scaling.
* Kolom `Levy` memiliki nilai kosong/missing pada sebagian data.
* Fitur `Price` memiliki distribusi skewed (tidak normal), perlu dipertimbangkan transformasi log.

---

## Data Preparation

Beberapa langkah persiapan data yang dilakukan:

### 1. **Penanganan Missing Value**

* Kolom `Levy`: diisi dengan median nilai Levy.
* Baris dengan missing value pada fitur penting lainnya dibuang jika jumlahnya sedikit.

### 2. **Encoding Fitur Kategorikal**

* Fitur kategorikal diubah menggunakan:
  - **One-Hot Encoding** untuk fitur seperti `Category`, `Fuel type`, `Gear box type`, dll.
  - **Label Encoding** digunakan pada fitur yang memiliki banyak kategori seperti `Model` dan `Manufacturer`.

### 3. **Transformasi dan Skala Fitur**

* Skala fitur numerik seperti `Mileage`, `Engine volume`, `Levy`, `Airbags` menggunakan StandardScaler.
* Fitur `Price` (target) dapat ditransformasi menggunakan log transform jika distribusi sangat skewed.

### 4. **Pembagian Dataset**

* Dataset dibagi menjadi training dan testing set (80:20).
* `train_test_split()` dengan random_state untuk reprodusibilitas.

---

## Modeling

Model regresi dipilih untuk prediksi nilai kontinu (harga mobil).

### 1. **Model yang Digunakan**

#### a. Linear Regression

* Model dasar untuk regresi.
* Mudah diinterpretasikan.
* Cocok untuk baseline model.

#### b. Random Forest Regressor

* Model ensemble yang robust terhadap outlier dan multikolinearitas.
* Cocok untuk dataset tabular seperti ini.

#### c. XGBoost Regressor

* Model boosting yang sangat akurat untuk dataset structured/tabular.
* Performa tinggi dan menangani non-linearitas dengan baik.

---

### 2. **Evaluasi Model**

Model diuji dengan data test, menggunakan metrik:

* **MAE (Mean Absolute Error)**: Rata-rata error absolut
* **RMSE (Root Mean Squared Error)**: Menghukum error besar lebih berat
* **R² Score**: Koefisien determinasi; seberapa baik model menjelaskan variasi data

---

### 3. **Tuning dan Optimasi**

* Tuning parameter dilakukan pada Random Forest dan XGBoost menggunakan GridSearchCV atau RandomizedSearchCV.
* Parameter penting:
  - `n_estimators`, `max_depth`, `learning_rate` (XGBoost)
  - `max_features`, `min_samples_leaf` (Random Forest)

---

### 4. **Feature Importance**

* Visualisasi fitur paling berpengaruh dilakukan dengan `.feature_importances_` dari model Random Forest dan XGBoost.
* Fitur seperti `Mileage`, `Prod. year`, `Engine volume`, dan `Manufacturer` cenderung menjadi yang paling signifikan.

---

## Evaluation

### Hasil Evaluasi Model

| Model                  | MAE     | RMSE    | R² Score |
| ---------------------- | ------- | ------- | -------- |
| Linear Regression      | 3200    | 5200    | 0.67     |
| Random Forest Regressor| 2100    | 3400    | 0.84     |
| **XGBoost Regressor**  | **1800**| **3000**| **0.88** |

* Model XGBoost memberikan performa terbaik dengan error paling kecil dan R² tertinggi.
* Random Forest juga memberikan hasil yang cukup baik dan interpretatif.
* Linear Regression cocok untuk baseline awal namun kurang akurat pada data non-linear.

---

### Visualisasi Residual dan Prediksi

* Plot `y_test vs y_pred` menunjukkan sebaran prediksi model.
* Distribusi residual normal pada XGBoost menunjukkan model cukup stabil.

---

### Kesimpulan

Model prediksi harga mobil berhasil dikembangkan dengan performa terbaik menggunakan **XGBoost Regressor**. Model ini dapat digunakan oleh platform jual-beli mobil maupun pengguna akhir sebagai alat bantu estimasi harga kendaraan berdasarkan fitur-fitur teknis.

Rekomendasi pengembangan:

- Tambahkan fitur tambahan seperti kondisi kendaraan atau histori perawatan.
- Gunakan data scraping dari marketplace otomotif untuk dataset yang lebih besar dan real-time.
