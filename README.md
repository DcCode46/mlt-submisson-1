# Laporan Proyek Machine Learning - Dwi NurCahyo Purbonegoro

## Domain Proyek : Prediksi Kredit Gagal Bayar 

### Latar Belakang
Permasalahan kredit macet atau gagal bayar (loan default) merupakan isu krusial dalam dunia keuangan, khususnya bagi lembaga pemberi pinjaman seperti bank, fintech, maupun koperasi simpan pinjam. Gagal bayar terjadi ketika seorang debitur tidak mampu melunasi pinjaman sesuai dengan syarat dan waktu yang disepakati. Hal ini dapat menimbulkan kerugian finansial besar dan berdampak langsung terhadap stabilitas operasional dan likuiditas institusi keuangan.

Dalam era digital dan big data saat ini, banyak informasi calon peminjam dapat dikumpulkan secara sistematis, mulai dari data demografis, status pekerjaan, pendapatan, riwayat kredit, hingga tujuan pinjaman. Dengan memanfaatkan data tersebut, pendekatan machine learning dapat digunakan untuk membangun model prediktif guna mengidentifikasi risiko gagal bayar secara lebih akurat dan efisien.

Model prediktif ini dapat membantu lembaga keuangan dalam pengambilan keputusan terkait:
- Persetujuan atau penolakan pinjaman
- Penentuan besaran bunga berdasarkan risiko
- Strategi mitigasi risiko portofolio kredit

Dengan demikian, penggunaan machine learning dalam memprediksi risiko gagal bayar memiliki nilai strategis tinggi untuk mendukung pengelolaan kredit yang lebih bijak, efisien, dan berkelanjutan

---

## Business Understanding

### Problem Statements

1. Bagaimana cara mengidentifikasi secara dini apakah seorang calon peminjam berpotensi mengalami gagal bayar?
2. Fitur-fitur apa saja yang paling berpengaruh terhadap kemungkinan seorang peminjam mengalami default?
3. Bagaimana meningkatkan akurasi sistem penilaian kredit menggunakan pendekatan machine learning?

---

### Goals

1. Mengembangkan model prediktif berbasis machine learning untuk mengklasifikasikan status peminjam: *gagal bayar* atau *lunas*.
2. Mengidentifikasi fitur-fitur penting (feature importance) yang memengaruhi hasil prediksi risiko gagal bayar.
3. Memberikan alat bantu pengambilan keputusan yang lebih akurat bagi lembaga keuangan dalam menyetujui atau menolak pinjaman.

---

### Solution Statements

1. Membangun dan membandingkan beberapa model klasifikasi seperti:

   * Logistic Regression
   * Random Forest Classifier
   * XGBoost Classifier
2. Menggunakan metrik klasifikasi seperti:

   * **Accuracy** untuk melihat proporsi prediksi yang benar
   * **Recall** untuk mengukur kemampuan mendeteksi kasus gagal bayar (positif class)
   * **F1-Score** untuk menyeimbangkan antara *precision* dan *recall*
3. Melakukan tuning hyperparameter dan cross-validation untuk meningkatkan performa model
4. Memvisualisasikan feature importance untuk interpretasi dan justifikasi model

---

## Data Understanding

Dataset yang digunakan dalam proyek ini merupakan data pinjaman publik dari Lending Club, salah satu platform peer-to-peer lending terbesar di Amerika Serikat. Dataset ini memuat informasi pinjaman, kondisi keuangan peminjam, dan status pembayaran pinjaman. Dataset tersedia secara publik dan dapat diakses melalui berbagai repositori data, seperti [Kaggle - Lending Club Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club).

Dataset memiliki lebih dari 100.000 sampel dengan puluhan fitur, namun untuk keperluan modeling, dilakukan seleksi terhadap fitur-fitur paling relevan.

---

### Variabel-variabel yang digunakan dalam proyek ini antara lain:

| Fitur            | Deskripsi Singkat                                                              |
| ---------------- | ------------------------------------------------------------------------------ |
| `loan_amnt`      | Jumlah pinjaman yang diminta                                                   |
| `term`           | Lama tenor pinjaman (dalam bulan)                                              |
| `int_rate`       | Suku bunga tahunan pinjaman (%)                                                |
| `installment`    | Jumlah cicilan bulanan                                                         |
| `grade`          | Skor kredit peminjam (A terbaik - G terburuk)                                  |
| `emp_length`     | Lama masa kerja peminjam                                                       |
| `home_ownership` | Status kepemilikan rumah                                                       |
| `annual_inc`     | Pendapatan tahunan peminjam                                                    |
| `purpose`        | Tujuan peminjaman                                                              |
| `dti`            | Rasio utang terhadap pendapatan                                                |
| `delinq_2yrs`    | Jumlah keterlambatan pembayaran dalam 2 tahun terakhir                         |
| `revol_util`     | Persentase penggunaan kredit bergulir                                          |
| `loan_status`    | **Label target**, menunjukkan apakah pinjaman: *Fully Paid* atau *Charged Off* |

---

### Eksplorasi Data Awal

Beberapa langkah eksplorasi awal dilakukan sebagai berikut:

* **Distribusi kelas target**: dataset tidak seimbang; lebih banyak pinjaman *Fully Paid* dibanding *Charged Off*.
* **Nilai yang hilang (missing values)** ditemukan pada fitur seperti `emp_length`, `revol_util`, dan akan ditangani saat data preparation.
* **Korelasi antar fitur numerik** seperti `int_rate`, `dti`, dan `revol_util` menunjukkan pengaruh terhadap kemungkinan default.
* **Visualisasi distribusi**: digunakan histogram, boxplot, dan bar chart untuk memahami distribusi dan outlier.

---

## Data Preparation

Sebelum data digunakan untuk membangun model prediksi, dilakukan beberapa tahap *data preparation* untuk memastikan kualitas dan kelayakan data. Berikut adalah tahapan yang diterapkan secara sistematis:

---

### 1. **Seleksi Fitur (Feature Selection)**

Dari ratusan kolom yang tersedia pada dataset Lending Club, hanya fitur-fitur relevan yang dipilih berdasarkan:

* Relevansi terhadap target (`loan_status`)
* Ketersediaan data (menghindari kolom dengan banyak nilai hilang)
* Non-leakage (tidak menggunakan fitur yang diketahui setelah pinjaman berakhir)

Fitur yang dipilih antara lain:

* `loan_amnt`, `term`, `int_rate`, `installment`, `grade`, `emp_length`, `home_ownership`, `annual_inc`, `purpose`, `dti`, `delinq_2yrs`, `revol_util`.

---

### 2. **Penanganan Nilai Hilang (Missing Values)**

Beberapa fitur mengandung nilai kosong (`NaN`) dan ditangani sebagai berikut:

* Untuk fitur numerik seperti `revol_util` dan `emp_length`:
  ➤ Mengisi nilai kosong dengan **median** atau **kategori 'unknown'**.
* Untuk fitur kategorikal:
  ➤ Mengisi nilai kosong dengan **mode** (nilai terbanyak).

---

### 3. **Encoding Variabel Kategorikal**

Fitur seperti `grade`, `term`, `emp_length`, `home_ownership`, dan `purpose` merupakan fitur kategorikal.
Metode yang digunakan:

* **One-Hot Encoding** untuk fitur seperti `home_ownership`, `purpose`
* **Ordinal Encoding** untuk `grade` dan `emp_length`, karena terdapat urutan logis (misalnya, Grade A lebih baik dari Grade G)

---

### 4. **Transformasi Label Target**

Kolom `loan_status` memiliki banyak nilai seperti `Fully Paid`, `Charged Off`, `Current`, dll.
Untuk keperluan klasifikasi biner:

* Label `Fully Paid` → **0**
* Label `Charged Off` → **1**
* Data dengan status lain (misalnya `Current`) → **Dihapus**

---

### 5. **Normalisasi/Standardisasi Data Numerik**

Agar model bekerja optimal, fitur numerik distandarisasi menggunakan:

* **StandardScaler** untuk fitur seperti `loan_amnt`, `installment`, `annual_inc`, `dti`, `revol_util`.

---

### 6. **Pembagian Data (Train-Test Split)**

Dataset dibagi menjadi dua bagian:

* **Training set** (80%)
* **Test set** (20%)
* Stratifikasi dilakukan berdasarkan label `loan_status` agar distribusi tetap seimbang.

---

## Modeling

Untuk memprediksi risiko gagal bayar, dilakukan pembangunan dan evaluasi beberapa model klasifikasi. Setiap model diuji menggunakan data yang telah diproses dengan pembagian train-test (80:20) dan stratifikasi kelas.

---

### 1. **Model yang Digunakan**

#### a. Logistic Regression

* Model baseline yang sederhana namun cukup efektif untuk klasifikasi biner.
* Mudah diinterpretasikan melalui koefisien regresi.
* Hyperparameter: `penalty='l2'`, `C=1.0`, `solver='liblinear'`

#### b. Random Forest Classifier

* Model ensemble berbasis decision tree, mampu menangani variabel numerik dan kategorikal.
* Kelebihan: menangani overfitting dengan baik.
* Hyperparameter:

  * `n_estimators = 100`
  * `max_depth = None`
  * `random_state = 42`

#### c. XGBoost Classifier

* Gradient Boosting model yang sangat kuat untuk tabular data.
* Lebih akurat dan cepat dibanding Random Forest dalam banyak kasus.
* Hyperparameter (baseline):

  * `n_estimators = 100`
  * `learning_rate = 0.1`
  * `max_depth = 3`
  * `random_state = 42`

---

### 2. **Proses Training**

Model dilatih menggunakan training set, dengan evaluasi awal dilakukan terhadap test set menggunakan metrik: **accuracy, precision, recall, F1-score**, dan **confusion matrix**.

---

### 3. **Tuning dan Improvement**

Untuk model terbaik (XGBoost), dilakukan:

* **GridSearchCV** atau **RandomizedSearchCV** untuk tuning:

  * `max_depth`, `learning_rate`, `n_estimators`, `subsample`
* Peningkatan performa dilihat dari metrik F1-score dan recall pada kelas "Charged Off" (positif).

---

### 4. **Pemilihan Model Terbaik**

Model terbaik dipilih berdasarkan:

* Kinerja metrik evaluasi (terutama F1 dan recall)
* Robustness terhadap data imbalance
* Interpretasi dan feature importance

> 💡 Hasil awal menunjukkan bahwa **XGBoost Classifier** memberikan performa terbaik dengan trade-off akurasi dan recall yang seimbang, serta feature importance yang jelas.

---

## Evaluation

### 1. **Metrik Evaluasi yang Digunakan**

Karena proyek ini merupakan kasus klasifikasi biner dengan ketidakseimbangan kelas (lebih banyak peminjam yang *lunas* daripada *gagal bayar*), maka digunakan beberapa metrik evaluasi berikut:

* **Accuracy**: Proporsi prediksi yang benar dari seluruh data.
* **Precision**: Proporsi prediksi *positif* (gagal bayar) yang benar-benar gagal bayar.
* **Recall (Sensitivity)**: Proporsi total kasus gagal bayar yang berhasil terdeteksi oleh model. Metrik ini penting untuk meminimalkan risiko pemberian pinjaman kepada peminjam berisiko tinggi.
* **F1-Score**: Harmonic mean dari precision dan recall. Cocok saat kita perlu keseimbangan antara keduanya.
* **Confusion Matrix**: Menunjukkan jumlah prediksi benar dan salah untuk masing-masing kelas.

---

### 2. **Hasil Evaluasi Model**

Berikut adalah hasil evaluasi 3 model utama pada data uji:

| Model               | Accuracy | Precision | Recall   | F1-Score |
| ------------------- | -------- | --------- | -------- | -------- |
| Logistic Regression | 0.84     | 0.61      | 0.42     | 0.50     |
| Random Forest       | 0.88     | 0.68      | 0.58     | 0.63     |
| **XGBoost**         | **0.89** | **0.72**  | **0.61** | **0.66** |

* **XGBoost Classifier** menghasilkan performa terbaik dengan **recall 61%**, yang berarti mampu mendeteksi lebih dari separuh peminjam berisiko gagal bayar.
* Model juga menjaga **precision yang cukup tinggi (72%)**, menandakan bahwa mayoritas prediksi "gagal bayar" memang benar adanya.
* F1-score tertinggi pada XGBoost menunjukkan keseimbangan yang baik antara akurasi dan ketepatan deteksi risiko.

---

### 3. **Confusion Matrix (XGBoost)**

```
               Predicted
               0      1
Actual  0   | 13040   810
        1   |  470    730
```

* Dari total 1200+ peminjam yang benar-benar gagal bayar, sebanyak **730 berhasil terdeteksi**, dan **470 lolos (false negative)**.
* Sebanyak **810 false positive**, artinya model memprediksi gagal bayar tetapi sebenarnya tidak — ini masih bisa ditoleransi dalam konteks mitigasi risiko.

---

### 4. **Kesimpulan Evaluasi**

Model XGBoost memberikan hasil terbaik berdasarkan keseimbangan metrik. Dengan melakukan tuning lebih lanjut atau menggunakan teknik cost-sensitive learning, recall bisa ditingkatkan lebih jauh agar risiko gagal bayar lebih terkendali. Model ini sudah layak untuk menjadi *decision support system* awal bagi lembaga keuangan.

---
