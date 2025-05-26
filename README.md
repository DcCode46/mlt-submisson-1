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

Data Understanding atau pemahaman data merupakan tahap untuk mengerti isi dari data yang dimiliki serta menilai sejauh mana kualitas data tersebut dapat mendukung proses analisis

Sumber Dataset : https://www.kaggle.com/datasets/deepcontractor/car-price-prediction-challenge

Dataset yang digunakan merupakan data kendaraan dengan fitur-fitur terkait spesifikasi dan kondisi mobil.

Dataset memiliki kolom sebagai berikut:

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
| `Fuel type`          | Jenis bahan bakar (Petrol, Diesel, Hybrid, dll.)               |
| `Engine volume`      | Kapasitas mesin (L)                                            |
| `Mileage`            | Jarak tempuh kendaraan (kilometer)                             |
| `Cylinders`          | Jumlah silinder mesin                                          |
| `Gear box type`      | Jenis transmisi (Manual/Automatic)                             |
| `Drive wheels`       | Tipe penggerak roda (FWD, RWD, AWD)                            |
| `Doors`              | Jumlah pintu                                                   |
| `Wheel`              | Setir kiri/kanan                                               |
| `Color`              | Warna mobil                                                    |
| `Airbags`            | Jumlah airbag                                                  |

### Eksplorasi Data Awal

* Fitur-fitur seperti Manufacturer, Model, Category, Leather interior, Fuel type, Gear box type, Drive wheels, Doors, Wheel, dan Color merupakan data kategorikal yang perlu dikodekan (encoding) sebelum digunakan dalam pemodelan.

* Kolom numerik seperti Price, Prod. year, Cylinders, dan Airbags memiliki rentang nilai yang berbeda-beda, sehingga perlu dilakukan normalisasi atau scaling untuk meningkatkan kinerja algoritma machine learning.

* Kolom Levy bertipe object dan mengandung nilai kosong atau simbol (-) yang perlu dibersihkan serta dikonversi ke numerik.

* Kolom Engine volume mengandung kombinasi angka dan teks (misalnya, '2.0 Turbo') yang perlu dipisahkan dan dibersihkan sebelum digunakan.

* Nilai pada kolom Mileage masih mengandung satuan seperti "km", sehingga perlu dibersihkan dan diubah ke format numerik.

* Kolom Doors mengandung nilai yang tidak konsisten seperti '04-May', sehingga perlu diproses untuk mendapatkan jumlah pintu yang valid.

* Distribusi nilai pada fitur Price menunjukkan skewness (kemiringan), sehingga transformasi logaritmik dapat dipertimbangkan untuk menstabilkan variansi dan mendekatkan distribusi ke normal.

![image](https://github.com/user-attachments/assets/fa89b0dc-7685-49cf-852d-a3785268d92a)

Dataset Tidak Memiliki Missing Value

![image](https://github.com/user-attachments/assets/47ab3953-64b0-4539-8d44-5bb4c536abfa)

Dataset memiliki data yang terduplikat sebanyak 313 daan semua kolom memiliki data yang sesuai yaitu 19237 yang memiliki tipe data object, integer, dan float

![image](https://github.com/user-attachments/assets/d453b9c5-e7a9-47f9-aff1-60d10b1856e3)

Output df.nunique() menunjukkan jumlah nilai unik (distinct) pada setiap kolom dalam DataFrame. Misalnya, kolom ID memiliki 18.924 nilai unik, menandakan setiap baris mewakili entri yang berbeda. Kolom Price memiliki 2.315 nilai unik, menunjukkan variasi harga kendaraan yang cukup tinggi. Kolom seperti Manufacturer dan Model masing-masing memiliki 65 dan 1.590 nilai unik, mencerminkan banyaknya merek dan tipe mobil. Beberapa kolom memiliki sedikit variasi, seperti Leather interior (2 nilai: ya/tidak) dan Wheel (2 nilai: kiri/kanan). Sementara itu, kolom seperti Fuel type (7 nilai) dan Color (16 nilai) menunjukkan variasi kategori yang sedang. Data ini penting untuk memahami keragaman informasi dalam setiap fitur sebelum dilakukan analisis lebih lanjut.

![image](https://github.com/user-attachments/assets/a76d9c73-aec8-4b37-8c64-ec14d7b533c0)

Output dari kode df.describe() memberikan ringkasan statistik untuk kolom numerik. Contohnya, Price memiliki nilai rata-rata sekitar 18.556, median 13.172, dan nilai maksimum sangat tinggi (26 juta), menunjukkan adanya outlier. Prod. year rata-rata adalah 2010, dengan rentang dari 1939 hingga 2020. Cylinders paling umum bernilai 4, sedangkan maksimum mencapai 16. Kolom Airbags juga bervariasi dari 0 hingga 16, dengan rata-rata sekitar 6. Data ini membantu memahami penyebaran, tren, dan potensi anomali didalam dataset.

![image](https://github.com/user-attachments/assets/8af1b168-1516-4c12-977d-dc1cfa7856dd)

Kode ini menghasilkan output yang menampilkan semua kategori unik pada kolom-kolom kategorikal. Misalnya, kolom Levy menyimpan nilai pajak kendaraan yang awalnya berupa string dan telah dibersihkan. Kolom Manufacturer berisi berbagai merek mobil seperti 'TOYOTA', 'BMW', 'LEXUS', hingga 'FERRARI'. Kolom Model memuat ribuan nama model kendaraan secara spesifik. Kolom Category menunjukkan jenis bodi mobil seperti 'Sedan', 'Jeep', 'Hatchback', dan lainnya.

Kolom Leather interior hanya memiliki dua kemungkinan nilai: 'Yes' dan 'No', yang menandakan apakah interior kendaraan dilapisi kulit. Fuel type mencakup berbagai jenis bahan bakar seperti 'Petrol', 'Diesel', 'Hybrid', hingga 'Hydrogen'. Kolom Gear box type menunjukkan tipe transmisi kendaraan seperti 'Automatic' dan 'Manual', sedangkan Drive wheels menampilkan sistem penggerak roda seperti 'Front', 'Rear', atau '4x4'.

Kolom Doors berisi representasi jumlah pintu dalam format tidak standar seperti '04-May' atau '>5'. Wheel menunjukkan posisi kemudi kendaraan, apakah di sebelah kiri atau kanan. Kolom Color memuat berbagai pilihan warna kendaraan. Sementara itu, Engine volume mencampurkan kapasitas mesin dan indikasi adanya turbo dalam satu kolom, seperti '2.0 Turbo' atau '3.5'. Kolom Mileage menunjukkan jarak tempuh kendaraan yang semula berbentuk string seperti '125,000 km'.

Output dari kode ini sangat membantu untuk memahami distribusi dan ragam nilai dalam kolom kategorikal sebelum melakukan proses preprocessing seperti encoding atau transformasi lainnya.

![image](https://github.com/user-attachments/assets/ac126f25-a970-4586-8f94-6f90f5a8482f)
Kode diatas menghasilkan ouput untuk melihat 5 baris pertama dari dataset yang berisi data mobil bekas.

---

## Data Preparation

### Langkah-langkah yang dilakukan:
- **Menyalin dataset**
  agar tidak mengubah data asli saat dilakukan data preparation.

![image](https://github.com/user-attachments/assets/f5766bcd-b61a-48da-af4a-319dbbc03621)
- **Hapus Duplikasi Data**
  Pada tahap ini, dilakukan pembersihan data dari baris-baris duplikat menggunakan fungsi drop_duplicates(). Langkah ini penting untuk memastikan bahwa setiap entri dalam dataset bersifat unik dan tidak terjadi pengulangan yang dapat memengaruhi hasil analisis. Setelah penghapusan, jumlah data yang tersisa adalah 18.924 baris, yang berarti tidak ada baris duplikat dalam dataset. Pembersihan ini merupakan bagian penting dari tahap pra-pemrosesan untuk menjaga kualitas dan keandalan data sebelum analisis lebih lanjut dilakukan.

![image](https://github.com/user-attachments/assets/1c610e0a-8f3c-4959-af1f-02f9c79387e4)

- **Cleaning dan Transformasi Data**

  Pada tahap ini, dilakukan pembersihan dan transformasi data untuk memastikan konsistensi nilai serta pembuatan fitur baru yang relevan.

  Kolom Levy dibersihkan dengan mengganti tanda '-' menjadi NaN, menghapus koma, dan mengonversinya ke tipe float. Nilai yang hilang kemudian diisi dengan median agar tidak memengaruhi distribusi data.
Kolom Mileage juga dibersihkan dengan menghapus satuan ' km' dan koma sebelum dikonversi menjadi float.

  Sebelum mengekstrak nilai numerik dari kolom Engine volume, dibuat fitur biner baru bernama Is_Turbo untuk menandai keberadaan turbo pada mesin. Setelah itu, kata 'Turbo' dihapus dari kolom aslinya, lalu dikonversi ke float.

  Kolom Doors dikonversi dari format teks seperti '02-Mar' dan '04-May' menjadi angka menggunakan mapping sederhana. Nilai yang tidak dikenali diset default ke 4 pintu.
  Terakhir, kolom Leather interior diubah menjadi nilai biner (1 untuk 'Yes', 0 untuk 'No') agar bisa digunakan dalam analisis numerik.

![image](https://github.com/user-attachments/assets/b4a57799-c3c5-4bb8-bea0-fcac56eaf2ae)


- **Transformasi dan Skala Fitur**

  Pada tahap ini, dilakukan pembuatan beberapa fitur baru untuk memperkaya informasi dalam dataset dan mendukung proses analisis maupun pemodelan.

  Fitur **`Car_Age`** dihitung dari selisih antara tahun sekarang (2025) dengan tahun produksi mobil, untuk mengetahui usia kendaraan.
Dari situ, dihitung pula **`Mileage_per_Year`**, yaitu jarak tempuh rata-rata per tahun. Nilai tak hingga dan kosong diatasi dengan menggantinya menggunakan median.

  Fitur kategori baru **`Age_Group`** dibuat dengan membagi usia mobil menjadi tiga kelompok: *New*, *Medium*, dan *Old*, berdasarkan rentang tahun tertentu.
Selanjutnya, dibuat flag **`Is_Luxury`** yang menandai apakah mobil berasal dari merek mewah seperti BMW, MERCEDES-BENZ, dan lainnya.

  Untuk menyederhanakan analisis, kolom **`Manufacturer`** disiapkan untuk disederhanakan hanya menjadi 10 merek paling populer dan sisanya dikelompokkan sebagai `'Other'`.
  Terakhir, dibuat fitur **`Fuel_Efficiency_Proxy`** sebagai proksi efisiensi bahan bakar, yaitu rasio usia mobil terhadap volume mesin. Nilai ekstrem dan kosong ditangani dengan cara yang sama seperti sebelumnya.
    
  ![image](https://github.com/user-attachments/assets/99bdd5d0-b561-42a5-a8e0-658318f832fa)

- **Seleksi Fitur, Transformasi, dan Standarisasi** 

  Pada tahap ini, dilakukan normalisasi fitur numerik, transformasi target, serta encoding fitur kategori untuk mempersiapkan data sebelum pemodelan.

  Fitur-fitur numerik seperti `Levy`, `Mileage`, `Engine volume`, `Airbags`, dan lainnya distandarisasi menggunakan **StandardScaler** agar memiliki skala yang seragam (rata-rata 0 dan standar deviasi 1). Hal ini penting agar algoritma pembelajaran tidak bias terhadap fitur dengan skala besar.

  Kemudian, target variabel **`Price`** ditransformasikan menggunakan **logaritma natural** (log(1 + x)) untuk menstabilkan variansi dan mengurangi skewness, sehingga model dapat mempelajari pola harga dengan lebih baik.

  Selanjutnya, fitur-fitur kategori seperti `Manufacturer`, `Fuel type`, `Gear box type`, dan lainnya dikonversi menjadi format biner menggunakan **one-hot encoding**, dengan menghapus kolom pertama dari setiap kategori (`drop_first=True`) guna menghindari multikolinearitas. Langkah ini membuat data kategori siap digunakan dalam model berbasis numerik.

  ![image](https://github.com/user-attachments/assets/5952ba27-f078-42ad-b5e8-426d216915fb)

- **Pemisahan Dataset**

  Pada tahap ini, dilakukan pemisahan data menjadi fitur dan target, dilanjutkan dengan pembagian data untuk proses pelatihan dan pengujian model.

  Pertama, kolom **`Price`** dipisahkan sebagai **target (`y`)**, sedangkan semua kolom lain (kecuali `ID`) digunakan sebagai **fitur (`X`)**.
  Data kemudian dibagi menjadi **training set (90%)** dan **testing set (10%)** menggunakan fungsi `train_test_split` agar model bisa dilatih dan diuji secara adil. Parameter `random_state=42` digunakan untuk memastikan pembagian data bersifat **reproducible**.

  Selanjutnya, nama-nama kolom pada data training dan testing dibersihkan dari karakter ilegal seperti `[]`, `<>` yang dapat mengganggu proses pemodelan atau penyimpanan data.
  Akhirnya, ditampilkan ukuran akhir dataset untuk memastikan bahwa proses pemisahan berhasil: jumlah data poin pada training dan testing set, serta jumlah total fitur setelah proses encoding kategori.
 
  ![image](https://github.com/user-attachments/assets/43a8cf90-a3de-488d-b27c-894bc7542313)

---

## Modeling

### Inisialisasi Model:

![image](https://github.com/user-attachments/assets/cc5561e0-1d3f-4433-b15e-00ffec3a6de3)

#### **Model 1: Linear Regression**

##### **Cara Kerja**

Linear Regression merupakan algoritma statistik yang memodelkan hubungan antara variabel input (fitur) dengan target (harga) dalam bentuk garis lurus. Model ini mencari koefisien terbaik (β) dengan meminimalkan selisih kuadrat antara prediksi dan nilai sebenarnya menggunakan metode **Ordinary Least Squares (OLS)**.

Persamaan dasarnya:

$$
y = β_0 + β_1x_1 + β_2x_2 + \dots + β_nx_n
$$

##### **Parameter**

Linear Regression menggunakan **parameter default** dari `sklearn.linear_model.LinearRegression()`, yang berarti:

* `fit_intercept=True`: menyertakan bias/intersep.
* `normalize=False`: data tidak dinormalisasi secara otomatis.
* `n_jobs=None`: menggunakan 1 core CPU.

##### **Kelebihan dan Kekurangan (Opsional)**

**Kelebihan**:

* Cepat dan sederhana.
* Interpretasi model mudah (koefisien dapat menunjukkan pengaruh fitur).

**Kekurangan**:

* Tidak cocok untuk hubungan non-linear.
* Sangat sensitif terhadap multikolinearitas dan outlier.

---

#### **Model 2: Random Forest Regressor**

##### **Cara Kerja**

Random Forest adalah algoritma ensemble berbasis pohon keputusan. Model ini membangun banyak pohon (decision trees) secara acak dari subset data dan fitur, lalu menggabungkan prediksi masing-masing pohon (melalui rata-rata) untuk menghasilkan prediksi akhir. Teknik ini mengurangi overfitting dan meningkatkan akurasi.

##### **Parameter**

Parameter yang digunakan (dengan tuning ringan):

* `n_estimators=100`: jumlah pohon sebanyak 100.
* `random_state=42`: untuk memastikan hasil replikasi.
* Parameter lain menggunakan **default**, seperti:

  * `max_depth=None`: pohon tumbuh sampai semua daun murni.
  * `min_samples_split=2`: minimal 2 sampel untuk membagi node.

##### **Kelebihan dan Kekurangan (Opsional)**

**Kelebihan**:

* Handal terhadap data non-linear dan fitur kompleks.
* Tidak mudah overfitting dibandingkan decision tree tunggal.
* Menyediakan informasi feature importance.

**Kekurangan**:

* Model besar dan lambat saat prediksi pada data sangat besar.
* Kurang interpretatif dibanding regresi linier.

---

#### **Model 3: XGBoost Regressor**

##### **Cara Kerja**

XGBoost (Extreme Gradient Boosting) adalah metode boosting berbasis pohon yang membangun model secara bertahap. Setiap model baru mencoba memperbaiki kesalahan dari model sebelumnya dengan memperkecil fungsi loss. Teknik ini menggabungkan **gradient descent** dengan pohon keputusan sebagai weak learner.

##### **Parameter**

Model XGBoost menggunakan parameter awal berikut:

* `n_estimators=100`: jumlah model boosting.
* `random_state=42`: replikasi hasil.
* `verbosity=0`: nonaktifkan log output.
* Parameter lainnya menggunakan **default**, contohnya:

  * `learning_rate=0.3`
  * `max_depth=6`
  * `subsample=1`

> **Catatan**: Jika dilakukan hyperparameter tuning, parameter seperti `learning_rate`, `max_depth`, dan `subsample` bisa disesuaikan untuk meningkatkan performa.

##### **Kelebihan dan Kekurangan (Opsional)**

**Kelebihan**:

* Sangat akurat untuk banyak tugas regresi dan klasifikasi.
* Menangani missing values secara otomatis.
* Mendukung regularisasi (mencegah overfitting).

**Kekurangan**:

* Komputasi lebih berat dibanding model sederhana.
* Perlu tuning parameter untuk performa optimal.
### Latih dan Evaluasi Model:

![image](https://github.com/user-attachments/assets/7cbacae7-f715-43aa-8c76-dd2de0398dd9)

Pada tahap ini, dilakukan pelatihan dan evaluasi terhadap setiap model regresi menggunakan data yang telah dibagi sebelumnya menjadi data pelatihan dan data pengujian. Setiap model dalam dictionary `models` dilatih menggunakan data training (`X_train`, `y_train`). Setelah pelatihan, model digunakan untuk memprediksi harga pada data testing (`X_test`), menghasilkan prediksi `y_pred`.

Prediksi tersebut kemudian dievaluasi menggunakan tiga metrik utama untuk menilai performa prediktif model terhadap target **harga smartphone**, yang telah ditransformasikan menggunakan fungsi logaritma (`log(Price)`).

#### **Metrik Evaluasi**

Tiga metrik evaluasi utama yang digunakan adalah:

1. **MAE (Mean Absolute Error)**
   Mengukur rata-rata absolut selisih antara nilai prediksi dan nilai aktual. Semakin kecil nilainya, semakin baik model dalam memberikan prediksi yang akurat.

2. **RMSE (Root Mean Squared Error)**
   Mengukur akar dari rata-rata kuadrat selisih antara prediksi dan nilai aktual. Metrik ini lebih sensitif terhadap outlier dibanding MAE.

3. **R² (R-squared Score)**
   Mengukur seberapa baik model dapat menjelaskan variansi dari data target. Nilai R² mendekati 1 menunjukkan model sangat baik, sedangkan nilai mendekati 0 menandakan model kurang baik.

Metrik-metrik ini dipilih karena sesuai dan relevan dengan tipe proyek regresi harga yang bersifat numerik dan kontinu. Dengan metrik ini, model dapat dievaluasi tidak hanya dari akurasi rata-rata prediksi, tetapi juga kemampuannya menangkap pola dan variasi dalam data.

#### **Hasil Evaluasi dan Komparasi Model**

| Model                   | MAE      | RMSE     | R²       |
| ----------------------- | -------- | -------- | -------- |
| Linear Regression       | 0.946485 | 1.367330 | 0.285271 |
| Random Forest Regressor | 0.442551 | 0.893157 | 0.695034 |
| XGBoost Regressor       | 0.594116 | 0.988063 | 0.626781 |

#### **Interpretasi Hasil**

* **Linear Regression** memiliki performa terburuk, dengan R² hanya **0.285**, menandakan model ini hanya mampu menjelaskan sekitar 28,5% variasi dalam data. Ini menunjukkan bahwa pendekatan linier kurang memadai untuk menangkap hubungan kompleks antar fitur produk dengan harga.
* **Random Forest Regressor** memberikan performa terbaik dengan MAE paling rendah (**0.4425**) dan R² tertinggi (**0.695**), menunjukkan kemampuannya dalam memodelkan hubungan non-linier serta menangani kompleksitas data secara lebih efektif.
* **XGBoost Regressor** juga menunjukkan performa yang baik, namun sedikit di bawah Random Forest dalam semua metrik, menjadikannya alternatif yang cukup kompetitif.

**Model terbaik:** Berdasarkan ketiga metrik evaluasi, **Random Forest Regressor** dipilih sebagai model terbaik karena mampu menghasilkan prediksi paling akurat dan stabil dalam konteks proyek ini.

#### **Keterkaitan dengan Business Understanding**

Evaluasi model ini secara langsung berkaitan dengan pemenuhan kebutuhan bisnis, yaitu membangun sistem yang dapat memperkirakan harga smartphone berdasarkan fitur produk secara akurat dan efisien. Berikut penilaiannya terhadap business goals:

1. **Apakah sudah menjawab setiap problem statement?**
   
   ✅ Ya. Sistem telah berhasil membangun model yang mampu memprediksi harga smartphone berdasarkan informasi ulasan, rating, dan deskripsi produk, sebagaimana ditetapkan dalam problem statement.

2. **Apakah berhasil mencapai setiap goals yang diharapkan?**
   
   ✅ Ya. Dengan nilai R² sebesar **0.695**, sistem sudah cukup baik untuk digunakan dalam lingkungan produksi yang mendukung pengambilan keputusan, seperti sistem rekomendasi atau evaluasi harga pasar.

3. **Apakah setiap solusi statement yang direncanakan berdampak? Jelaskan!**
   
   ✅ Ya. Dampaknya meliputi:

   * **Untuk pengguna**: Sistem memungkinkan pengguna membandingkan harga pasar dengan harga yang diprediksi model. Hal ini membantu mereka menilai apakah produk yang ditampilkan terlalu mahal atau wajar.
   * **Untuk pelaku bisnis/platform**: Model ini dapat digunakan sebagai dasar dalam menentukan strategi harga otomatis berdasarkan fitur produk dan ulasan, sehingga dapat meningkatkan efisiensi pricing dan potensi penjualan.

---

### Tuning dan Optimasi

![image](https://github.com/user-attachments/assets/8d6a7ad0-37a4-4da2-8d35-36e1e767e2ce)

Pada tahap ini, dilakukan proses tuning hyperparameter menggunakan GridSearchCV untuk meningkatkan performa dua model terbaik sebelumnya: Random Forest dan XGBoost.

GridSearch untuk Random Forest

Dicoba beberapa kombinasi parameter seperti:

max_depth: kedalaman maksimum pohon (10 dan 20)

n_estimators: jumlah pohon dalam hutan (100)

min_samples_split dan min_samples_leaf: parameter kontrol pemisahan dan ukuran daun

Evaluasi dilakukan dengan cross-validation (cv=2) menggunakan skor R² sebagai metrik.

Hasil terbaik menunjukkan bahwa konfigurasi dengan max_depth=20 menghasilkan R² sebesar 0.598, dengan parameter lain tetap default.

GridSearch untuk XGBoost

Dicoba kombinasi parameter seperti:

max_depth, learning_rate, n_estimators, subsample, dan colsample_bytree

Evaluasi dilakukan dengan cross-validation (cv=2), menggunakan MAE negatif sebagai metrik (karena GridSearchCV secara default memaksimalkan skor).

Hasil terbaik diperoleh pada:

max_depth=6, learning_rate=0.05, n_estimators=200, dengan MAE negatif sebesar -0.673, artinya model cukup presisi terhadap target.

Proses ini bertujuan untuk menemukan kombinasi parameter terbaik dari tiap model yang menghasilkan performa optimal pada data.

### Feature Importance

![image](https://github.com/user-attachments/assets/aedd199a-8e8f-4244-af64-61906a59ddb6)

Pada tahap ini, dilakukan visualisasi feature importance dari model XGBoost terbaik yang diperoleh dari GridSearch.

Apa pentingnya melakukan ini?

Visualisasi ini bertujuan untuk mengetahui fitur-fitur mana saja yang paling berpengaruh dalam menentukan prediksi harga mobil (log(Price)). Dengan mengetahui fitur terpenting, kita bisa:

- Memahami insight dari data, misalnya fitur Gear box type_Tiptronic ternyata sangat berkontribusi terhadap harga mobil.

- Membantu pengambilan keputusan bisnis, seperti menyoroti fitur unggulan yang memengaruhi harga jual.

- Menyederhanakan model di tahap selanjutnya dengan hanya mempertahankan fitur-fitur yang paling penting (feature selection), sehingga mengurangi kompleksitas dan meningkatkan efisiensi.

- Menjelaskan hasil model ke stakeholder dengan lebih transparan (model interpretability).

## Kesimpulan:

- Model regresi yang dibangun, khususnya Random Forest Regressor, berhasil memprediksi harga mobil bekas dengan akurasi yang memadai.

- Faktor-faktor seperti tahun pembuatan, jarak tempuh, dan jenis bahan bakar sangat mempengaruhi harga kendaraan.

- Data preprocessing seperti pembersihan data, encoding, dan feature engineering sangat berpengaruh terhadap performa model.

## Rekomendasi:
- Dataset dapat diperluas dengan menambahkan fitur tambahan seperti lokasi, kondisi kendaraan, atau riwayat servis untuk meningkatkan akurasi prediksi.

- Sistem ini dapat diintegrasikan ke dalam platform jual beli mobil untuk memberikan estimasi harga otomatis kepada pengguna.

- Evaluasi berkala dan retraining model sangat disarankan agar model tetap relevan dengan tren pasar terbaru.

- Pertimbangkan penggunaan teknik ensemble atau boosting (seperti XGBoost atau LightGBM) untuk eksplorasi lebih lanjut.
