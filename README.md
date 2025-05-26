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
- **Menyalin dataset** agar tidak mengubah data asli saat dilakukan data preparation.

![image](https://github.com/user-attachments/assets/f5766bcd-b61a-48da-af4a-319dbbc03621)
- **Hapus Duplikasi Data** Pada tahap ini, dilakukan pembersihan data dari baris-baris duplikat menggunakan fungsi drop_duplicates(). Langkah ini penting untuk memastikan bahwa setiap entri dalam dataset bersifat unik dan tidak terjadi pengulangan yang dapat memengaruhi hasil analisis. Setelah penghapusan, jumlah data yang tersisa adalah 18.924 baris, yang berarti tidak ada baris duplikat dalam dataset. Pembersihan ini merupakan bagian penting dari tahap pra-pemrosesan untuk menjaga kualitas dan keandalan data sebelum analisis lebih lanjut dilakukan.

![image](https://github.com/user-attachments/assets/1c610e0a-8f3c-4959-af1f-02f9c79387e4)

- **Cleaning dan Transformasi Data** Pada tahap ini, dilakukan serangkaian pembersihan dan transformasi terhadap beberapa fitur dalam dataset untuk memastikan konsistensi dan kualitas data yang akan digunakan dalam proses analisis.

  1. Pertama, kolom Levy dibersihkan dengan mengganti nilai '-' menjadi NaN, lalu menghapus tanda koma dan mengubahnya menjadi tipe data numerik (float). Ini penting karena Levy semestinya bernilai angka, namun dalam bentuk awalnya masih berupa string. Setelah itu, nilai-nilai yang hilang pada kolom Levy diisi menggunakan nilai median dari kolom tersebut. Pengisian dengan median dipilih karena lebih tahan terhadap outlier dibandingkan dengan rata-rata.

  2. Selanjutnya, kolom Mileage juga dibersihkan dari teks ' km' dan tanda koma agar dapat dikonversi menjadi angka (float). Proses ini diperlukan agar data jarak tempuh kendaraan bisa dianalisis secara numerik, misalnya untuk korelasi dengan harga atau usia kendaraan.

  3. Kemudian dibuat fitur baru bernama Is_Turbo. Fitur ini menunjukkan apakah kendaraan memiliki turbo atau tidak. Nilainya dihasilkan dengan memeriksa apakah string 'Turbo' terdapat dalam kolom Engine volume, lalu dikonversi menjadi nilai biner (1 untuk ya, 0 untuk tidak). Pembuatan fitur ini dilakukan sebelum membersihkan kolom Engine volume agar informasi tentang turbo tidak hilang.

  4. Setelah fitur turbo berhasil diekstrak, kolom Engine volume dibersihkan dengan menghapus teks ' Turbo', kemudian dikonversi ke tipe data numerik (float). Hal ini memungkinkan kita untuk menganalisis kapasitas mesin secara langsung sebagai variabel numerik.

  5. Kolom Doors yang memiliki format seperti '04-May', '02-Mar', atau '>5' diubah ke bentuk numerik menggunakan pemetaan manual (map). Nilai-nilai yang tidak dikenali dipetakan ke angka 4 sebagai default. Ini penting untuk menyederhanakan representasi jumlah pintu dalam analisis dan modeling.

  5. Terakhir, kolom Leather interior diubah menjadi representasi biner menggunakan mapping 'Yes' menjadi 1 dan 'No' menjadi 0. Hal ini bertujuan untuk menyederhanakan fitur ini agar bisa digunakan dalam model machine learning yang tidak menerima input dalam bentuk string.

  Seluruh langkah ini merupakan bagian dari proses pra-pemrosesan data yang bertujuan untuk mengubah data mentah menjadi bentuk yang lebih bersih, konsisten, dan siap digunakan untuk analisis maupun pemodelan machine learning.

![image](https://github.com/user-attachments/assets/b4a57799-c3c5-4bb8-bea0-fcac56eaf2ae)


- **Transformasi dan Skala Fitur :**
  Pada tahap ini, dilakukan pembuatan beberapa fitur baru untuk memperkaya informasi dalam dataset dan mendukung proses analisis maupun pemodelan.

  Pertama, dibuat fitur Car_Age yang merepresentasikan usia kendaraan dengan cara mengurangkan tahun sekarang (2025) dengan tahun produksi kendaraan. Fitur ini membantu memahami umur mobil yang dapat memengaruhi harga dan performa.

  Kemudian dihitung Mileage_per_Year, yaitu rata-rata jarak tempuh per tahun. Nilai ini diperoleh dengan membagi total jarak tempuh (Mileage) dengan usia kendaraan (Car_Age). Nilai tak hingga atau tak valid diganti dengan NaN dan diisi dengan median untuk menjaga stabilitas data.

  Selanjutnya, kendaraan dikategorikan ke dalam grup usia menggunakan fitur Age_Group dengan tiga label: 'New', 'Medium', dan 'Old', berdasarkan interval umur tertentu. Ini berguna untuk klasifikasi kondisi mobil secara sederhana.

  Dibuat pula fitur Is_Luxury, yaitu indikator biner apakah kendaraan termasuk merek mewah seperti BMW, Mercedes-Benz, atau Tesla. Hal ini penting untuk analisis segmen pasar.

  Kemudian, untuk menyederhanakan variasi merek, hanya 10 merek terpopuler yang dipertahankan pada fitur Manufacturer, sementara merek lainnya dikategorikan sebagai 'Other'. Ini membantu mengurangi kompleksitas model.

  Terakhir, ditambahkan fitur Fuel_Efficiency_Proxy, yaitu proksi efisiensi bahan bakar yang dihitung dari pembagian usia mobil dengan volume mesin. Meskipun bukan representasi langsung, fitur ini memberi indikasi kasar tentang efisiensi relatif kendaraan. Nilai tak valid juga ditangani dengan mengganti ke median.

  Langkah-langkah ini bertujuan untuk memperkaya dan menyederhanakan data agar lebih informatif dan siap untuk dianalisis lebih lanjut.
  
  ![image](https://github.com/user-attachments/assets/99bdd5d0-b561-42a5-a8e0-658318f832fa)

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
