# Laporan Proyek Machine Learning - Nur Oktavin Idris

## Domain Proyek

Mobil adalah salah satu aset yang memiliki nilai jual bervariasi bergantung pada berbagai faktor seperti merek, model, spesifikasi teknis, dan fitur lainnya. Prediksi harga jual mobil dapat membantu produsen, penjual, dan konsumen untuk memahami nilai pasar kendaraan secara lebih akurat. Proyek ini bertujuan untuk membangun model machine learning yang dapat memprediksi harga jual mobil dengan menggunakan dataset yang mencakup berbagai fitur kendaraan.
  
  Format Referensi: [Judul Referensi](https://scholar.google.com/) 

## Business Understanding

Pada bagian ini, kamu perlu menjelaskan proses klarifikasi masalah.

Bagian laporan ini mencakup:

### Problem Statements

1. Fitur apa saja yang paling berpengaruh terhadap harga jual mobil?
2. Bagaimana cara memproses data agar dapat dilatih dengan baik oleh model machine learning?
3. Seberapa akurat prediksi harga jual mobil berdasarkan karakteristik tertentu?


### Goals

1. Menentukan fitur yang paling berpengaruh pada harga jual mobil.
2. Melakukan persiapan data untuk memastikan data siap digunakan untuk pemodelan.
3. Membangun model machine learning yang mampu memberikan prediksi harga jual mobil yang akurat.


### Solution statements
    
1. Melakukan analisis eksplorasi data untuk memahami hubungan antar fitur dan target.
2. Mengolah data menggunakan teknik seperti normalisasi dan one-hot encoding.
3. Melatih beberapa model regresi seperti Random Forest, AdaBoost, dan KNN untuk memprediksi harga.
4. Mengevaluasi kinerja model menggunakan metrik evaluasi seperti Mean Absolute Error (MAE).


## Data Understanding
Dataset yang digunakan terdiri dari 11,914 sampel dan 16 fitur, dengan target variabel adalah 'MSRP' (harga jual mobil). Sumber dataset ini dari  [Kaggle](https://www.kaggle.com/datasets/CooperUnion/cardataset/data). Dataset berisi informasi tentang kendaraan dengan fitur-fitur berikut:
- Make: Merek mobil.
- Model: Model mobil.
- Year: Tahun produksi.
- Engine HP: Tenaga mesin dalam horsepower.
- Engine Cylinders: Jumlah silinder mesin.
- Engine HP: Tenaga mesin (Horsepower).
- Engine Cylinders: Jumlah silinder mesin.
- Transmission Type: Jenis transmisi.
- Driven_Wheels: Sistem penggerak roda.
- Vehicle Size: Ukuran kendaraan.
- Number of Doors : Jumlah pintu
- Market Category
- Vehicle Style: Gaya kendaraan.
- highway MPG: Efisiensi bahan bakar di jalan raya.
- city mpg: Efisiensi bahan bakar di dalam kota.
- Popularity: Popularitas merek mobil.
- MSRP: Harga kendaraan (Target).

Dari ke 12 fitur terdapat bebrapa fitur yang memiliki banyak nilai kosong dan duplikat karena relevansinya rendah dan tidak diperlukan dalam membagun model prediksi harga jual mobil

#### Korelasi Fitur Numerik

#### Korelasi Fitur Kategorik

## Data Preparation
Data telah disiapkan untuk pelatihan model dengan langkah-langkah berikut:
- One-hot encoding 
One hot encoding adalah teknik mengubah data kategorik menjadi data numerik dimana setiap kategori menjadi kolom baru dengan nilai 0 atau 1. Fitur yang akan diubah menjadi numerik pada proyek ini adalah make, model, engine fuel type, transmission type, driven_wheels, vehicle size, dan vehicle type

- Train Test Split
Train test split merupakan proses membagi data menjadi data latih dan data uji. Data latih akan digunakan untuk membangun model, sedangkan data uji akan digunakan untuk menguji performa model. Pada proyek ini dataset sebesar 11199 dibagi menjadi 8959 untuk data latih dan 2240 untuk data uji.

- Normalization
Algoritma machine learning akan memiliki performa lebih baik dan bekerja lebih cepat jika dimodelkan dengan data seragam yang memiliki skala relatif sama. Salah satu teknik normalisasi yang digunakan pada proyek ini adalah Standarisasi dengan StandardScaler.


## Modeling
Algoritma pada proyek ini melakukan pemodelan dengan 3 algoritma, yaitu Random Forest, Gradient Boosting dan Linear Regression  
- Random Forest 
Algoritma random forest adalah teknik dalam machine learning dengan metode ensemble. Teknik ini beroperasi dengan membangun banyak decision tree pada waktu pelatihan. Proyek ini menggunakan sklearn.ensemble.RandomForestRegressor dengan memasukkan X_train dan y_train dalam membangun model. Parameter yang digunakan pada proyek ini adalah random_state untuk mengontrol seed acak yang diberikan pada setiap base_estimator pada setiap iterasi boosting.


## Evaluation
Mean Absolute Error (MAE) berfungsi untuk mengukur seberapa dekat hasil prediksi suatu model dengan nilai sebenarnya. MAE mengukur rata-rata dari selisih absolut antara prediksi model dan nilai sebenarnya dari data yang diamati. Alih-alih membiarkan nilai apa adanya, pada MAE setiap nilai yang diambil sifatnya absolut sehingga nilai negatif hilang. Berikut adalah penjelasan detail tentang Mean Absolute Error.
MAE dihitung dengan menjumlahkan selisih absolut antara setiap prediksi individu dan nilai sebenarnya, kemudian dibagi dengan jumlah total observasi yang dapat direpresentasikan sebagai berikut.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

