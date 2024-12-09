# Laporan Proyek Machine Learning - Nur Oktavin Idris

### Prediksi Harga Mobil
Ini adalah proyek pertama predictive analytics untuk memenuhi submission dicoding. Proyek ini membangun model machine learning untuk memprediksi harga mobil

## Domain Proyek

Mobil adalah salah satu aset yang memiliki nilai jual bervariasi bergantung pada berbagai faktor seperti merek, model, spesifikasi teknis, dan fitur lainnya. Prediksi harga jual mobil dapat membantu produsen, penjual, dan konsumen untuk memahami nilai pasar kendaraan secara lebih akurat. Proyek ini bertujuan untuk membangun model machine learning yang dapat memprediksi harga jual mobil dengan menggunakan dataset yang mencakup berbagai fitur kendaraan.
  
  Format Referensi: [Judul Referensi](https://scholar.google.com/) 

## Business Understanding

Pada bagian ini, kamu perlu menjelaskan proses klarifikasi masalah.

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
Dataset yang digunakan terdiri dari 11,914 sampel dan 16 fitur, dengan target variabel adalah 'MSRP'. MSRP adalah harga yang direkomendasikan oleh produsen kendaraan kepada dealer untuk dijadikan acuan dalam menjual produk ke konsumen. Sumber dataset ini dari  [Kaggle](https://www.kaggle.com/datasets/CooperUnion/cardataset/data). Dataset berisi informasi tentang kendaraan dengan fitur-fitur berikut:
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

 <div><img src="https://github.com/oktavin28/proyek-predictive-analytics/blob/07897aec6c169e46eb12f099ed7d7dc6dc96947b/images/korelasinumerik.png"></div>

Matriks korelasi ini memberikan gambaran hubungan antar fitur numerik dalam dataset, yang sangat membantu untuk analisis data lebih lanjut. Matriks ini membantu mengidentifikasi fitur mana yang relevan atau saling berkaitan untuk digunakan dalam analisis atau model prediktif ke depannya.Pada gambar tersebut terlihat antara Engine HP dan Engine Cylinders (0.77), serta antara City MPG dan Highway MPG (0.89) mencerminkan korelasi tinggi. Hal ini juga terlihat antara City MPG dan Highway MPG (0.89), yang menunjukkan bahwa daya mesin lebih besar cenderung meningkatkan harga mobil. 
Sebaliknya, fitur seperti Highway MPG dan Engine Cylinders menunjukkan korelasi negatif (-0.6), menandakan bahwa mobil dengan lebih banyak silinder biasanya kurang efisien bahan bakarnya. 

#### Korelasi Fitur Kategorik
 <div><img src="https://github.com/oktavin28/proyek-predictive-analytics/blob/c93189f10ac253c0a2ccdd3ec176e7df2687fba6/images/korelasi%20make.png"></div>

Korelasi fitur ini menampilkan rata-rata harga MSRP berdasarkan merek mobil (Make), yang memberikan informasi tentang bagaimana merek memengaruhi harga jual. Dari grafik ini, terlihat bahwa merek seperti Bugatti dan Maybach memiliki rata-rata MSRP yang jauh lebih tinggi dibandingkan merek lain, mencerminkan bahwa kedua merek itu sebagai produsen mobil mewah kelas atas. Sebaliknya, merek seperti Hyundai, Honda, dan Chevrolet cenderung memiliki MSRP rata-rata yang lebih rendah, mengindikasikan fokus pada pasar mobil dengan harga yang lebih terjangkau. Variasi harga yang signifikan pada beberapa merek, seperti McLaren dan Rolls-Royce, mencakup berbagai segmen harga. Grafik ini diperlukan untuk memahami perbedaan pasar yang dilayani oleh setiap merek dan bagaimana faktor merek berkontribusi terhadap penetapan harga dalam industri otomotif.

 <div><img src="https://github.com/oktavin28/proyek-predictive-analytics/blob/a5edae19655215e83ec3a145c09d05e5bd4e3a87/images/korelasitransmisi.png"></div>

Grafik ini menunjukkan rata-rata harga MSRP berdasarkan jenis transmisi mobil, yang memberikan informasi tentang bagaimana tipe transmisi mempengaruhi nilai pasar kendaraan. Dari visualisasi ini, terlihat bahwa mobil dengan transmisi Automated Manual memiliki rata-rata MSRP tertinggi dibandingkan tipe lainnya, menunjukkan bahwa teknologi transmisi ini sering ditemukan pada mobil premium atau performa tinggi. Sebaliknya, transmisi Manual dan Automatic memiliki MSRP yang relatif lebih rendah, yang umumnya sesuai dengan pasar mobil konvensional. Tipe transmisi Direct Drive, yang sering ditemukan pada mobil listrik, memiliki harga yang lebih tinggi dibandingkan transmisi manual atau otomatis, tetapi masih di bawah Automated Manual. Sementara itu, kategori Unknown memiliki MSRP paling rendah, kemungkinan berasal dari data yang kurang lengkap atau kendaraan dengan spesifikasi yang kurang jelas. Grafik ini membantu memahami peran tipe transmisi dalam segmen pasar otomotif dan bagaimana hal itu terkait dengan harga jual kendaraan.

<div><img src="https://github.com/oktavin28/proyek-predictive-analytics/blob/a5edae19655215e83ec3a145c09d05e5bd4e3a87/images/korelasidriven.png"></div>

Grafik ini menunjukkan rata-rata harga MSRP berdasarkan jenis transmisi mobil, memberikan wawasan tentang bagaimana tipe transmisi memengaruhi nilai pasar kendaraan. Dari visualisasi ini, terlihat bahwa mobil dengan transmisi Automated Manual memiliki rata-rata MSRP tertinggi dibandingkan tipe lainnya, mencerminkan bahwa teknologi transmisi ini sering ditemukan pada mobil premium atau performa tinggi. Sebaliknya, transmisi Manual dan Automatic memiliki MSRP yang relatif lebih rendah, yang umumnya sesuai dengan pasar mobil konvensional. Tipe transmisi Direct Drive, yang sering ditemukan pada mobil listrik, memiliki harga yang lebih tinggi dibandingkan transmisi manual atau otomatis, tetapi masih di bawah Automated Manual. Sementara itu, kategori Unknown memiliki MSRP paling rendah, kemungkinan berasal dari data yang kurang lengkap atau kendaraan dengan spesifikasi yang kurang jelas. Grafik ini membantu memahami peran tipe transmisi dalam segmen pasar otomotif dan bagaimana hal itu terkait dengan harga jual kendaraan.

<div><img src="https://github.com/oktavin28/proyek-predictive-analytics/blob/a5edae19655215e83ec3a145c09d05e5bd4e3a87/images/korelasivehicle.png"></div>

Grafik ini memperlihatkan rata-rata MSRP berdasarkan ukuran kendaraan (Vehicle Size), yang menunjukkan tentang bagaimana dimensi kendaraan mempengaruhi harga pasar. Terlihat bahwa kendaraan berukuran Large memiliki rata-rata MSRP tertinggi, mencerminkan bahwa mobil besar, seperti SUV mewah atau kendaraan kelas atas lainnya, biasanya dihargai lebih mahal karena fitur, kapasitas, dan teknologi yang lebih canggih. Sementara itu, kendaraan berukuran Midsize memiliki harga rata-rata tingkat menengah, yang sering kali mewakili kendaraan keluarga dengan keseimbangan antara performa, kapasitas, dan efisiensi. Kendaraan Compact memiliki rata-rata MSRP terendah, sesuai dengan karakteristiknya sebagai mobil yang lebih kecil dan efisien, yang biasanya dirancang untuk segmen pasar yang lebih ekonomis. Grafik ini memberikan informasi penting untuk memahami segmentasi pasar otomotif berdasarkan ukuran kendaraan.

<div><img src="https://github.com/oktavin28/proyek-predictive-analytics/blob/a5edae19655215e83ec3a145c09d05e5bd4e3a87/images/korelasivehiclestyle.png"></div>

Grafik ini menunjukkan rata-rata MSRP berdasarkan jenis gaya kendaraan (Vehicle Style), memberikan informasi tentang bagaimana desain dan fungsi kendaraan mempengaruhi harga pasar. Terlihat bahwa kendaraan dengan gaya Convertible dan Coupe memiliki rata-rata MSRP tertinggi, karena gaya ini sering ditemukan pada mobil premium atau sport dengan desain yang eksklusif. 4dr SUV juga memiliki MSRP rata-rata yang cukup tinggi, mencerminkan popularitas SUV besar dengan fitur yang lebih lengkap dan mewah. Di sisi lain, gaya kendaraan seperti 4dr Hatchback dan Cargo Minivan memiliki MSRP rata-rata lebih rendah, yang sesuai dengan segmentasinya sebagai kendaraan fungsional atau ekonomis. Variasi harga yang signifikan ini memberikan gambaran bagaimana fungsi, desain, dan target pasar memengaruhi nilai kendaraan, sekaligus menjadi panduan penting bagi konsumen atau produsen dalam memahami tren harga di berbagai segmen kendaraan.


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
Algoritma random forest adalah teknik dalam machine learning dengan metode ensemble. Proyek ini menggunakan sklearn.ensemble.RandomForestRegressor dengan memasukkan X_train dan y_train dalam membangun model. Parameter yang digunakan pada proyek ini adalah: 

    * random_state = Mengontrol seed acak yang diberikan pada setiap base_estimator pada setiap iterasi boosting.
    * n_estimators = Jumlah maksimum estimator di mana boosting dihentikan.
    * max_depth = Kedalaman maksimum setiap tree.

- Gradient Boosting
Gradient Boosting adalah algoritma machine learning berbasis ensemble. Untuk proyek ini menggunakan sklearn.ensemble.GradientBoostingRegressor dengan parameter sebagai berikut:
  * random_state = Mengontrol seed acak yang diberikan pada setiap base_estimator pada setiap iterasi boosting.
  * n_estimators = Jumlah maksimum estimator di mana boosting dihentikan.
  * learning_rate =  parameter yang mengatur seberapa besar langkah yang diambil untuk memperbarui bobot model selama pelatihan     

- Linear Regression
Linear Regression adalah algoritma machine learning yang sederhana namun sangat efektif untuk memodelkan hubungan antara variabel independen (fitur) dan variabel dependen (target). Proyek ini menggunakan sklearn.linear_model.LinearRegression dengan parameter sebagai berikut:
  * fit_intercept = Parameter ini menentukan apakah model harus menghitung intercept (titik potong pada sumbu y) dalam persamaan linear.
  * n_jobs = Parameter ini menentukan jumlah thread (proses paralel) yang digunakan oleh model saat melakukan komputasi.


## Evaluation
Mean Square Error (MSE) adalah salah satu metrik evaluasi yang umum digunakan dalam statistik dan machine learning. Fungsi dari MSE adalah untuk mengukur seberapa baik suatu model dalam memetakan nilai prediksi ke nilai sebenarnya dengan menggunakan kuadrat kesalahan sebagai dasar perhitungan. Berikut rumus yang digunakan untuk menghitung MSE:

<div><img src="https://github.com/oktavin28/proyek-predictive-analytics/blob/c0fea9fda0a3b64da9ec4d97d7465c1501845ac7/images/MSE.jpg"></div>

Keterangan:
- n adalah jumlah total observasi atau sampel.
- yi adalah nilai aktual atau nilai sebenarnya dari observasi ke-i.
- Ŷi adalah nilai prediksi model untuk observasi ke-i.


<div><img src="https://github.com/oktavin28/proyek-predictive-analytics/blob/a5edae19655215e83ec3a145c09d05e5bd4e3a87/images/hasil%20evaluasi.png"></div>

<div><img src="https://github.com/oktavin28/proyek-predictive-analytics/blob/a5edae19655215e83ec3a145c09d05e5bd4e3a87/images/perbandingan.png"></div>

Tabel dan grafik ini sama-sama menunjukkan evaluasi kinerja tiga model machine learning Random Forest, Gradient Boosting, dan Linear Regression berdasarkan metrik MSE (Mean Squared Error) dan R² Score. Terlihat pada tabel bahwa Gradient Boosting menunjukkan performa terbaik dengan R² Score sebesar 0.963868 dan MSE terendah, menandakan bahwa model ini memiliki kemampuan prediksi yang sangat baik dan error yang kecil. Grafik R² Score juga mendukung hasil ini, menunjukkan bahwa Gradient Boosting memiliki nilai R² mendekati 1, yang berarti model ini mampu menunjukkan variabilitas data dengan sangat baik. Random Forest juga memiliki performa yang cukup baik dengan R² Score sebesar 0.899657, meskipun MSE-nya lebih tinggi dibanding Gradient Boosting. Sebaliknya, Linear Regression memiliki performa yang jauh lebih rendah, dengan R² Score hanya 0.417905 dan MSE yang tinggi, menunjukkan bahwa model ini kurang cocok untuk data yang lebih kompleks. Jadi Gradient Boosting sebagai pilihan terbaik untuk memaksimalkan akurasi prediksi dalam kasus ini.

<div><img src="https://github.com/oktavin28/proyek-predictive-analytics/blob/a5edae19655215e83ec3a145c09d05e5bd4e3a87/images/predict.png"></div>

Tabel ini menampilkan perbandingan antara nilai aktual (Actual) dan nilai prediksi (Predicted) dari model Gradient Boosting untuk beberapa sampel data. Secara umum, hasil prediksi terlihat cukup mendekati nilai aktual, dengan perbedaan yang relatif kecil, seperti pada baris pertama di mana nilai aktual adalah 34,160, sementara nilai prediksi adalah 33,757.54. Hal ini menunjukkan bahwa model memiliki kemampuan prediksi yang cukup baik dan dapat menangkap pola dalam data dengan akurasi yang layak. 


