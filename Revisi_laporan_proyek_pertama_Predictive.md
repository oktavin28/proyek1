# Laporan Proyek Machine Learning - Nur Oktavin Idris

### Prediksi Harga Mobil
Ini adalah proyek pertama predictive analytics untuk memenuhi submission dicoding. Proyek ini membangun model machine learning untuk memprediksi harga mobil

## Domain Proyek

Mobil adalah salah satu aset yang memiliki nilai jual bervariasi bergantung pada berbagai faktor seperti merek, model, spesifikasi teknis, dan fitur lainnya. Prediksi harga jual mobil dapat membantu produsen, penjual, dan konsumen untuk memahami nilai pasar kendaraan secara lebih akurat. Proyek ini bertujuan untuk membangun model machine learning yang dapat memprediksi harga jual mobil dengan menggunakan dataset yang mencakup berbagai fitur kendaraan.
  
Referensi: [Prediksi Harga Mobil Menggunakan Algoritma Regressi dengan Hyper-Parameter Tuning](https://jurnal.unprimdn.ac.id/index.php/JUSIKOM/article/view/2479/1459) 

## Business Understanding

Proyek ini dirancang untuk membantu dalam memahami nilai pasar kendaraan berdasarkan berbagai faktor yang memengaruhi harga jual. Perusahaan dengan karakteristik bisnis berikut dapat memanfaatkan hasil dari proyek ini:

- Dealer kendaraan yang ingin menentukan harga jual optimal berdasarkan spesifikasi teknis, gaya, dan ukuran kendaraan.
- Perusahaan jasa penilaian kendaraan yang menawarkan layanan konsultasi kepada pelanggan terkait harga kendaraan bekas atau baru.
- Produsen otomotif yang ingin menganalisis bagaimana berbagai atribut produk memengaruhi harga pasar kendaraan mereka.

Pendekatan ini menggunakan algoritma machine learning untuk memprediksi harga berdasarkan dataset yang mencakup berbagai fitur kendaraan. Dengan model prediktif yang akurat, perusahaan dapat meningkatkan efisiensi dalam menentukan harga, menargetkan pasar, dan mengoptimalkan penjualan. 

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
Dataset yang digunakan terdiri dari 11,914 sampel dan 16 fitur, dengan target variabel adalah 'MSRP'. MSRP adalah harga yang direkomendasikan oleh produsen kendaraan kepada dealer untuk dijadikan acuan dalam menjual produk ke konsumen. Sumber dataset ini dari  [Kaggle: Car Features and MSRP](https://www.kaggle.com/datasets/CooperUnion/cardataset/data). Dataset berisi informasi tentang kendaraan dengan fitur-fitur berikut:
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

Dari ke 12 fitur terdapat beberapa fitur yang memiliki banyak nilai kosong dan duplikat karena relevansinya rendah dan tidak diperlukan dalam membagun model prediksi harga jual mobil, seperti Model, Market Category, dan Number of Doors

#### Korelasi Fitur Numerik

https://github.com/oktavin28/proyek-predictive-analytics/blob/9ccb439ce28698c2a9dc0cdb499774fb039d87e9/capture-20231123-142806.png

Matriks korelasi ini memberikan gambaran hubungan antar fitur numerik dalam dataset, yang sangat membantu untuk analisis data lebih lanjut. Matriks ini membantu mengidentifikasi fitur mana yang relevan atau saling berkaitan untuk digunakan dalam analisis atau model prediktif ke depannya.Pada gambar tersebut terlihat antara Engine HP dan Engine Cylinders (0.77), serta antara City MPG dan Highway MPG (0.89) mencerminkan korelasi tinggi. Hal ini juga terlihat antara City MPG dan Highway MPG (0.89), yang menunjukkan bahwa daya mesin lebih besar cenderung meningkatkan harga mobil. 
Sebaliknya, fitur seperti Highway MPG dan Engine Cylinders menunjukkan korelasi negatif (-0.6), menandakan bahwa mobil dengan lebih banyak silinder biasanya kurang efisien bahan bakarnya. 

#### Korelasi Fitur Kategorik

![korelasi make](https://github.com/user-attachments/assets/4afcaa9d-42bb-4ec3-becb-478a6b706e72)

Korelasi fitur ini menampilkan rata-rata harga MSRP berdasarkan merek mobil (Make), yang memberikan informasi tentang bagaimana merek memengaruhi harga jual. Dari grafik ini, terlihat bahwa merek seperti Bugatti dan Maybach memiliki rata-rata MSRP yang jauh lebih tinggi dibandingkan merek lain, mencerminkan bahwa kedua merek itu sebagai produsen mobil mewah kelas atas. Sebaliknya, merek seperti Hyundai, Honda, dan Chevrolet cenderung memiliki MSRP rata-rata yang lebih rendah, mengindikasikan fokus pada pasar mobil dengan harga yang lebih terjangkau. Variasi harga yang signifikan pada beberapa merek, seperti McLaren dan Rolls-Royce, mencakup berbagai segmen harga. Grafik ini diperlukan untuk memahami perbedaan pasar yang dilayani oleh setiap merek dan bagaimana faktor merek berkontribusi terhadap penetapan harga dalam industri otomotif.


![korelasitransmisi](https://github.com/user-attachments/assets/80f433fc-0d79-482a-a4b8-586215482412)

Grafik ini menunjukkan rata-rata harga MSRP berdasarkan jenis transmisi mobil, yang memberikan informasi tentang bagaimana tipe transmisi mempengaruhi nilai pasar kendaraan. Dari visualisasi ini, terlihat bahwa mobil dengan transmisi Automated Manual memiliki rata-rata MSRP tertinggi dibandingkan tipe lainnya, menunjukkan bahwa teknologi transmisi ini sering ditemukan pada mobil premium atau performa tinggi. Sebaliknya, transmisi Manual dan Automatic memiliki MSRP yang relatif lebih rendah, yang umumnya sesuai dengan pasar mobil konvensional. Tipe transmisi Direct Drive, yang sering ditemukan pada mobil listrik, memiliki harga yang lebih tinggi dibandingkan transmisi manual atau otomatis, tetapi masih di bawah Automated Manual. Sementara itu, kategori Unknown memiliki MSRP paling rendah, kemungkinan berasal dari data yang kurang lengkap atau kendaraan dengan spesifikasi yang kurang jelas. Grafik ini membantu memahami peran tipe transmisi dalam segmen pasar otomotif dan bagaimana hal itu terkait dengan harga jual kendaraan.


![korelasidriven](https://github.com/user-attachments/assets/1551fa1c-0de5-4330-933a-bae24cb77924)

Grafik ini menunjukkan rata-rata harga MSRP berdasarkan jenis transmisi mobil, memberikan wawasan tentang bagaimana tipe transmisi memengaruhi nilai pasar kendaraan. Dari visualisasi ini, terlihat bahwa mobil dengan transmisi Automated Manual memiliki rata-rata MSRP tertinggi dibandingkan tipe lainnya, mencerminkan bahwa teknologi transmisi ini sering ditemukan pada mobil premium atau performa tinggi. Sebaliknya, transmisi Manual dan Automatic memiliki MSRP yang relatif lebih rendah, yang umumnya sesuai dengan pasar mobil konvensional. Tipe transmisi Direct Drive, yang sering ditemukan pada mobil listrik, memiliki harga yang lebih tinggi dibandingkan transmisi manual atau otomatis, tetapi masih di bawah Automated Manual. Sementara itu, kategori Unknown memiliki MSRP paling rendah, kemungkinan berasal dari data yang kurang lengkap atau kendaraan dengan spesifikasi yang kurang jelas. Grafik ini membantu memahami peran tipe transmisi dalam segmen pasar otomotif dan bagaimana hal itu terkait dengan harga jual kendaraan.


![korelasivehicle](https://github.com/user-attachments/assets/c7b52223-e850-487e-b384-cf18b46ed686)

Grafik ini memperlihatkan rata-rata MSRP berdasarkan ukuran kendaraan (Vehicle Size), yang menunjukkan tentang bagaimana dimensi kendaraan mempengaruhi harga pasar. Terlihat bahwa kendaraan berukuran Large memiliki rata-rata MSRP tertinggi, mencerminkan bahwa mobil besar, seperti SUV mewah atau kendaraan kelas atas lainnya, biasanya dihargai lebih mahal karena fitur, kapasitas, dan teknologi yang lebih canggih. Sementara itu, kendaraan berukuran Midsize memiliki harga rata-rata tingkat menengah, yang sering kali mewakili kendaraan keluarga dengan keseimbangan antara performa, kapasitas, dan efisiensi. Kendaraan Compact memiliki rata-rata MSRP terendah, sesuai dengan karakteristiknya sebagai mobil yang lebih kecil dan efisien, yang biasanya dirancang untuk segmen pasar yang lebih ekonomis. Grafik ini memberikan informasi penting untuk memahami segmentasi pasar otomotif berdasarkan ukuran kendaraan.


![korelasivehiclestyle](https://github.com/user-attachments/assets/2146460c-2d2a-448d-9a33-6ad42ee3f7fe)

Grafik ini menunjukkan rata-rata MSRP berdasarkan jenis gaya kendaraan (Vehicle Style), memberikan informasi tentang bagaimana desain dan fungsi kendaraan mempengaruhi harga pasar. Terlihat bahwa kendaraan dengan gaya Convertible dan Coupe memiliki rata-rata MSRP tertinggi, karena gaya ini sering ditemukan pada mobil premium atau sport dengan desain yang eksklusif. 4dr SUV juga memiliki MSRP rata-rata yang cukup tinggi, mencerminkan popularitas SUV besar dengan fitur yang lebih lengkap dan mewah. Di sisi lain, gaya kendaraan seperti 4dr Hatchback dan Cargo Minivan memiliki MSRP rata-rata lebih rendah, yang sesuai dengan segmentasinya sebagai kendaraan fungsional atau ekonomis. Variasi harga yang signifikan ini memberikan gambaran bagaimana fungsi, desain, dan target pasar memengaruhi nilai kendaraan, sekaligus menjadi panduan penting bagi konsumen atau produsen dalam memahami tren harga di berbagai segmen kendaraan.


## Data Preparation
Data telah disiapkan untuk pelatihan model dengan langkah-langkah berikut:
- Membaca Dataset

Langkah pertama dalam data preparation adalah memastikan memiliki data yang bisa diolah. Pada bagian ini, kita membaca dataset dari file `data.csv` menggunakan fungsi `pd.read_csv()` dari library pandas. Fungsi ini digunakan untuk memuat data dalam format CSV ke dalam struktur DataFrame, yang memudahkan manipulasi dan analisis data. Kemudian 5 baris awalnya ditampilkan dengan df.head(). 

- Memeriksa Informasi Dataset
  
Pada bagian ini, menggunakan fungsi `df.info()` untuk mendapatkan gambaran umum tentang dataset, termasuk jumlah baris, kolom dan tipe data. Dataset ini memiliki total **11,914 entri**, yang berarti ada 11,914 sampel data, terdiri dari **16 kolom**, yang mencakup berbagai atribut kendaraan seperti merek, model, tahun produksi, dan lainnya. Tipe data `object` (Kolom dengan data teks atau kategorik, seperti `Make`, `Model`, dan `Engine Fuel Type`). `int64` (Kolom dengan data numerik integer, seperti `Year` dan `MSRP`). `float64`* (Kolom dengan data numerik desimal, seperti `Engine HP` dan `Engine Cylinders`).

- Memeriksa Duplikasi Data

Mengidentifikasi jumlah baris duplikat dalam dataset, menggunakan fungsi `df.duplicated().sum()`. Pada data ini menghasilkan 715 baris duplikat. Baris duplikat dapat menyebabkan bias dalam analisis data dan pemodelan machine learning. 

- Menghapus Data Duplikat

Menghapus 715 baris duplikat untuk memastikan dataset bersih dari data redundan, menggunakan fungsi `df.drop_duplicates()`. Menghapus data duplikat bertujuan untuk mencegah bias dalam analisis dan model machine learning. Baris yang sama persis dapat memengaruhi hasil analisis atau prediksi, sehingga penghapusan duplikat adalah langkah penting dalam pembersihan data.

- Identifikasi Missing Values

Menghitung jumlah Missing Values di setiap kolom. Pada langkah ini, kita menggunakan fungsi `df.isnull().sum()` untuk menghitung jumlah missing values di setiap kolom dalam dataset. Analisis ini membantu mengidentifikasi kolom mana yang memerlukan penanganan, seperti imputasi atau penghapusan baris/kolom. Langkah ini penting untuk memastikan dataset siap digunakan untuk analisis dan pemodelan machine learning. Pada data proyek ini menunjukkan beberapa kolom yang teridentifikasi yaitu `Engine Fuel Type` : 3 nilai, `Engine HP` : 69 nilai, `Engine Cylinders` : 30 nilai, `Number of Doors` : 6 nilai, `Market Category` : 3,376 nilai.

- Menghapus Kolom yang tidak Relevan

Menghapus kolom dianggap yang tidak relevan atau tidak memiliki pengaruh sifnifikan terhadap target prediksi (harga/MSRP). Dalam proyek ini, kolom seperti  "Model", "Market Category", dan "Number of Doors" dihapus karena dianggap tidak relevan untuk target  (harga/MSRP). Hal ini bertujuan Mengurangi kompleksitas dataset dengan hanya menyertakan fitur yang relevan untuk prediksi harga, selain itu menghindari memasukkan kolom yang dapat menyebabkan noise atau mengurangi akurasi model. Langkah ini merupakan bagian penting dari proses seleksi fitur, memastikan hanya fitur yang relevan digunakan untuk pemodelan.

- Menangani Missing Values

Setelah missing values diidentifikasi, langkah selanjutnya adalah menanganinya dengan menggunakan metode imputasi berbasis statistik. Metode imputasi ini menjaga informasi dalam data sebanyak mungkin tanpa menghapus baris yang mengandung nilai kosong. Fungsi `(df.median())` digunakan untuk mengisi nilai kosong pada kolom numerik dengan nilai tengah (median) dari kolom tersebut. Median digunakan untuk mengurangi dampak outlier pada data, sehingga lebih representatif dibandingkan rata-rata. Kemudian fungsi `(df.mode())` digunakan untuk mengisi nilai kosong pada kolom non-numerik (kategorik) dengan nilai yang paling sering muncul (modus) di kolom tersebut. Mode dipilih karena mode merepresentasikan kategori yang paling umum. Strategi ini memastikan dataset tetap utuh dan representatif. 

- One-hot encoding  

One hot encoding adalah teknik mengubah data kategorik menjadi data numerik dimana setiap kategori menjadi kolom baru dengan nilai 0 atau 1. Fitur yang akan diubah menjadi numerik pada proyek ini adalah make, model, engine fuel type, transmission type, driven_wheels, vehicle size, dan vehicle type

- Train Test Split

Train test split merupakan proses membagi data menjadi data latih dan data uji. Data latih akan digunakan untuk membangun model, sedangkan data uji akan digunakan untuk menguji performa model. Pada proyek ini dataset sebanyak 11199 dibagi menjadi 80% untuk data latih yaitu 8959 dan untuk data uji menjadi 20% yaitu 2240 

- Normalization

Algoritma machine learning akan memiliki performa lebih baik dan bekerja lebih cepat jika dimodelkan dengan data seragam yang memiliki skala relatif sama. Salah satu teknik normalisasi yang digunakan pada proyek ini adalah Standarisasi dengan `StandardScaler` dari libarary sklearn. Proses ini bertujuan untuk menyelaraskan skala data agar algoritma machine learning dapat bekerja lebih optimal.


## Modeling
Algoritma pada proyek ini melakukan pemodelan dengan melatih 3 algoritma dengan pendekatan yang berbeda dalam memprediksi harga kmobil (MSRP), yaitu Random Forest, Gradient Boosting dan Linear Regression. Masing-masing model menggunakan parameter tertentu yang telah disesuaikan.
- **Random Forest**. Algoritma random forest adalah teknik dalam machine learning dengan metode ensemble. Proyek ini menggunakan sklearn.ensemble.RandomForestRegressor dengan memasukkan `X_train` dan `y_train` dalam membangun model. Parameter yang digunakan pada proyek ini adalah: 
    * `random_state=42` : Mengontrol seed acak yang diberikan pada setiap base_estimator pada setiap iterasi boosting.
    * `n_estimators=100` : Jumlah pohon keputusan yang digunakan dalam ensemble.
    * `max_depth=20` : Kedalaman maksimum setiap tree Kedalaman maksimum setiap pohon untuk mencegah overfitting.
   
- **Gradient Boosting**. Gradient Boosting adalah algoritma machine learning berbasis ensemble. Untuk proyek ini menggunakan sklearn.ensemble.GradientBoostingRegressor dengan parameter sebagai berikut:
   * `n_estimators=200`: Jumlah iterasi boosting.
   * `max_depth=5`: Kedalaman maksimum setiap pohon keputusan.
   *  `learning_rate=0.2`: Mengontrol kontribusi setiap pohon terhadap prediksi akhir.

- **Linear Regression**. Linear Regression adalah algoritma machine learning yang sederhana namun sangat efektif untuk memodelkan hubungan antara variabel independen (fitur) dan variabel dependen (target). Proyek ini menggunakan sklearn.linear_model.LinearRegression dengan parameter sebagai berikut:
   * `fit_intercept=True`: Model menghitung intercept (titik potong dengan sumbu y).
   * `n_jobs=-1`: Memungkinkan komputasi paralel untuk mempercepat pelatihan.
  

## Evaluation
Langkah selanjutnya evaluasi performa masing-masing model menggunakan data uji (`X_test` dan `y_test`) dengan metrik seperti Mean Squared Error (MSE) dan R² score.
Mean Square Error (MSE) adalah salah satu metrik evaluasi yang umum digunakan dalam statistik dan machine learning. Fungsi dari MSE adalah untuk mengukur seberapa baik suatu model dalam memetakan nilai prediksi ke nilai sebenarnya dengan menggunakan kuadrat kesalahan sekecil mungkin sebagai dasar perhitungan. Berikut rumus yang digunakan untuk menghitung MSE:

![MSE](https://github.com/user-attachments/assets/613cf189-0c04-4a36-a697-785c5d4ba6c7)

Keterangan:
- n adalah jumlah total observasi atau sampel.
- yi adalah nilai aktual atau nilai sebenarnya dari observasi ke-i.
- Ŷi adalah nilai prediksi model untuk observasi ke-i.

R Square (R²) adalah ukuran statistik yang menunjukkan seberapa besar variasi suatu variabel dependen dapat dijelaskan oleh variabel independen dalam suatu model regresi. Dengan kata lain, R² menunjukkan seberapa baik suatu model regresi (variabel independen) memprediksi hasil data observasi (variabel dependen). R² merupakan angka yang berkisar antara 0 sampai 1 yang mengindikasikan besarnya kombinasi variabel independen secara bersama – sama mempengaruhi nilai variabel dependen. Terdapat tiga kategori pengelompokan pada nilai R² yaitu kategori kuat, kategori moderat, dan kategori lemah [Hair et al., 2011](https://accounting.binus.ac.id/2021/08/12/memahami-r-square-koefisien-determinasi-dalam-penelitian-ilmiah/). Hair et al menyatakan bahwa nilai R square 0,75 termasuk ke dalam kategori kuat, nilai R square 0,50 termasuk kategori moderat dan nilai R square 0,25 termasuk kategori lemah. Untuk menghitung R², berikut [rumusnya](https://info.populix.co/articles/r-square-adalah/):

![capture-20231123-142806](https://github.com/user-attachments/assets/36c47294-2e99-4555-8c3b-06244e132a09)

Keterangan:

* SSregression adalah jumlah kuadrat akibat regresi (dijelaskan jumlah kuadrat)
* SStotal adalah jumlah total kuadrat
Jumlah kuadrat akibat regresi mengukur seberapa baik model regresi mewakili data yang digunakan untuk pemodelan. Sementara jumlah total kuadrat mengukur variasi data yang diamati (data yang digunakan dalam pemodelan regresi).

Berikut dibawah ini adalah hasil evaluasi performa masing-masing model yaitu Random Forest, Gradient Boosting, dan Linear Regression:

![hasil evaluasi](https://github.com/user-attachments/assets/3d13c30a-855f-4e40-8b8d-9574b92f7dfe)


![perbandingan](https://github.com/user-attachments/assets/597df1dc-c45d-4a83-963b-6110994a1297)

Tabel dan grafik ini sama-sama menunjukkan evaluasi kinerja tiga model machine learning Random Forest, Gradient Boosting, dan Linear Regression berdasarkan metrik MSE (Mean Squared Error) dan R² Score. Terlihat pada tabel bahwa Gradient Boosting menunjukkan performa terbaik dengan R² Score sebesar 0.963868 dan MSE terendah, menandakan bahwa model ini memiliki kemampuan prediksi yang sangat baik dan error yang kecil. Grafik R² Score juga mendukung hasil ini, menunjukkan bahwa Gradient Boosting memiliki nilai R² mendekati 1, yang berarti model ini mampu menunjukkan variabilitas data dengan sangat baik. Random Forest juga memiliki performa yang cukup baik dengan R² Score sebesar 0.899657, meskipun MSE-nya lebih tinggi dibanding Gradient Boosting. Sebaliknya, Linear Regression memiliki performa yang jauh lebih rendah, dengan R² Score hanya 0.417905 dan MSE yang tinggi, menunjukkan bahwa model ini kurang cocok untuk data yang lebih kompleks. Jadi Gradient Boosting sebagai pilihan terbaik untuk memaksimalkan akurasi prediksi dalam kasus ini.


![predict](https://github.com/user-attachments/assets/6c19f57a-c4ae-45ea-b86c-d5031be408c9)

Tabel ini menampilkan perbandingan antara nilai aktual (Actual) dan nilai prediksi (Predicted) dari model Gradient Boosting untuk beberapa sampel data. Secara umum, hasil prediksi terlihat cukup mendekati nilai aktual, dengan perbedaan yang relatif kecil, seperti pada baris pertama di mana nilai aktual adalah 34,160, sementara nilai prediksi adalah 33,757.54. Hal ini menunjukkan bahwa model memiliki kemampuan prediksi yang cukup baik dan dapat menangkap pola dalam data dengan akurasi yang layak. 


### Dampak Evaluasi Model Terhadap Business Understanding
Model yang dievaluasi menunjukkan bahwa beberapa fitur seperti `Year`, `Engine HP`, `Engine Cylinder`s, dan `Vehicle Size` memiliki pengaruh besar terhadap harga jual mobil (MSRP) berdasarkan analisis korelasi dan pentingnya fitur dalam model. Model terbaik, yaitu `Gradient Boosting`, berhasil mencapai nilai `R² Score yang tinggi` dan `MSE yang rendah`, menunjukkan akurasi prediksi yang sangat baik, sehingga mampu memprediksi harga mobil secara akurat berdasarkan karakteristik tertentu. Langkah-langkah seperti penanganan `missing values`, `encoding` fitur kategorik, dan `normalisasi` fitur numerik memastikan data dalam kondisi optimal untuk pemodelan, memungkinkan tercapainya tujuan membangun model prediksi yang akurat. Solusi ini berdampak positif bagi dealer mobil dengan membantu menetapkan harga kompetitif, mengurangi risiko overpricing atau underpricing, bagi konsumen dengan memastikan harga yang mencerminkan nilai pasar kendaraan, serta bagi produsen mobil dengan memberikan wawasan tentang atribut yang memengaruhi harga, yang dapat digunakan untuk mengembangkan produk baru sesuai kebutuhan pasar. Dengan demikian, model ini tidak hanya memenuhi problem statement tetapi juga memberikan solusi yang relevan dan berdampak signifikan dalam pengambilan keputusan berbasis data di pasar kendaraan.

