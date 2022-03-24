## Structured vs Unstrucuted Data
1. 8 MP camera dapat diartikan membuat **Sebuah gambar** dengan 8 **Million** Piksel , dan tentu hal ini dapat meningkantakna Input **Size*** Dari ML Modelnya sendiri which *Leads to* Inssufficient computing power dan juga **Long Training**
2. Take a look at this picture ![[Pasted image 20220324091031.png]] Apabila melihat dari gambar, **Model Machine learning** Bakalan melakukan komparasi Piksel piksel, jadi **Kedua gambar tersebut berbeda**
3. Dalam Melakukan komparasi kesamaan antar kedua Vektor  **bisa bekerja lebih baik** pada **Structured data**. Namun, ketika dihadapkan dengan **Unstructured data** Become ga baik.  karena kita  **Membadingkan Piksel images**. Take Look Contoh gambar sebelumnya deh . Ketika kamu coba melakukan ZOOM in, maka nanti piksel nya berubah ya gak ? Nah ketika pikselnya berubah maka tidak bisa lagi tuh kamu melakukan yang namanya ***Euclidean distnace*** Hasilnya bakalan **Unreliable**.
4. ***Semakin Kecil Nilai Euclidean Distance*** Semakin **MIRIP**
5. VISI KOMPUTER Means --> **BAGAIMANA kita modelling Hubungan antara piksel ?**
## Images note
1. Pada saat melakukan **Trainig** pada image, adabaiknya kita memahami dahulu **Konteks permasalahannya**, 
	1. MultiClass --> **Kita prediksi satu hal dengan n Classes**
2. Apabila Modelnya **Unsure** yang berarti kurang paham atau kesulitan, **Silahkan Lihat confidence nya dan juga yang namanya Kemampuan manusia**
3. Softmax Function pada dasaranya adalah ***Mengubah nilai Floating values menjadi probabilities Ranging dari m - n*** Tapi ga merubah relative Ordernya ( Atau Urutan sebelumnya dengan cara melakukan **Ekponensiasi dan normalisasi pada inputnya**). Dimana eksponensiasi berarti **Memperbesar BOBOT dari floating point tersebut **
4. Loss Function --> Memberitahukan **Kualitas Dari Solusi/prediksi yang dilakukan oleh model** Semakin Kecil semakin baik
5. Dude ini Note penting, sebelum lu melakukan Komputasi **Mengenai Deep Neural net** Pastikan Simpler modelnya udah ga bisa dipake ya, **even sudah berusaha menggunakan feature engineering** Apalagi ketika datasetnya **Non Linear** 