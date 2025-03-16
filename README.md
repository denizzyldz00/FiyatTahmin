Bu proje Makine Öğrenmesi(Machine Learning) dersi için ödev projesi olarak hazırlanmıştır.

Bu proje, makine öğrenmesi modeli kullanarak araç özelliklerine göre fiyat tahmini yapan bir Streamlit web uygulamasıdır.
Bu proje, tamamen yapay zeka kullanılarak hazırlanmış bir makine öğrenmesi projesidir.
Özellikler
Marka, model, yıl, kilometre, yakıt tipi, vites tipi, motor hacmi ve motor gücü bilgilerine göre araç fiyat tahmini
Kullanıcı dostu arayüz
Gerçek zamanlı tahmin
Kurulum
pip install streamlit pandas joblib numpy



Kullanım
streamlit run app.py



Gerekli Dosyalar
app.py: Ana uygulama kodu
eniyi.joblib: Eğitilmiş makine öğrenmesi modeli
scaler.joblib: Veri ölçeklendirme modeli
feature_names.joblib: Özellik isimleri
Teknolojiler
Python
Streamlit
Pandas
Joblib
NumPy
