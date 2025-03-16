import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Modeli ve gerekli dosyaları yükle
loaded_model = joblib.load('eniyi.joblib')
scaler = joblib.load('scaler.joblib')
feature_names = joblib.load('feature_names.joblib')

# Sabit listeler
markalar = ['Toyota', 'Honda', 'Ford', 'Volkswagen', 'BMW', 'Mercedes', 'Audi', 'Hyundai', 'Renault', 'Fiat']
modeller = {
    'Toyota': ['Corolla', 'Yaris', 'RAV4'],
    'Honda': ['Civic', 'City', 'CR-V'],
    'Ford': ['Focus', 'Fiesta', 'Kuga'],
    'Volkswagen': ['Golf', 'Polo', 'Passat'],
    'BMW': ['3 Serisi', '5 Serisi', 'X5'],
    'Mercedes': ['C Serisi', 'E Serisi', 'A Serisi'],
    'Audi': ['A3', 'A4', 'Q5'],
    'Hyundai': ['i20', 'Tucson', 'i10'],
    'Renault': ['Clio', 'Megane', 'Kadjar'],
    'Fiat': ['Egea', '500', 'Doblo']
}
yakit_tipleri = ['Benzin', 'Dizel', 'LPG']
vites_tipleri = ['Manuel', 'Otomatik']

st.title('Araç Fiyat Tahmini')

# Kullanıcı girişleri
col1, col2 = st.columns(2)

with col1:
    marka = st.selectbox('Marka', markalar)
    model = st.selectbox('Model', modeller[marka])
    yil = st.number_input('Model Yılı', min_value=2010, max_value=2024, value=2020)
    kilometre = st.number_input('Kilometre', min_value=0, max_value=300000, value=50000)

with col2:
    yakit = st.selectbox('Yakıt Tipi', yakit_tipleri)
    vites = st.selectbox('Vites Tipi', vites_tipleri)
    motor_hacmi = st.selectbox('Motor Hacmi', [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2])
    motor_gucu = st.number_input('Motor Gücü (HP)', min_value=75, max_value=250, value=100)

if st.button('Fiyat Tahmini Yap'):
    # Boş DataFrame oluştur
    input_df = pd.DataFrame(0, index=[0], columns=feature_names)
    
    # Sayısal değerleri ata
    input_df['Model_Yili'] = float(yil)
    input_df['Kilometre'] = float(kilometre)
    input_df['Motor_Hacmi'] = float(motor_hacmi)
    input_df['Motor_Gucu'] = float(motor_gucu)
    
    # Kategorik değişkenleri ata
    input_df[f'Marka_{marka}'] = 1
    input_df[f'Model_{model}'] = 1
    input_df[f'Yakit_Tipi_{yakit}'] = 1
    input_df[f'Vites_Tipi_{vites}'] = 1
    
    # Verileri ölçeklendir
    input_scaled = scaler.transform(input_df)
    
    # Tahmin yap
    prediction = abs(loaded_model.predict(input_scaled)[0])
    
    st.success(f'Tahmini Fiyat: {prediction:,.0f} TL')
