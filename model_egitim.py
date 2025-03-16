import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import joblib

# Veri setini yükle
df = pd.read_csv('veri.csv')

# Kategorik değişkenleri dönüştür
le = LabelEncoder()
kategorik_kolonlar = ['Marka', 'Model', 'Yakit_Tipi', 'Vites_Tipi']
for kolon in kategorik_kolonlar:
    df[kolon] = le.fit_transform(df[kolon])

# Özellikler ve hedef değişkeni ayır
X = df.drop(['Fiyat'], axis=1)
y = df['Fiyat']

# Veriyi eğitim ve test olarak böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaler'ı oluştur ve uygula
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modelleri tanımla
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'SVR': SVR(kernel='rbf'),
    'XGBoost': xgb.XGBRegressor(random_state=42),
    'LightGBM': lgb.LGBMRegressor(random_state=42),
    'CatBoost': cb.CatBoostRegressor(verbose=False, random_state=42),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'AdaBoost': AdaBoostRegressor(random_state=42),
    'KNN': KNeighborsRegressor(n_neighbors=5)
}

# Sonuçları saklamak için sözlük
results = {}

# Her modeli eğit ve değerlendir
for name, model in models.items():
    # Modeli eğit
    model.fit(X_train_scaled, y_train)
    
    # Tahmin yap
    y_pred = model.predict(X_test_scaled)
    
    # RMSE hesapla
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results[name] = rmse
    
    print(f'{name} RMSE: {rmse:,.2f}')

# En iyi modeli bul
best_model_name = min(results, key=results.get)
best_model = models[best_model_name]
print(f'\nEn iyi model: {best_model_name} (RMSE: {results[best_model_name]:,.2f})')

# En iyi modeli ve scaler'ı kaydet
joblib.dump(best_model, 'eniyi.joblib')
joblib.dump(scaler, 'scaler.joblib')

# Sonuçları görselleştir
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.bar(results.keys(), results.values())
plt.xticks(rotation=45)
plt.title('Model Karşılaştırması - RMSE Değerleri')
plt.ylabel('RMSE')
plt.tight_layout()
plt.show()
