import joblib

# feature_names'i yükle
features = joblib.load('feature_names.joblib')

# Tüm özellikleri yazdır
print("Özellik sayısı:", len(features))
print("\nÖzellik isimleri:")
for i, feature in enumerate(features, 1):
    print(f"{i}. {feature}")
