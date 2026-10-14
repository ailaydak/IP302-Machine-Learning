# Task 3: Kutuphaneleri Ice Aktarma
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Task 1 & 4: Veriyi GitHub'dan Yukleme
url = "https://raw.githubusercontent.com/Salma-AZIZ/IP302/main/data/titanic.csv"
titanic_data = pd.read_csv(url)

# Task 5: Ilk 5 Satir
print("--- Task 5: Ilk 5 Satir ---")
print(titanic_data.head())

# Task 6: Boyutlar
print("\n--- Task 6: Boyutlar ---")
print("# of rows:", titanic_data.shape[0])
print("# of columns:", titanic_data.shape[1])

# Task 7: Veri Ozeti
print("\n--- Task 7: Veri Ozeti ---")
titanic_data.info()

# Task 8: Sayisal ve Kategorik Ozetler
print("\n--- Task 8: Sayisal Degiskenler Ozeti ---")
print(titanic_data.describe())

print("\n--- Task 8: Kategorik Degiskenler (Sex ve Embarked) ---")
print("Sex Dagilimi:\n", titanic_data["Sex"].value_counts())
print("\nEmbarked Dagilimi:\n", titanic_data["Embarked"].value_counts())

print("\n--- Task 8: Kategorik Degiskenler (describe ile) ---")
print(titanic_data.describe(include=["object"]))

# Task 9: Eksik Deger Analizi
print("\n--- Task 9: Eksik Deger Sayilari ---")
print(titanic_data.isnull().sum())
print("\nEksik Deger Yuzdeleri (%):")
print(titanic_data.isnull().mean() * 100)

# Task 10: Histogramlar
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.histplot(titanic_data["Age"].dropna(), kde=True, bins=30)
plt.title("Age Dagilimi")

plt.subplot(1, 2, 2)
sns.histplot(titanic_data["Fare"], kde=True, bins=30)
plt.title("Fare Dagilimi")
plt.tight_layout()
plt.savefig("histograms.png")
plt.close()

# Task 11: Kutu Grafikleri (Box Plot)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.boxplot(data=titanic_data, x="Age")
plt.title("Age Kutu Grafigi")

plt.subplot(1, 2, 2)
sns.boxplot(data=titanic_data, x="Fare")
plt.title("Fare Kutu Grafigi")
plt.tight_layout()
plt.savefig("boxplots.png")
plt.close()

# Task 12: Korelasyon Isı Haritasi
plt.figure(figsize=(8, 6))
correlation_matrix = titanic_data.corr(numeric_only=True)
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Korelasyon Isi Haritasi")
plt.savefig("correlation_heatmap.png")
plt.close()

print("\nTum grafikler ve analizler basariyla tamamlandi!")
