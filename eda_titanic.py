import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

url = "https://raw.githubusercontent.com/Salma-AZIZ/IP302/main/data/titanic.csv"
titanic_data = pd.read_csv(url)

# Preview data
print(titanic_data.head())

# Shape of dataset
print("Rows:", titanic_data.shape[0])
print("Columns:", titanic_data.shape[1])

# Info
titanic_data.info()

# Summary statistics
print(titanic_data.describe())

# Categorical variables
print(titanic_data["Sex"].value_counts())
print(titanic_data["Embarked"].value_counts())
print(titanic_data.describe(include=["object"]))

# Missing values
print(titanic_data.isnull().sum())
print(titanic_data.isnull().mean() * 100)

# Histograms
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.histplot(titanic_data["Age"].dropna(), kde=True, bins=30)
plt.title("Distribution of Age")

plt.subplot(1, 2, 2)
sns.histplot(titanic_data["Fare"], kde=True, bins=30)
plt.title("Distribution of Fare")
plt.tight_layout()
plt.savefig("histograms.png")
plt.close()

# Boxplots
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.boxplot(data=titanic_data, x="Age")
plt.title("Age Boxplot")

plt.subplot(1, 2, 2)
sns.boxplot(data=titanic_data, x="Fare")
plt.title("Fare Boxplot")
plt.tight_layout()
plt.savefig("boxplots.png")
plt.close()

# Correlation matrix
plt.figure(figsize=(8, 6))
corr = titanic_data.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.close()
