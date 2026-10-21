import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
url = "https://raw.githubusercontent.com/Salma-AZIZ/IP302/main/data/titanic.csv"
titanic_data = pd.read_csv(url)

# Initial preview & dimensions
print(titanic_data.head(3))
print("Dataset shape:", titanic_data.shape)

# Dataset details
titanic_data.info()
print(titanic_data.describe())

# Identifying variable types
continuous_vars = ["Age", "Fare"]
discrete_vars = ["SibSp", "Parch"]
categorical_vars = titanic_data.select_dtypes(include=["object"]).columns.tolist()
id_vars = [col for col in titanic_data.columns if titanic_data[col].nunique() == len(titanic_data)]

print("Continuous:", continuous_vars)
print("Discrete:", discrete_vars)
print("Categorical:", categorical_vars)
print("ID Variables:", id_vars)

# Categorical visualization: Count plots
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.countplot(x="Sex", data=titanic_data)
plt.title("Count of Sex")

plt.subplot(1, 2, 2)
sns.countplot(x="Embarked", data=titanic_data)
plt.title("Count of Embarked")
plt.tight_layout()
plt.savefig("countplot_categorical.png")
plt.close()

# Crosstabs & Heatmap
ct = pd.crosstab(titanic_data["Sex"], titanic_data["Survived"])
print("\nCrosstab Sex vs Survived:\n", ct)

plt.figure(figsize=(6, 4))
sns.heatmap(ct, annot=True, cmap="Blues", fmt="d")
plt.title("Sex vs Survived Heatmap")
plt.savefig("crosstab_heatmap.png")
plt.close()

# Detecting Outliers using IQR
for col in ["Age", "Fare"]:
    q1 = titanic_data[col].quantile(0.25)
    q3 = titanic_data[col].quantile(0.75)
    iqr = q3 - q1
    outliers = titanic_data[(titanic_data[col] < (q1 - 1.5 * iqr)) | (titanic_data[col] > (q3 + 1.5 * iqr))]
    print(f"Number of outliers in {col}:", len(outliers))

# Skewness
for col in ["Age", "Fare"]:
    print(f"Skewness of {col}:", titanic_data[col].skew())

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.histplot(titanic_data["Age"], kde=True)
plt.title("Age Distribution & KDE")

plt.subplot(1, 2, 2)
sns.histplot(titanic_data["Fare"], kde=True)
plt.title("Fare Distribution & KDE")
plt.tight_layout()
plt.savefig("skewness_kde.png")
plt.close()

# Handling Missing Values
missing_percent = titanic_data.isnull().mean() * 100
print("\nMissing Values Percentage:\n", missing_percent)

# Drop columns with > 70% missing
cols_to_drop = missing_percent[missing_percent > 70].index
titanic_data.drop(columns=cols_to_drop, inplace=True)
print("Dropped columns:", list(cols_to_drop))

# Imputation (Median for Age due to skewness, Mode for Embarked)
if "Age" in titanic_data.columns:
    titanic_data["Age"].fillna(titanic_data["Age"].median(), inplace=True)

if "Embarked" in titanic_data.columns:
    titanic_data["Embarked"].fillna(titanic_data["Embarked"].mode()[0], inplace=True)

print("\nMissing values after preprocessing:\n", titanic_data.isnull().sum())

# Log Transformation for Fare
titanic_data["Log_Fare"] = np.log1p(titanic_data["Fare"])

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.histplot(titanic_data["Fare"], kde=True)
plt.title("Fare Before Log Transform")

plt.subplot(1, 2, 2)
sns.histplot(titanic_data["Log_Fare"], kde=True)
plt.title("Fare After Log Transform")
plt.tight_layout()
plt.savefig("log_fare_comparison.png")
plt.close()
