import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# Data Acquisition & Loading
url = "https://raw.githubusercontent.com/Salma-AZIZ/IP302/main/data/titanic_preprocessed.csv"
titanic_data = pd.read_csv(url)

# Overview & info
print(titanic_data.head())
titanic_data.info()
print(titanic_data.describe())

# Check missing and duplicates
print("Missing values:\n", titanic_data.isnull().sum())

duplicate_count = titanic_data.duplicated().sum()
print("Duplicates before:", duplicate_count)
if duplicate_count > 0:
    titanic_data = titanic_data.drop_duplicates()
print("Duplicates after:", titanic_data.duplicated().sum())

# One-Hot Encoding: Embarked
if "Embarked" in titanic_data.columns:
    embarked_dummies = pd.get_dummies(titanic_data["Embarked"], drop_first=True)
    titanic_data = pd.concat([titanic_data, embarked_dummies], axis=1)
    titanic_data.drop(columns=["Embarked"], inplace=True)

# Label Encoding: Sex
if "Sex" in titanic_data.columns and titanic_data["Sex"].dtype == "object":
    encoder = LabelEncoder()
    titanic_data["Sex"] = encoder.fit_transform(titanic_data["Sex"])

# Binning: Age Group
bins = [0, 12, 19, 30, 50, 120]
labels = ["Child", "Teenager", "Young Adult", "Adult", "Senior"]
titanic_data["Age_Group"] = pd.cut(titanic_data["Age"], bins=bins, labels=labels, right=False)

plt.figure(figsize=(8, 4))
sns.countplot(x="Age_Group", data=titanic_data)
plt.title("Passenger Count by Age Group")
plt.tight_layout()
plt.savefig("age_group_distribution.png")
plt.close()

# Survival Rate by Age Group
plt.figure(figsize=(8, 4))
sns.countplot(x="Age_Group", hue="Survived", data=titanic_data)
plt.title("Survival by Age Group")
plt.tight_layout()
plt.savefig("survival_age_group.png")
plt.close()

# Log Transformation: Fare
titanic_data["Log_Fare"] = np.log1p(titanic_data["Fare"])

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.histplot(titanic_data["Fare"], kde=True)
plt.title("Original Fare")

plt.subplot(1, 2, 2)
sns.histplot(titanic_data["Log_Fare"], kde=True)
plt.title("Log Transformed Fare")
plt.tight_layout()
plt.savefig("fare_log_comparison.png")
plt.close()

# Feature Scaling: Normalization (Min-Max)
min_max_scaler = MinMaxScaler()
titanic_data["Age_normalized"] = min_max_scaler.fit_transform(titanic_data[["Age"]])
titanic_data["Fare_normalized"] = min_max_scaler.fit_transform(titanic_data[["Fare"]])

# Feature Scaling: Standardization (Z-score)
standard_scaler = StandardScaler()
titanic_data["Age_standardized"] = standard_scaler.fit_transform(titanic_data[["Age"]])
titanic_data["Fare_standardized"] = standard_scaler.fit_transform(titanic_data[["Fare"]])

# Interaction Feature: Age_Class
if "Pclass" in titanic_data.columns:
    titanic_data["Age_Class"] = titanic_data["Age"] * titanic_data["Pclass"]

print("\nFinal Data Preview:")
print(titanic_data.head())
