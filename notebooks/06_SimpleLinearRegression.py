import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Task 3 & 4: Load and explore dataset
housing_data = fetch_california_housing()
print("Keys:", housing_data.keys())
print("Feature shape:", housing_data.data.shape)
print("Target shape:", housing_data.target.shape)
print("Feature names:", housing_data.feature_names)

# Task 5: Create DataFrame
df = pd.DataFrame(housing_data.data, columns=housing_data.feature_names)
df['MedianHouseValue'] = housing_data.target
print(df.head())

# Task 6: Data inspection & statistics
df.info()
print(df.describe())
print("Missing values:\n", df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())

# Task 7: Histograms
df.hist(bins=30, figsize=(12, 8))
plt.tight_layout()
plt.savefig("housing_histograms.png")
plt.close()

# Task 8: Boxplots
plt.figure(figsize=(14, 6))
sns.boxplot(data=df)
plt.title("Distribution of Features")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("housing_boxplots.png")
plt.close()

# Task 9: Correlation analysis
plt.figure(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()

# Task 10: Spatial visualization
plt.figure(figsize=(8, 6))
plt.scatter(df['Longitude'], df['Latitude'], c=df['MedianHouseValue'], cmap='coolwarm', alpha=0.4, s=10)
plt.colorbar(label='MedianHouseValue')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Geospatial Distribution of House Values')
plt.tight_layout()
plt.savefig("spatial_distribution.png")
plt.close()

# Task 11: Setup dataset for Simple Linear Regression
X = df[['MedInc']]
y = df['MedianHouseValue']
print("X shape:", X.shape)
print("y shape:", y.shape)

# Task 12: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Task 13: Train model
simple_reg_model = LinearRegression()
simple_reg_model.fit(X_train, y_train)

# Task 14: Evaluate on training data
y_train_pred = simple_reg_model.predict(X_train)
mse_train = mean_squared_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)
print(f"Training MSE: {mse_train:.2f}")
print(f"Training R2: {r2_train:.2f}")

# Task 15: Visualize fit on training data
plt.figure(figsize=(8, 5))
plt.scatter(X_train, y_train, alpha=0.2, label='Actual Data')
plt.plot(X_train.values.ravel(), y_train_pred, color='red', label='Regression Line')
plt.xlabel('MedInc')
plt.ylabel('MedianHouseValue')
plt.title('Fit on Training Data')
plt.legend()
plt.tight_layout()
plt.savefig("train_fit.png")
plt.close()

# Task 16: Model parameters
slope = simple_reg_model.coef_[0]
intercept = simple_reg_model.intercept_
print(f"Slope: {slope:.4f}")
print(f"Intercept: {intercept:.4f}")
print(f"Equation: y = {slope:.4f} * MedInc + {intercept:.4f}")

# Task 17: Assess on test data
y_test_pred = simple_reg_model.predict(X_test)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
print(f"Test MSE: {mse_test:.2f}")
print(f"Test R2: {r2_test:.2f}")

# Task 18: Visualize on test data
plt.figure(figsize=(8, 5))
plt.scatter(X_test, y_test, alpha=0.2, label='Actual Test Data')
plt.plot(X_test.values.ravel(), y_test_pred, color='red', label='Regression Line')
plt.xlabel('MedInc')
plt.ylabel('MedianHouseValue')
plt.title('Fit on Test Data')
plt.legend()
plt.tight_layout()
plt.savefig("test_fit.png")
plt.close()
