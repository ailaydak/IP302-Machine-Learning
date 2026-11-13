import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Task 3: Load and inspect dataset
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedianHouseValue'] = housing.target

print(df.head())
df.info()
print(df.describe())
print("Missing values:\n", df.isnull().sum())

# Task 4: Feature selection & correlation
plt.figure(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("mlr_correlation.png")
plt.close()

# Selected features (dropping AveBedrms and Longitude to reduce multicollinearity)
selected_features = ['MedInc', 'HouseAge', 'AveRooms', 'Population', 'AveOccup', 'Latitude']
X = df[selected_features]
y = df['MedianHouseValue']

# Task 5: Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Task 6: Standardization
scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=selected_features, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=selected_features, index=X_test.index)

# Task 7: Build and train model
mlr_model = LinearRegression()
mlr_model.fit(X_train_scaled, y_train)

# Task 8 & 11: Model parameters
print("Intercept:", mlr_model.intercept_)
for feature, coef in zip(selected_features, mlr_model.coef_):
    print(f"{feature}: {coef:.4f}")

# Task 9: Model evaluation
y_train_pred = mlr_model.predict(X_train_scaled)
y_test_pred = mlr_model.predict(X_test_scaled)

print("\nTraining Metrics:")
print("R2:", r2_score(y_train, y_train_pred))
print("MAE:", mean_absolute_error(y_train, y_train_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_train, y_train_pred)))

print("\nTest Metrics:")
print("R2:", r2_score(y_test, y_test_pred))
print("MAE:", mean_absolute_error(y_test, y_test_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_test_pred)))

# Task 10: Model diagnostics (Residuals)
residuals = y_test_pred - y_test
plt.figure(figsize=(8, 5))
plt.scatter(y_test_pred, residuals, alpha=0.3)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.tight_layout()
plt.savefig("mlr_residuals.png")
plt.close()

# Task 12 & 13: Outlier removal function and re-training
def remove_outliers(dataframe):
    df_clean = dataframe.copy()
    for col in df_clean.select_dtypes(include=[np.number]).columns:
        q1 = df_clean[col].quantile(0.25)
        q3 = df_clean[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

df_no_outliers = remove_outliers(df[selected_features + ['MedianHouseValue']])
print("\nShape after outlier removal:", df_no_outliers.shape)

X_clean = df_no_outliers[selected_features]
y_clean = df_no_outliers['MedianHouseValue']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

scaler_c = StandardScaler()
X_train_c_scaled = scaler_c.fit_transform(X_train_c)
X_test_c_scaled = scaler_c.transform(X_test_c)

mlr_model_clean = LinearRegression()
mlr_model_clean.fit(X_train_c_scaled, y_train_c)

y_test_pred_c = mlr_model_clean.predict(X_test_c_scaled)

print("\nTest Metrics Without Outliers:")
print("R2:", r2_score(y_test_c, y_test_pred_c))
print("MAE:", mean_absolute_error(y_test_c, y_test_pred_c))
print("RMSE:", np.sqrt(mean_squared_error(y_test_c, y_test_pred_c)))
