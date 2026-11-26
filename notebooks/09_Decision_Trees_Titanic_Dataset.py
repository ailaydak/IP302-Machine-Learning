import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

url = "https://raw.githubusercontent.com/Salma-AZIZ/IP302/main/data/titanic.csv"
df = pd.read_csv(url)

print(df.head())
df.info()
print(df.describe())
print(df.isnull().sum())

selected_features = ['Age', 'Sex', 'Pclass', 'Fare', 'SibSp', 'Parch', 'Embarked']
df_features = df[selected_features].copy()
df_features['Family_Size'] = df_features['SibSp'] + df_features['Parch'] + 1
df_features.drop(columns=['SibSp', 'Parch'], inplace=True)

X = df_features
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

num_cols = ['Age', 'Fare', 'Pclass', 'Family_Size']
cat_cols = ['Sex', 'Embarked']

num_imputer = SimpleImputer(strategy='median')
X_train_num = pd.DataFrame(
    num_imputer.fit_transform(X_train[num_cols]),
    columns=num_cols,
    index=X_train.index
)
X_test_num = pd.DataFrame(
    num_imputer.transform(X_test[num_cols]),
    columns=num_cols,
    index=X_test.index
)

cat_imputer = SimpleImputer(strategy='most_frequent')
X_train_cat_imp = pd.DataFrame(
    cat_imputer.fit_transform(X_train[cat_cols]),
    columns=cat_cols,
    index=X_train.index
)
X_test_cat_imp = pd.DataFrame(
    cat_imputer.transform(X_test[cat_cols]),
    columns=cat_cols,
    index=X_test.index
)

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_train_cat_enc = pd.DataFrame(
    encoder.fit_transform(X_train_cat_imp),
    columns=encoder.get_feature_names_out(cat_cols),
    index=X_train.index
)
X_test_cat_enc = pd.DataFrame(
    encoder.transform(X_test_cat_imp),
    columns=encoder.get_feature_names_out(cat_cols),
    index=X_test.index
)

X_train_final = pd.concat([X_train_num, X_train_cat_enc], axis=1)
X_test_final = pd.concat([X_test_num, X_test_cat_enc], axis=1)

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_final, y_train)

y_pred_dt = dt_model.predict(X_test_final)
y_prob_dt = dt_model.predict_proba(X_test_final)[:, 1]

print("Confusion Matrix (Decision Tree):")
print(confusion_matrix(y_test, y_pred_dt))

print("\nClassification Report (Decision Tree):")
print(classification_report(y_test, y_pred_dt))

auc_dt = roc_auc_score(y_test, y_prob_dt)
print(f"ROC AUC Score: {auc_dt:.4f}")

fpr, tpr, thresholds = roc_curve(y_test, y_prob_dt)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"Decision Tree (AUC = {auc_dt:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("dt_roc_curve.png")
plt.close()

importances = dt_model.feature_importances_
feature_names = X_train_final.columns
feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x='Importance', y='Feature', data=feat_imp_df)
plt.title("Feature Importances")
plt.tight_layout()
plt.savefig("dt_feature_importance.png")
plt.close()

plt.figure(figsize=(16, 10))
plot_tree(
    dt_model,
    feature_names=feature_names,
    class_names=['Not Survived', 'Survived'],
    filled=True,
    max_depth=3,
    fontsize=10
)
plt.title("Decision Tree Visualization")
plt.tight_layout()
plt.savefig("decision_tree_plot.png")
plt.close()

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_final, y_train)
y_pred_lr = lr_model.predict(X_test_final)

print("\nModel Comparison:")
print(f"Decision Tree - Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}, Precision: {precision_score(y_test, y_pred_dt):.4f}, Recall: {recall_score(y_test, y_pred_dt):.4f}, F1: {f1_score(y_test, y_pred_dt):.4f}")
print(f"Logistic Reg  - Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}, Precision: {precision_score(y_test, y_pred_lr):.4f}, Recall: {recall_score(y_test, y_pred_lr):.4f}, F1: {f1_score(y_test, y_pred_lr):.4f}")
