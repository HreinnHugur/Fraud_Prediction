# Import necessary libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import joblib
import warnings


warnings.filterwarnings("ignore")


dataset_path = "creditcard_2023.csv"
df = pd.read_csv(dataset_path)
df.drop('id', axis=1, inplace=True)

print(f"Number of entries: {len(df)}")

scaler = RobustScaler()
df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df.drop(['Amount'], axis=1, inplace=True)
df.insert(1, 'scaled_amount', df.pop('scaled_amount'))


joblib.dump(scaler, 'scaler.pkl')
print("Scaler saved as scaler.pkl")


X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


classifiers = {
    "LogisticRegression": LogisticRegression(class_weight={0:5, 1:2}, solver='saga'),

    "DecisionTree": DecisionTreeClassifier(max_depth=5, min_samples_split=50, min_samples_leaf=20),

    "RandomForest": RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_leaf=10),

   "NeuralNetwork": MLPClassifier(max_iter=300, alpha=0.0001, hidden_layer_sizes=(100, 50), random_state=42),

   "GradientBoosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3),

    "XGBoost": xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3),

    "SVC": SVC(probability=True, class_weight="balanced",max_iter=300),

}


for name, clf in classifiers.items():
    print(f"\nTraining {name}...")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)  # Output as dictionary for detailed metrics
    report_df = pd.DataFrame(report).transpose()

    print(f"\nFull Classification Report for {name}:")
    print(report_df)

    print(f"\nConfusion Matrix for {name}:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_test)
    auc = roc_auc_score(y_test, y_prob)
    print(f"\nROC-AUC Score for {name}: {auc:.4f}")

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for {name}: {accuracy:.4f}")

    joblib.dump(clf, f'{name}_model.pkl')
    print(f"{name} model saved as {name}_model.pkl")
