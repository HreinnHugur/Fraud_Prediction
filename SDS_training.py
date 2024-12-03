import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import joblib
import warnings
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

df = pd.read_csv("fraud_data.csv")

class_counts = df['is_fraud'].value_counts()
class_percentages = df['is_fraud'].value_counts(normalize=True) * 100

print("Class Counts:\n", class_counts)
print("\nClass Percentages:\n", class_percentages)

df['is_fraud'] = df['is_fraud'].str.extract(r'(\d)').astype(int)

df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['hour'] = df['trans_date_trans_time'].dt.hour
df['day'] = df['trans_date_trans_time'].dt.day
df['month'] = df['trans_date_trans_time'].dt.month
df.drop(['trans_date_trans_time', 'dob', 'trans_num'], axis=1, inplace=True)

df = pd.get_dummies(df, columns=['merchant', 'category', 'city', 'state', 'job'], drop_first=True)

scaler = RobustScaler()
df['scaled_amt'] = scaler.fit_transform(df[['amt']])
df.drop(['amt'], axis=1, inplace=True)
df.insert(0, 'scaled_amt', df.pop('scaled_amt'))

joblib.dump(scaler, 'scaler.pkl')

X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

classifiers = {
    "LogisticRegression": LogisticRegression(class_weight="balanced", solver='liblinear'),
    "DecisionTree": DecisionTreeClassifier(max_depth=5, min_samples_split=50, min_samples_leaf=20),
    "RandomForest": RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_leaf=10),
    "NeuralNetwork": MLPClassifier(max_iter=300, alpha=0.0001, hidden_layer_sizes=(100, 50), random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3),
    "XGBoost": xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3),
    "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    "SVC": SVC(probability=True, class_weight="balanced"),
}

for name, clf in classifiers.items():
    print(f"\nTraining {name}...")

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(f"\nFull Classification Report for {name}:")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
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
