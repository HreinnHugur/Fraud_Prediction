import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, accuracy_score

model = joblib.load('SVC_model.pkl')
scaler = joblib.load('scaler.pkl')

new_data_path = "creditcard.csv"
df_new = pd.read_csv(new_data_path)

df_new['scaled_amount'] = scaler.transform(df_new['Amount'].values.reshape(-1, 1))
df_new.drop(['Amount'], axis=1, inplace=True)
df_new.insert(1, 'scaled_amount', df_new.pop('scaled_amount'))

required_columns = ['scaled_amount', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                    'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                    'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Class']

df_new = df_new[required_columns]

X_new = df_new.drop('Class', axis=1)
y_true = df_new['Class']

predictions = model.predict(X_new)
y_prob = model.predict_proba(X_new)[:, 1]

report = classification_report(y_true, predictions, output_dict=True)  # Use output_dict=True to capture the report
report_df = pd.DataFrame(report).transpose()

print(f"\nFull Classification Report:")
print(report_df)

cm = confusion_matrix(y_true, predictions)
print(f"\nConfusion Matrix:\n{cm}")

roc_auc = roc_auc_score(y_true, y_prob)
accuracy = accuracy_score(y_true, predictions)
print(f"ROC-AUC Score: {roc_auc:.4f}")
print(f"Accuracy Score: {accuracy:.4f}")

