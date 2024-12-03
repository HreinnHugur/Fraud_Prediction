import os
import sys
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from urllib.parse import unquote, urlparse
from urllib.error import HTTPError
from zipfile import ZipFile
import tarfile
import shutil
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.patches as mpatches
import time


CHUNK_SIZE = 40960
DATA_SOURCE_MAPPING = 'D:\\modelFraudDetection\\creditcard.csv.zip'
INPUT_PATH = '/DATA/input'
WORKING_PATH = '/DATA/working'
KAGGLE_SYMLINK = 'DATA'

for root, dirs, files in os.walk(INPUT_PATH):
    for file in files:
        os.remove(os.path.join(root, file))

os.makedirs(INPUT_PATH, exist_ok=True)
os.makedirs(WORKING_PATH, exist_ok=True)

try:
    os.symlink(INPUT_PATH, os.path.join("..", 'input'), target_is_directory=True)
except FileExistsError:
    pass

try:
    os.symlink(WORKING_PATH, os.path.join("..", 'working'), target_is_directory=True)
except FileExistsError:
    pass

for data_source_mapping in DATA_SOURCE_MAPPING.split(','):
    directory = os.path.splitext(os.path.basename(data_source_mapping))[0]
    source_file = os.path.join(INPUT_PATH, data_source_mapping)
    destination_path = os.path.join(INPUT_PATH, directory)

    try:
        if data_source_mapping.endswith('.zip'):
            with ZipFile(source_file, 'r') as zfile:
                zfile.extractall(destination_path)
        elif data_source_mapping.endswith('.tar.gz'):
            with tarfile.open(source_file, 'r:gz') as tar:
                tar.extractall(destination_path)
        else:
            print(f'Unsupported file format for: {data_source_mapping}')
            continue

        print(f'Extracted {data_source_mapping} to {destination_path}')
    except Exception as e:
        print(f'Failed to extract {data_source_mapping}: {e}')

print('Data source import complete.')

import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/DATA/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import collections

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from collections import Counter
from sklearn.model_selection import  KFold, StratifiedKFold
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("creditcard.csv")
df.head()
print(df.columns)

df.describe()

df.isnull().sum().max()

df.columns

print('Legal', round(df['Class'].value_counts()[0] / len(df) *100,2), '%of the dataset')
print('Fraud', round(df['Class'].value_counts()[1] / len(df) *100,2), '%of the dataset')

colors = ["#03ff46", "#ff030f"]

sns.countplot(x='Class', data=df, palette=colors)
plt.title('Class Distribution \n (0: Legal || 1:Fraud', fontsize=14)


fig, ax = plt.subplots(1,2,figsize=(20,10))

amount_val = df['Amount'].values
time_val = df['Time'].values

sns.distplot(amount_val, ax=ax[0], color='r')
ax[0].set_title('Transaction Amount', fontsize=14)
ax[0].set_xlim([min(amount_val), max(amount_val)])

sns.distplot(time_val, ax=ax[1], color='b')
ax[1].set_title('Transaction Time', fontsize=14)
ax[1].set_xlim([min(time_val), max(time_val)])

plt.show()


from sklearn.preprocessing import StandardScaler, RobustScaler

std_scaler = StandardScaler()
rob_scaler = RobustScaler()

df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))

df.drop(['Time', 'Amount'], axis=1, inplace=True)
df

scaled_amount = df['scaled_amount']
scaled_time = df['scaled_time']

df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
df.insert(0, 'scaled_amount', scaled_amount)
df.insert(1, 'scaled_time', scaled_time)

df.head()

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

print('No Frauds', round(df['Class'].value_counts()[0]/len(df) * 100,2), '% of the dataset')
print('Frauds', round(df['Class'].value_counts()[1]/len(df) * 100,2), '% of the dataset')

X = df.drop('Class', axis=1)
y = df['Class']

sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

for train_index, test_index in sss.split(X,y):
    print("Train:", train_index, "Test:", test_index)
    original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]
    original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]

original_Xtrain = original_Xtrain.values
original_Xtest = original_Xtest.values
original_ytrain = original_ytrain.values
original_ytest = original_ytest.values

train_unique_label, train_counts_label = np.unique(original_ytrain, return_counts=True)
test_unique_label, test_counts_label = np.unique(original_ytest, return_counts=True)

print('-'*100)

print('Label Distributions: \n')
print(train_counts_label / len(original_ytrain))
print(test_counts_label / len(original_ytest))

df = df.sample(frac=1)
fraud_df = df.loc[df['Class'] == 1]
non_fraud_df  = df.loc[df['Class'] == 0][:492]
normal_distributed_df= pd.concat([fraud_df, non_fraud_df])
new_df = normal_distributed_df.sample(frac=1, random_state=42)
new_df.head()

print('Distribution of the Classes in the subsample dataset')
print(new_df['Class'].value_counts()/len(new_df))

sns.countplot(x='Class', data=new_df, palette=colors)
plt.title('Equally Distributed Classes', fontsize=14)
plt.show()

F, (ax1, ax2) = plt.subplots(2,1, figsize=(24,20))

corr = df.corr()
sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax1)
ax1.set_title("Imbalanced Colrrelation Matrix \n (don't use for reference)", fontsize=14)

sub_sample_corr = new_df.corr()
sns.heatmap(sub_sample_corr, cmap='coolwarm_r', annot_kws={'size':20},ax=ax2 )
ax2.set_title('SubSample Correlation Matrix \n (use for reference)', fontsize=14)
plt.show()

f, axes = plt.subplots(ncols=4, figsize = (20,4))
sns.boxplot(x="Class", y="V17", data=new_df, palette=colors, ax=axes[0])
axes[0].set_title('V17 vs Class Negative Correlation')

sns.boxplot(x="Class", y="V14", data=new_df, palette=colors, ax=axes[1])
axes[1].set_title('V14 vs Class Negative Correlation')

sns.boxplot(x="Class", y="V12", data=new_df, palette=colors, ax=axes[2])
axes[2].set_title('V12 vs Class Negative Correlation')

sns.boxplot(x="Class", y="V10", data=new_df, palette=colors, ax=axes[3])
axes[3].set_title('V10 vs Class Negative Correlation')

plt.show()

f, axes = plt.subplots(ncols=4, figsize = (20,4))
sns.boxplot(x="Class", y="V11", data=new_df, palette=colors, ax=axes[0])
axes[0].set_title('V11 vs Class Negative Correlation')

sns.boxplot(x="Class", y="V4", data=new_df, palette=colors, ax=axes[1])
axes[1].set_title('V4 vs Class Negative Correlation')

sns.boxplot(x="Class", y="V1", data=new_df, palette=colors, ax=axes[2])
axes[2].set_title('V1 vs Class Negative Correlation')

sns.boxplot(x="Class", y="V19", data=new_df, palette=colors, ax=axes[3])
axes[3].set_title('V19 vs Class Negative Correlation')

plt.show()

from scipy.stats import norm

f, (ax1, ax2, ax3) = plt.subplots(1,3,figsize = (20,6))

v14_fraud_dist = new_df['V14'].loc[new_df['Class'] ==1 ].values
sns.distplot(v14_fraud_dist, ax=ax1, fit=norm, color='#FB8861')
ax1.set_title('V14 Distribution \n (Fraud Transactions)', fontsize=14)

v12_fraud_dist = new_df['V12'].loc[new_df['Class'] ==1 ].values
sns.distplot(v12_fraud_dist, ax=ax2, fit=norm, color='#56F9BB')
ax2.set_title('V12 Distribution \n (Fraud Transactions)', fontsize=14)

v10_fraud_dist = new_df['V10'].loc[new_df['Class'] ==1 ].values
sns.distplot(v10_fraud_dist, ax=ax3, fit=norm, color='#C5B2F9')
ax3.set_title('V10 Distribution \n (Fraud Transactions)', fontsize=14)

v14_fraud = new_df['V14'].loc[new_df['Class'] == 1].values
q25, q75 = np.percentile(v14_fraud, 25), np.percentile(v14_fraud, 75)
print('Quartile 25 : {} | Quartile 75: {}'.format(q25, q75))
v14_iqr = q75 - q25
print('iqr: {}'.format(v14_iqr))

v14_cut_off = v14_iqr * 1.5
v14_lower, v14_upper  = q25 - v14_cut_off, q75 + v14_cut_off
print('Cut off: {}'.format(v14_cut_off))
print('V14 Lower: {}'.format(v14_lower))

outliers = [x for x in v14_fraud if x < v14_lower or x> v14_upper]
print('Feature V14 Outliers for Fraud Cases: {}'.format(len(outliers)))

print('V14 outliers:{}'.format(outliers))

new_df = new_df.drop(new_df[(new_df['V14'] > v14_upper) | (new_df['V14'] < v14_lower)].index)
print('----'*44)

V12_fraud = new_df['V12'].loc[new_df['Class'] == 1].values
q25, q75 = np.percentile(V12_fraud, 25), np.percentile(V12_fraud, 75)
print('Quartile 25 : {} | Quartile 75: {}'.format(q25, q75))
V12_iqr = q75 - q25
print('iqr: {}'.format(V12_iqr))

V12_cut_off = V12_iqr * 1.5
V12_lower, V12_upper  = q25 - V12_cut_off, q75 + V12_cut_off
print('Cut off: {}'.format(V12_cut_off))
print('V12 Lower: {}'.format(V12_lower))

outliers = [x for x in V12_fraud if x < V12_lower or x> V12_upper]
print('Feature V12 Outliers for Fraud Cases: {}'.format(len(outliers)))

print('V12 outliers:{}'.format(outliers))

new_df = new_df.drop(new_df[(new_df['V12'] > V12_upper) | (new_df['V12'] < V12_lower)].index)
print('Number of Instances after outliers removal: {}'.format(len(new_df)))
print('----'*44)

V10_fraud = new_df['V10'].loc[new_df['Class'] == 1].values
q25, q75 = np.percentile(V10_fraud, 25), np.percentile(V10_fraud, 75)
print('Quartile 25 : {} | Quartile 75: {}'.format(q25, q75))
V10_iqr = q75 - q25
print('iqr: {}'.format(V10_iqr))

V10_cut_off = V10_iqr * 1.5
V10_lower, V10_upper  = q25 - V10_cut_off, q75 + V10_cut_off
print('Cut off: {}'.format(V10_cut_off))
print('V10 Lower: {}'.format(V10_lower))

outliers = [x for x in V10_fraud if x < V10_lower or x> V10_upper]
print('Feature V10 Outliers for Fraud Cases: {}'.format(len(outliers)))

print('V10 outliers:{}'.format(outliers))

new_df = new_df.drop(new_df[(new_df['V10'] > V10_upper) | (new_df['V10'] < V10_lower)].index)
print('Number of Instances after outliers removal: {}'.format(len(new_df)))
print('----'*44)

f, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(20,6))
colors = ['#B3F9C5', '#f9c5b3']

sns.boxplot(x="Class", y="V14", data=new_df, ax=ax1, palette=colors)
ax1.set_title("V14 Feature \n Reduction of outliers", fontsize=14)
ax1.annotate('Fewer extreme \n outliers ', xy=(0.98, -17.5), xytext=(0, -12),
            arrowprops=dict(facecolor='black'), fontsize=14)

sns.boxplot(x="Class", y="V12", data=new_df, ax=ax2, palette=colors)
ax2.set_title("V12 Feature \n Reduction of outliers", fontsize=14)
ax2.annotate('Fewer extreme \n outliers ', xy=(0.98, -17.3), xytext=(0, -12),
            arrowprops=dict(facecolor='black'), fontsize=14)

sns.boxplot(x="Class", y="V10", data=new_df, ax=ax3, palette=colors)
ax3.set_title("V10 Feature \n Reduction of outliers", fontsize=14)
ax3.annotate('Fewer extreme \n outliers ', xy=(0.95, -16.5), xytext=(0, -12),
            arrowprops=dict(facecolor='black'), fontsize=14)

plt.show()


X = new_df.drop('Class', axis=1)
y = new_df['Class']

t0 = time.time()
X_reduced_tsne = TSNE(n_components=2, random_state=42).fit_transform(X.values)
t1 = time.time()
print("T-SNE took {:.2} s".format(t1-t0))

t0 = time.time()
X_reduced_pca = PCA(n_components=2, random_state=42).fit_transform(X.values)
t1 = time.time()
print("PCA took {:.2} s".format(t1-t0))

t0 = time.time()
X_reduced_svd = TruncatedSVD(n_components=2, algorithm='randomized', random_state=42).fit_transform(X.values)
t1 = time.time()
print("Truncated SVD took {:.2} s".format(t1-t0))

f, (ax1, ax2, ax3 ) = plt.subplots(1,3, figsize=(24,6))
f.suptitle('Clustering using Dimensionality Reduction', fontsize = 14)

blue_patch = mpatches.Patch(color='#0A0AFF', label='No Fraud')
red_patch = mpatches.Patch(color='#AF0000', label='Fraud')
color_num = 2

ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y==0), cmap= plt.cm.coolwarm, label='No Fraud', linewidths=2)
ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y==1), cmap= plt.cm.coolwarm, label='Fraud', linewidths=2)
ax1.set_title('t-SNE', fontsize=14)

ax1.grid(True)
ax1.legend(handles =[blue_patch, red_patch])

ax2.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(y==0), cmap= plt.cm.coolwarm, label='No Fraud', linewidths=2)
ax2.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(y==1), cmap=plt.cm.coolwarm, label='Fraud', linewidths=2)
ax2.set_title('PCA', fontsize=14)

ax2.grid(True)
ax2.legend(handles =[blue_patch, red_patch])

ax3.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1], c=(y==0), cmap= plt.cm.coolwarm, label='No Fraud', linewidths=2)
ax3.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1], c=(y==1), cmap=plt.cm.coolwarm, label='Fraud', linewidths=2)
ax3.set_title('Truncated SVD', fontsize=14)

ax3.grid(True)
ax3.legend(handles =[blue_patch, red_patch])

plt.show()

X = new_df.drop('Class', axis = 1)
y = new_df['Class']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=42)

X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

classifier = {
    "LogisticRegression": LogisticRegression(class_weight={0:5, 1:2}, solver='saga'),

    "DecisionTree": DecisionTreeClassifier(max_depth=5, min_samples_split=50, min_samples_leaf=20),

    "RandomForest": RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_leaf=10),

   "NeuralNetwork": MLPClassifier(max_iter=300, alpha=0.0001, hidden_layer_sizes=(100, 50), random_state=42),

   "GradientBoosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3),

    "XGBoost": xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3),

    "SVC": SVC(probability=True, class_weight="balanced",max_iter=300),
}


# Insert this code for evaluation
for model_name, model in classifier.items():
    print(f"\nEvaluating {model_name}...")
    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    auc = roc_auc_score(y_test, y_pred)
    print(f"\nROC-AUC Score: {auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

from sklearn.model_selection import cross_val_score

for key, classifier in classifier.items():
    classifier.fit(X_train, y_train)
    training_score = cross_val_score(classifier, X_train, y_train, cv=5)
    print("Classifiers: ", classifier.__class__.__name__, "Has a training score of ", round(training_score.mean(), 2)*100, "% accuracy score")

