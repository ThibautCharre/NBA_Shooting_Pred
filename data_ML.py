# This scripts enables us to make predictions using ML algorithms

###
# Packages
###
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

###
# Packages Options
###
pd.options.mode.chained_assignment = None


###
# INPUTS
###
season = '2020-2021'

###
# DL of cleaned datas
###
data_dir = '%s%s%s%s' % (os.getcwd(), '\\Datas\\', season, '\\df_shots_cleaned_SG.csv')
df_cleaned = pd.read_csv(data_dir)

###
# get rid of some columns
###
columns_to_delete = ['home_score', 'away_score']
df_cleaned = df_cleaned.drop(columns_to_delete, axis=1)

###
# conversion of categorical values in dummies variables
###
df_cleaned_ML = pd.get_dummies(df_cleaned)

###
# Definition of X and Y and training of datas
###
y = df_cleaned_ML['result']
X = df_cleaned_ML.drop('result', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

###
# PCA using scaled X
###
scaler = StandardScaler()
scaler.fit(X)
scaled_X = scaler.transform(X)

PCA_all_features = PCA(scaled_X.shape[1], random_state=42)
PCA_all_features.fit(scaled_X)
X_PCA_all = PCA_all_features.transform(scaled_X)
PCA_all_features.explained_variance_ratio_*100

###
# Random Forest
###
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)
print(classification_report(y_test, pred_rfc))
res = confusion_matrix(y_test, pred_rfc)
print(confusion_matrix(y_test, pred_rfc))
test = ConfusionMatrixDisplay(confusion_matrix=res, display_labels=rfc.classes_)
test.plot()

plt.barh(X_train.columns, rfc.feature_importances_)
plt.show()
plt.xlabel("Random Forest Feature Importance")

###
# SVM
##
clf = svm.SVC()
clf.fit(X_train, y_train)
predict_clf = clf.predict(X_test)
print(classification_report(y_test, predict_clf))
print(confusion_matrix(y_test, predict_clf))