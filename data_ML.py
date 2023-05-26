# This scripts enables us to make predictions using ML algorithms

###
# Packages
###
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# Cross validation, standardization & matrix confusions
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Random Forest
from sklearn.ensemble import RandomForestClassifier

# Gradient Boosting
from xgboost import XGBClassifier

###
# Packages Options
###
pd.options.mode.chained_assignment = None

###
# INPUTS
###
season = '2020-2021'
optim_mode = False

###
# DL of cleaned datas
###
data_dir = '%s%s%s%s' % (os.getcwd(), '\\Datas\\', season, '\\df_shots_cleaned.csv')
df_cleaned = pd.read_csv(data_dir)

###
# get rid of some columns
###
columns_to_delete = ['home_score', 'away_score', 'player_away']
df_cleaned = df_cleaned.drop(columns_to_delete, axis=1)

###
# conversion of categorical values in dummies variables
###
df_cleaned_ML = pd.get_dummies(df_cleaned)

# We train test datas
y = df_cleaned_ML['result']
X = df_cleaned_ML.drop('result', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

###
# Correlation matrice
###
corr_df = df_cleaned.select_dtypes(include=['float64', 'int64']).corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr_df, dtype=bool))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(corr_df, mask=mask, cmap=cmap, square=False, linewidths=1,
            cbar_kws={'label': 'Correlation'}, annot=False, ax=ax)
ax.tick_params(axis='x', labelrotation=45)

###
# I - Random Forest
###

# Random Forest Optimization
param_grid = {
    'n_estimators': [50, 100, 200, 500, 1000],
    'max_depth': np.arange(3, 10),
    'max_features': ['sqrt', 'log2', None],
    'min_samples_split': np.arange(2, 7),
    'criterion': ['gini', 'entropy']
}

# RF classifier definition
if optim_mode:
    rfc = RandomizedSearchCV(RandomForestClassifier(), param_grid, cv=5, n_jobs=-1, random_state=42)
    search_rfc = rfc.fit(X_train, y_train)
    pred_rfc = search_rfc.best_estimator_.predict(X_test)
    print("\n The best parameters across ALL searched params:\n", search_rfc.best_params_)
else:
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    pred_rfc = rfc.predict(X_test)

# Predictions and results of accuracy and confusion matrix
accur_rfc = accuracy_score(y_test, pred_rfc)
print(f'Accuracy Score: {accur_rfc:.2f}')

# Confusion Matrix and scoring
conf_mat_rfc = confusion_matrix(y_test, pred_rfc)
print(conf_mat_rfc)

tn_rfc = conf_mat_rfc[0][0]
fn_rfc = conf_mat_rfc[0][1]
fp_rfc = conf_mat_rfc[1][0]
tp_rfc = conf_mat_rfc[1][1]

precision_rfc = round(tp_rfc / (tp_rfc + fp_rfc), 2)
recall_rfc = round(tp_rfc / (tp_rfc + fn_rfc), 2)
f1_rfc = round(2 * tp_rfc / (2 * tp_rfc + fp_rfc + fn_rfc), 2)

###
# II - XG Boost
###

# XGC Optimization
param_grid_xgc = {
    'n_estimators': [50, 100, 200, 500, 1000],
    'max_depth': np.arange(3, 10),
    'learning_rate': [0.1, 0.01, 0.05, 0.001]
}

# XGC classifier definition
if optim_mode:
    xgc = RandomizedSearchCV(XGBClassifier(), param_grid_xgc, cv=5, n_jobs=-1, random_state=42)
    search_xgc = xgc.fit(X_train, y_train)
    pred_xgc = search_xgc.best_estimator_.predict(X_test)
    print("\n The best parameters across ALL searched params:\n", search_xgc.best_params_)

else:
    xgc = XGBClassifier()
    xgc.fit(X_train, y_train)
    pred_xgc = xgc.predict(X_test)

# Print precision
accur_xgc = accuracy_score(y_test, pred_xgc)
print(f'Accuracy Score: {accur_xgc:.2f}')

# Confusion Matrix and scores
conf_mat_xgc = confusion_matrix(y_test, pred_xgc)
print(conf_mat_xgc)

tn_xgc = conf_mat_xgc[0][0]
fn_xgc = conf_mat_xgc[0][1]
fp_xgc = conf_mat_xgc[1][0]
tp_xgc = conf_mat_xgc[1][1]

precision_xgc = round(tp_xgc / (tp_xgc + fp_xgc), 2)
recall_xgc = round(tp_xgc / (tp_xgc + fn_xgc), 2)
f1_xgc = round(2 * tp_xgc / (2 * tp_xgc + fp_xgc + fn_xgc), 2)

###
# III - Results comparison
###
df_result = pd.DataFrame({'Model': ['RF', 'RF', 'RF', 'RF', 'XGB', 'XGB', 'XGB', 'XGB'],
                          'Score': ['Accuracy', 'Precision', 'Recall', 'F1', 'Accuracy','Precision', 'Recall', 'F1'],
                          'Result': [accur_rfc, precision_rfc, recall_rfc, f1_rfc, accur_xgc, precision_xgc, recall_xgc, f1_xgc]
                          })

fig, ax = plt.subplots(figsize=(15, 8))
sns.barplot(data=df_result, x='Score', y='Result', hue='Model', palette=sns.set_palette('Set1', 2))
