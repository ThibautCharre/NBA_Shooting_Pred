# This scripts enables us to make predictions using ML algorithms

###
# Packages
###
import os
import pandas as pd
import numpy as np

# Cross validation, standardization & matrix confusions
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, accuracy_score

# Random Forest
from sklearn.ensemble import RandomForestClassifier

# Gradient Boosting
from xgboost import XGBClassifier

# Scaling
from sklearn.preprocessing import MinMaxScaler

# Neural network
from keras.models import Sequential
from keras.layers import Dense

# MatPlotLib
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
# I - Random Forest
###
# RF classifier definition
rfc = RandomForestClassifier(n_jobs=-1, random_state=42)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)

# Print precision
print(f'Accuracy Score: {100*accuracy_score(y_test, pred_rfc):.4f}')

# Report of confusion matrix
print(classification_report(y_test, pred_rfc))

# Reports of normal confusion matrix
rf_matrix_conf = confusion_matrix(y_test, pred_rfc)
rf_matrix_conf_plot = ConfusionMatrixDisplay(confusion_matrix=rf_matrix_conf, display_labels=rfc.classes_)
rf_matrix_conf_plot.plot()

###
# II - XG Boost
###
# RF classifier definition
xgc = XGBClassifier()
xgc.fit(X_train, y_train)
pred_xgc = xgc.predict(X_test)

# Print precision
print(f'Accuracy Score: {100*accuracy_score(y_test, pred_xgc):.4f}')

# Report of confusion matrix
print(classification_report(y_test, pred_xgc))

# Reports of normal confusion matrix
xg_matrix_conf = confusion_matrix(y_test, pred_xgc)
xg_matrix_conf_plot = ConfusionMatrixDisplay(confusion_matrix=xg_matrix_conf, display_labels=xgc.classes_)
xg_matrix_conf_plot.plot()


# Random Forest & XG Boosting Optimization
# param_grid = {
#     'n_estimators': [50, 100, 200, 500, 1000],
#     'max_depth': np.arange(3, 10),
#     'max_features': ['sqrt', 'log2', None],
#     'max_samples': np.linspace(0.5, 1.0, 6)
# }
# rfc2 = RandomizedSearchCV(RandomForestClassifier(n_jobs=-1), param_grid, cv=5, random_state=42, verbose=1)
# rfc2.fit(X_train, y_train)
#
# # Launch of optimized Random Forest
# rfc_optim = RandomForestClassifier(n_jobs=-1,
#                                    n_estimators=rfc2.best_params_['n_estimators'],
#                                    max_samples=rfc2.best_params_['max_samples'],
#                                    max_features=rfc2.best_params_['max_features'],
#                                    max_depth=rfc2.best_params_['max_depth'],
#                                    random_state=42
#                                    )
# rfc_optim.fit(X_train, y_train)
# pred_rfc_optim = rfc_optim.predict(X_test)

# Reports of optimal confusion matrix
# rf_optim_matrix_conf = confusion_matrix(y_test, pred_rfc_optim)
# rf_optim_matrix_conf_plot = ConfusionMatrixDisplay(confusion_matrix=rf_optim_matrix_conf,
# display_labels=rfc_optim.classes_)
# rf_optim_matrix_conf_plot.plot()

###
# III - Bonus : Neural Network (disgusting results)
###
###
# Scaling of datas using min and max
###
# df_cleaned_ML_scaled = df_cleaned_ML.copy()
#
# scaler = MinMaxScaler()
# df_cleaned_ML_scaled[['player_team_scorediff', 'play_length',
#                                     'shot_distance', 'player_season_points',
#                                     'player_game_points',
#                                     'shots_player_made', 'shots_player_total',
#                                     'FG_player', 'player_streak',
#                                     'assist_player_total', 'ratio_assist_player',
#                                     'Experience', 'Age']] = \
#     scaler.fit_transform(df_cleaned_ML_scaled[['player_team_scorediff', 'play_length',
#                                     'shot_distance', 'player_season_points',
#                                     'player_game_points',
#                                     'shots_player_made', 'shots_player_total',
#                                     'FG_player', 'player_streak',
#                                     'assist_player_total', 'ratio_assist_player',
#                                     'Experience', 'Age']])
#
# # We train new scaled datas
# X_scaled = df_cleaned_ML_scaled.drop('result', axis=1)
# y = df_cleaned_ML_scaled['result']
# X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
#
# # Neural Network creation
# model = Sequential()
# model.add(Dense(128, activation='relu', input_dim=X_scaled.shape[1]))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# # Training and prediction of the network
# model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test), epochs=10, batch_size=100)
# y_predicted_nn = model.predict(X_test) > 0.5
#
# # Confusion matrix
# mat_nn = confusion_matrix(y_test, y_predicted_nn)
# conf_matrix = ConfusionMatrixDisplay(confusion_matrix=mat_nn)
# conf_matrix.plot()
# plt.xlabel('Predicted label')
# plt.ylabel('Actual label')
