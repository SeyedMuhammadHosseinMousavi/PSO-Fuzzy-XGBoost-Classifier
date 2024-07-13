# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 00:16:07 2024

@author: S.M.H Mousavi
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import matplotlib.pyplot as plt
from pyswarm import pso

# Load data
data_df = pd.read_csv('tr.csv')
features = data_df.iloc[:, :-1]
labels = data_df.iloc[:, -1] - 1

# Normalize features
scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)
features_normalized_df = pd.DataFrame(features_normalized, columns=features.columns)

# Since Chi-Squared Test requires non-negative features, ensure all values are positive
# This can be done by scaling features to a non-negative range or by using absolute values
features_positive = np.abs(features_normalized)  # Using absolute values here

# Select a specific number of features using Chi-Squared Test
num_features = 20  # Set the number of features you want to retain
chi2_selector = SelectKBest(chi2, k=num_features)
features_kbest = chi2_selector.fit_transform(features_positive, labels)
features_kbest_df = pd.DataFrame(features_kbest, columns=[features.columns[i] for i in chi2_selector.get_support(indices=True)])

# Fuzzify features
def fuzzify_features(features_df):
    for column in features_df.columns:
        low = features_df[column].quantile(0.33)
        high = features_df[column].quantile(0.67)
        features_df[column] = features_df[column].apply(lambda x: 0 if x < low else (1 if x < high else 2))
    return features_df

features_fuzzified_df = fuzzify_features(features_kbest_df.copy())

# Split data
X_train, X_test, y_train, y_test = train_test_split(features_fuzzified_df, labels, test_size=0.3, random_state=42)

# Define the PSO objective function for XGBoost
best_accuracy = 0
def xgb_objective(params):
    global best_accuracy
    max_depth, learning_rate = int(params[0]), params[1]
    model = xgb.XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, num_class=4, objective='multi:softmax', verbosity=1, seed=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    loss = 1 - accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        print(f"New best parameters: max_depth={max_depth}, learning_rate={learning_rate}, accuracy={accuracy:.4f}")
    return loss

# PSO bounds
lb = [3, 0.01]
ub = [10, 0.3]

# Perform PSO
xopt, fopt = pso(xgb_objective, lb, ub, swarmsize=50, maxiter=200)

# Train the model with optimized parameters
model_opt = xgb.XGBClassifier(max_depth=int(xopt[0]), learning_rate=xopt[0.1], num_class=4, objective='multi:softmax', verbosity=1, seed=42)
model_opt.fit(X_train, y_train)
y_pred_opt = model_opt.predict(X_test)

# Evaluate and print optimized accuracy
accuracy_opt = accuracy_score(y_test, y_pred_opt)
print(f'Optimized Test Accuracy: {accuracy_opt:.4f}')
