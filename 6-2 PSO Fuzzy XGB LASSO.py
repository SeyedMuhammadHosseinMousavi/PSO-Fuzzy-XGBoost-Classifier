# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 00:18:56 2024

@author: S.M.H Mousavi
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
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

# Desired number of features
desired_features = 10

# Finding the right alpha for the desired number of features
alphas = np.logspace(-4, 0.5, 300)  # Generates 300 values between 10^-4 and 10^0.5
selected_features_mask = np.zeros(len(features.columns), dtype=bool)
for alpha in alphas:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(features_normalized_df, labels)
    selected_features_mask = np.sum(lasso.coef_ != 0) == desired_features
    if selected_features_mask:
        break

# Check if we have the correct number of features; if not, adjust the range and density of alphas
if not selected_features_mask:
    print("Could not find the exact number of desired features. Try different alpha range or density.")
else:
    selected_features_df = features_normalized_df.loc[:, lasso.coef_ != 0]

# Fuzzify features
def fuzzify_features(features_df):
    for column in features_df.columns:
        low = features_df[column].quantile(0.33)
        high = features_df[column].quantile(0.67)
        features_df[column] = features_df[column].apply(lambda x: 0 if x < low else (1 if x < high else 2))
    return features_df

features_fuzzified_df = fuzzify_features(selected_features_df.copy())

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
xopt, fopt = pso(xgb_objective, lb, ub, swarmsize=5, maxiter=30)

# Train the model with optimized parameters
model_opt = xgb.XGBClassifier(max_depth=int(xopt[0]), learning_rate=xopt[1], num_class=4, objective='multi:softmax', verbosity=1, seed=42)
model_opt.fit(X_train, y_train)
y_pred_opt = model_opt.predict(X_test)

# Evaluate and print optimized accuracy
accuracy_opt = accuracy_score(y_test, y_pred_opt)
print(f'Optimized Test Accuracy: {accuracy_opt:.4f}')
