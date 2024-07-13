import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
from pyswarm import pso

# Function to fuzzify features
def fuzzify_features(features_df):
    for column in features_df.columns:
        low_threshold = features_df[column].quantile(0.33)
        high_threshold = features_df[column].quantile(0.67)
        features_df[column] = features_df[column].apply(
            lambda x: 0 if x < low_threshold else (1 if x < high_threshold else 2)
        )
    return features_df

# Load data
data_df = pd.read_csv('tr.csv')  # Adjust the path as needed
features = data_df.iloc[:, :-1]
labels = data_df.iloc[:, -1] - 1  # Adjust labels from [1, 2, 3, 4] to [0, 1, 2, 3]

# Preprocess features
scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)
features_fuzzified = fuzzify_features(pd.DataFrame(features_normalized, columns=features.columns))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features_fuzzified, labels, test_size=0.3, random_state=42)

# Global variable to store the best score
global_best_score = float('inf')
global_best_params = None

# Objective function to minimize
def xgb_objective(params):
    global global_best_score, global_best_params
    max_depth, learning_rate = int(params[0]), params[1]
    model = xgb.XGBClassifier(
        max_depth=max_depth,
        learning_rate=learning_rate,
        num_class=4,
        objective='multi:softmax',
        verbosity=1,
        seed=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    loss = 1 - accuracy

    # Update global best and log
    if loss < global_best_score:
        global_best_score = loss
        global_best_params = params
        print(f"New best parameters found: max_depth={max_depth}, learning_rate={learning_rate}, accuracy={accuracy}")

    return loss

# Define bounds for max_depth and learning_rate
lb = [3, 0.01]
ub = [10, 0.3]

# Perform PSO
xopt, fopt = pso(xgb_objective, lb, ub, swarmsize=8, maxiter=100)

# Train the model with optimal hyperparameters
model_opt = xgb.XGBClassifier(
    max_depth=int(xopt[0]),
    learning_rate=xopt[1],
    num_class=4,
    objective='multi:softmax',
    verbosity=1,
    seed=42
)
model_opt.fit(X_train, y_train)
y_pred_opt = model_opt.predict(X_test)

# Evaluate and print optimized accuracy
accuracy_opt = accuracy_score(y_test, y_pred_opt)
print(f'Optimized Test Accuracy: {accuracy_opt:.4f}')
