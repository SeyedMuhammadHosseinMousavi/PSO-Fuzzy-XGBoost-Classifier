import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel, chi2, SelectKBest
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb

def fuzzify_features(features_df):
    # Define membership functions for each feature
    def fuzzy_membership(val, low, high):
        if val < low:
            return 0  # Low
        elif low <= val < high:
            return 1  # Medium
        else:
            return 2  # High

# Apply fuzzification to each feature
    for column in features_df.columns:
        low_threshold = features_df[column].quantile(0.33)
        high_threshold = features_df[column].quantile(0.67)
        features_df[column] = features_df[column].apply(fuzzy_membership, args=(low_threshold, high_threshold))
    return features_df

# Load data from CSV
file_path = 'tr.csv'  # Update this with your file path
data_df = pd.read_csv(file_path)

# Assuming the last column is the label
features = data_df.iloc[:, :-1]
labels = data_df.iloc[:, -1]
labels -= 1  # Adjusting labels

# Normalize features for PCA and Lasso
scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)
features_normalized_df = pd.DataFrame(features_normalized, columns=features.columns)

# Min-Max Scaling for Chi-squared test
min_max_scaler = MinMaxScaler()
features_min_max = min_max_scaler.fit_transform(features)

# Fuzzify features
features_fuzzified_df = fuzzify_features(features_normalized_df.copy())

# Feature Selection using PCA, Lasso, and Chi-squared on normalized data
pca = PCA(n_components=0.95)
features_pca = pca.fit_transform(features_normalized_df)

lasso = LassoCV().fit(features_normalized_df, labels)
model_lasso = SelectFromModel(lasso, prefit=True)
features_lasso = model_lasso.transform(features_normalized_df)

chi2_selector = SelectKBest(chi2, k=10)  # Adjust k as needed
features_chi2 = chi2_selector.fit_transform(features_min_max, labels)

# Define the model
model = xgb.XGBClassifier(
    objective='multi:softmax',        # Objective function for multi-class classification
    num_class=4,                      # Number of classes
    verbosity=1,                      # Verbosity of printing messages
    seed=42,                          # Random seed for reproducibility
    n_estimators=100,                 # Number of gradient boosted trees. Equivalent to number of boosting rounds
    max_depth=6,                      # Maximum tree depth for base learners
    learning_rate=0.1,                # Boosting learning rate (also known as "eta")
    subsample=0.8,                    # Subsample ratio of the training instances
    colsample_bytree=0.8,             # Subsample ratio of columns when constructing each tree
    gamma=0,                          # Minimum loss reduction required to make a further partition on a leaf node of the tree
    reg_lambda=1,                     # L2 regularization term on weights (analogous to Ridge regression)
    reg_alpha=0,                      # L1 regularization term on weight (analogous to Lasso regression)
    scale_pos_weight=1,               # Balancing of positive and negative weights
    min_child_weight=1                # Minimum sum of instance weight (hessian) needed in a child
)
# Function to train and test model multiple times
def evaluate_model(X, y, model, runs=5):
    accuracies = []
    for i in range(runs):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
    return np.mean(accuracies), np.std(accuracies)

# Evaluate the model with different feature sets
accuracy_raw, std_raw = evaluate_model(features_normalized_df, labels, model)
accuracy_fuzzified, std_fuzzified = evaluate_model(features_fuzzified_df, labels, model)
accuracy_pca, std_pca = evaluate_model(features_pca, labels, model)
accuracy_lasso, std_lasso = evaluate_model(features_lasso, labels, model)
accuracy_chi2, std_chi2 = evaluate_model(features_chi2, labels, model)

# Print results
print(f'Accuracy with Fuzzified Features: {accuracy_fuzzified:.4f}, Std Dev: {std_fuzzified:.4f}')
print(f'Accuracy with PCA Features: {accuracy_pca:.4f}, Std Dev: {std_pca:.4f}')
print(f'Accuracy with Lasso Features: {accuracy_lasso:.4f}, Std Dev: {std_lasso:.4f}')
print(f'Accuracy with Chi2 Features: {accuracy_chi2:.4f}, Std Dev: {std_chi2:.4f}')
