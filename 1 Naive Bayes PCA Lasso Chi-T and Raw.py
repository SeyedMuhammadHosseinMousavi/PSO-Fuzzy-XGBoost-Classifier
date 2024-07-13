%reset -f
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel, chi2, SelectKBest
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

import time
# Record the start time
start_time = time.time()
import warnings
# Suppress warnings
warnings.filterwarnings("ignore")

# Load data from CSV
file_path = 'tr.csv'   
data_df = pd.read_csv(file_path)
features = data_df.iloc[:, :-1]
labels = data_df.iloc[:, -1]
labels -= 1  # Adjusting labels

# Normalize features for PCA and Lasso
scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)

# Min-Max Scaling for Chi-squared test
min_max_scaler = MinMaxScaler()
features_min_max = min_max_scaler.fit_transform(features)

#  train-test split with varying random states
def split_data(features, labels, state):
    return train_test_split(features, labels, test_size=0.3, random_state=state)

#  train and test Gaussian Naive Bayes
def train_and_test(X_train, X_test, y_train, y_test):
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

# Multiple runs and average results
n_runs = 5
accuracies = {'PCA': [], 'Lasso': [], 'Chi2': [], 'Raw': []}

for i in range(n_runs):
    # Splitting data with varying random states to introduce variability
    X_train, X_test, y_train, y_test = split_data(features_normalized, labels, i)
    X_train_min_max, X_test_min_max, y_train_min_max, y_test_min_max = split_data(features_min_max, labels, i)
    
    # PCA for dimensionality reduction
    pca = PCA(n_components=0.95)  # Retain 95% of variance
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    # Lasso for feature selection
    from sklearn.linear_model import LassoCV
    from sklearn.feature_selection import SelectFromModel
    import numpy as np
    # Fit the LassoCV model to get the best alpha and coefficients
    lasso = LassoCV(cv=5).fit(X_train, y_train)
    # Get the coefficients and sort them by their absolute values in descending order
    coefs = np.abs(lasso.coef_)
    indices = np.argsort(coefs)[::-1][:10]  # Get indices of the top 10 coefficients
    # Select the top 10 features based on these indices
    X_train_lasso = X_train[:, indices]
    X_test_lasso = X_test[:, indices]
    
    # Chi-squared for feature selection
    chi2_selector = SelectKBest(chi2, k=10)  # Select top 10 features
    X_train_chi2 = chi2_selector.fit_transform(X_train_min_max, y_train_min_max)
    X_test_chi2 = chi2_selector.transform(X_test_min_max)
    
    # Evaluating models
    accuracies['PCA'].append(train_and_test(X_train_pca, X_test_pca, y_train, y_test))
    accuracies['Lasso'].append(train_and_test(X_train_lasso, X_test_lasso, y_train, y_test))
    accuracies['Chi2'].append(train_and_test(X_train_chi2, X_test_chi2, y_train_min_max, y_test_min_max))
    accuracies['Raw'].append(train_and_test(X_train, X_test, y_train, y_test))
    print(f"Processing...")

# Calculate mean and standard deviation of accuracies
for method, acc_list in accuracies.items():
    print(f"{method} - Mean Accuracy: {np.mean(acc_list):.4f}, Std Dev: {np.std(acc_list):.4f}")

# Print the runtime
end_time = time.time()
runtime_seconds = end_time - start_time
runtime_minutes = runtime_seconds / 60
print(f"\nTotal Runtime: {runtime_seconds:.2f} seconds ({runtime_minutes:.2f} minutes)")
