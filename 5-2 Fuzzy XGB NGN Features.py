import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import matplotlib.pyplot as plt

class NeuralGas:
    def __init__(self, n_units=3, max_iter=1000, eta_start=0.5, eta_end=0.01, lambda_start=30, lambda_end=0.1):
        self.n_units = n_units
        self.max_iter = max_iter
        self.eta_start = eta_start
        self.eta_end = eta_end
        self.lambda_start = lambda_start
        self.lambda_end = lambda_end
        self.units = None
        self.feature_importance = None

    def _update_learning_rate(self, i):
        return self.eta_start * (self.eta_end / self.eta_start) ** (i / self.max_iter)

    def _update_neighborhood_range(self, i):
        return self.lambda_start * (self.lambda_end / self.lambda_start) ** (i / self.max_iter)

    def train(self, data):
        n_samples, n_features = data.shape
        self.units = np.random.rand(self.n_units, n_features)
        self.feature_importance = np.zeros(n_features)

        for i in range(self.max_iter):
            eta = self._update_learning_rate(i)
            lambd = self._update_neighborhood_range(i)

            indices = np.random.permutation(n_samples)
            for index in indices:
                x = data[index]
                dists = np.linalg.norm(self.units - x, axis=1)
                ranking = np.argsort(dists)
                
                for rank, idx in enumerate(ranking):
                    influence = np.exp(-rank / lambd)
                    self.feature_importance += eta * influence * np.abs(x - self.units[idx])
                    self.units[idx] += eta * influence * (x - self.units[idx])
            
            # Print the iteration number
            print(f"NGN Training Iteration: {i+1}/{self.max_iter}")

    def get_feature_importance(self):
        return self.feature_importance / np.sum(self.feature_importance)

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
        # Assuming low and high thresholds (customize as necessary)
        low_threshold = features_df[column].quantile(0.33)
        high_threshold = features_df[column].quantile(0.67)
        features_df[column] = features_df[column].apply(fuzzy_membership, args=(low_threshold, high_threshold))
    
    return features_df

# Load data from CSV
file_path = 'tr.csv'  # Make sure to replace this with the actual path to your CSV file
data_df = pd.read_csv(file_path)

# Assuming the last column is the label
features = data_df.iloc[:, :-1]
labels = data_df.iloc[:, -1]

# Shift labels from [1, 2, 3, 4] to [0, 1, 2, 3]
labels -= 1

# Normalize features
scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)
features_normalized_df = pd.DataFrame(features_normalized, columns=features.columns)

# Initialize and train Neural Gas
ng = NeuralGas(n_units=9, max_iter=100)  # Adjust n_units and max_iter as needed
ng.train(features_normalized)

# Get and print feature importance
feature_importance = ng.get_feature_importance()
print("Feature Importance from Neural Gas:")
print(feature_importance)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.bar(features.columns, feature_importance)
plt.title('Feature Importance using Neural Gas')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.show()

# Select top features based on importance
top_features_indices = np.argsort(feature_importance)[-20:]  # Adjust the number of top features as needed
features_top = features_normalized_df.iloc[:, top_features_indices]

# Fuzzify features
features_fuzzified_df = fuzzify_features(features_top.copy())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_fuzzified_df, labels, test_size=0.3, random_state=42)

# Initialize XGBoost classifier
model = xgb.XGBClassifier(objective='multi:softmax', num_class=4, verbosity=1, seed=42)

# Train the model
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy with Fuzzy NGN selected features: {accuracy:.4f}')
