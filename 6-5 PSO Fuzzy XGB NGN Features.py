import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import matplotlib.pyplot as plt
from pyswarm import pso

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
                    self.units[idx] += eta * influence * (x - self.units[idx])
            print(f"NGN Training Iteration: {i+1}/{self.max_iter}")

    def get_feature_importance(self):
        return self.feature_importance / np.sum(self.feature_importance)

# Load data
data_df = pd.read_csv('tr.csv')
features = data_df.iloc[:, :-1]
labels = data_df.iloc[:, -1] - 1

# Normalize features
scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)
features_normalized_df = pd.DataFrame(features_normalized, columns=features.columns)

# Initialize and train Neural Gas
ng = NeuralGas(n_units=9, max_iter=100)
ng.train(features_normalized)

# Get feature importance and select top features
feature_importance = ng.get_feature_importance()
top_features_indices = np.argsort(feature_importance)[-22:]
features_top = features_normalized_df.iloc[:, top_features_indices]

# Fuzzify selected features
def fuzzify_features(features_df):
    for column in features_df.columns:
        low = features_df[column].quantile(0.33)
        high = features_df[column].quantile(0.67)
        features_df[column] = features_df[column].apply(lambda x: 0 if x < low else (1 if x < high else 2))
    return features_df

features_fuzzified_df = fuzzify_features(features_top.copy())


# Split data
X_train, X_test, y_train, y_test = train_test_split(features_fuzzified_df, labels, test_size=0.3, random_state=42)

import matplotlib.pyplot as plt
import xgboost as xgb
from pyswarm import pso
from sklearn.metrics import accuracy_score

# List to store the best loss values at each iteration
best_cost_history = []

# Define the PSO objective function for XGBoost
def xgb_objective(params):
    global best_accuracy
    global best_cost_history
    max_depth, learning_rate = int(params[0]), params[1]
    model = xgb.XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, num_class=4, objective='multi:softmax', verbosity=1, seed=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    loss = 1 - accuracy
    if best_cost_history:
        # Ensure the best cost history list always contains the minimum loss seen so far
        best_cost_history.append(min(loss, best_cost_history[-1]))
    else:
        # Initialize the list with the first loss if empty
        best_cost_history.append(loss)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        print(f"New best parameters: max_depth={max_depth}, learning_rate={learning_rate}, accuracy={accuracy:.4f}")
    return loss

# Initialize the best accuracy
best_accuracy = 0

# PSO bounds
lb = [3, 0.01]
ub = [10, 0.3]

# Perform PSO
xopt, fopt = pso(xgb_objective, lb, ub, swarmsize=7, maxiter=100)

# Train the model with optimized parameters
model_opt = xgb.XGBClassifier(max_depth=int(xopt[0]), learning_rate=xopt[1], num_class=4, objective='multi:softmax', verbosity=1, seed=42)
model_opt.fit(X_train, y_train)
y_pred_opt = model_opt.predict(X_test)

# Evaluate and print optimized accuracy
accuracy_opt = accuracy_score(y_test, y_pred_opt)
print(f'Optimized Test Accuracy: {accuracy_opt:.4f}')

# Plot the best cost history over iterations
plt.figure(figsize=(10, 6))
plt.plot(best_cost_history, label='Best Cost per Iteration', color='blue', linewidth=2)  # Thicker curve
plt.title('PSO Best Cost Over Iterations', fontweight='bold', fontsize=14)  # Adjustable font size for the title
plt.xlabel('Iteration', fontweight='bold', fontsize=14)  # Bold and adjustable font size for X-axis
plt.ylabel('Loss', fontweight='bold', fontsize=14)  # Bold and adjustable font size for Y-axis
plt.xticks(fontweight='bold', fontsize=14)  # Bold and adjustable font size for X-axis ticks
plt.yticks(fontweight='bold', fontsize=14)  # Bold and adjustable font size for Y-axis ticks
plt.legend()
plt.grid(True)
plt.show()

import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# Compute the confusion matrix for the optimized predictions
cm = confusion_matrix(y_test, y_pred_opt)
cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Plotting the confusion matrix by percentage
plt.figure(figsize=(7, 5))
sns.heatmap(cm_percentage, annot=True, fmt=".2%", cmap='Blues', annot_kws={"size": 12, "weight": "bold"})
plt.title('Confusion Matrix by Percentage - Optimized Model', size=14, weight='bold')
plt.ylabel('Actual Label', size=12, weight='bold')
plt.xlabel('Predicted Label', size=12, weight='bold')
plt.xticks(fontsize=10, fontweight='bold')
plt.yticks(fontsize=10, fontweight='bold')
plt.show()


