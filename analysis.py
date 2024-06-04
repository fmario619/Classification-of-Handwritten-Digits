import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import GridSearchCV

import pandas as pd
# Load data
(x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data(path="mnist.npz")


# Reshape features and assign to X,y variables
X = np.reshape(x_train, (60000, 784))
y = y_train
# Descriptive statistics
num_class = np.unique(y_train)
feat_shape = np.shape(x_train)
targ_shape = np.shape(y_train)
min_feat = round(np.min(x_train), 1)
max_feat = round(np.max(x_train), 1)

# Print statistics
'''print(f'Classes: {num_class}')
print(f"Features' shape: {feat_shape}")
print(f"Target's shape: {targ_shape}")
print(f"min: {min_feat}, max: {max_feat}")'''

A= X[:6000, :]
b= y[:6000]

A_train, A_test, b_train, b_test = train_test_split(A, b, test_size=0.3, random_state=40)

scaler = Normalizer()

x_train_norm= scaler.fit_transform(A_train)
x_test_norm= scaler.transform(A_test)

param_grid_knn = {'n_neighbors': [3, 4], 'weights': ['uniform', 'distance'], 'algorithm': ['auto', 'brute']}
param_grid_rf = {'n_estimators': [300, 500], 'max_features': ['sqrt', 'log2'], 'class_weight': ['balanced', 'balanced_subsample']}

def fit_predict_eval(model, features_train, features_test, target_train, target_test):
    # here you fit the model
    model.fit(features_train, target_train)
    # make a prediction
    y_pred = model.predict(features_test)
    # calculate accuracy and save it to score
    score = accuracy_score(target_test, y_pred)
    return score

grid_search_knn = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_grid_knn, scoring='accuracy', n_jobs=-1)
grid_search_knn.fit(x_train_norm, b_train)
results_estimator_knn = grid_search_knn.best_estimator_
results_parameters_knn = grid_search_knn.best_params_
results_score_knn = grid_search_knn.best_score_

grid_search_rf = GridSearchCV(estimator=RandomForestClassifier(random_state=40), param_grid=param_grid_rf, scoring='accuracy', n_jobs=-1)
grid_search_rf.fit(x_train_norm, b_train)
results_estimator_rf = grid_search_rf.best_estimator_
results_parameters_rf = grid_search_rf.best_params_
results_score_rf = grid_search_rf.best_score_

print("K-nearest neighbors algorithm")
print(f"best estimator: {results_estimator_knn}")
print(f"accuracy: {fit_predict_eval(grid_search_knn, x_train_norm, x_test_norm, b_train, b_test)}\n")
print("")
print("Random Forest algorithm")
print(f"best estimator: {results_estimator_rf}")
print(f"accuracy: {fit_predict_eval(grid_search_rf, x_train_norm, x_test_norm, b_train, b_test)}\n")
