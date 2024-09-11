import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report
import joblib

# Load data from CSV
data = pd.read_csv('history_data.csv')

# Drop machine_id and maintenance_date as they are not needed for training
data = data.drop(columns=['machine_id', 'maintenance_date'])

# Ensure the 'failure' column has only binary values (0 or 1)
data = data[data['failure'].isin([0, 1])]

# Handle missing values
imputer = SimpleImputer(strategy='mean')
data_imputed = imputer.fit_transform(data)

# Feature scaling
scaler = StandardScaler()
X = data_imputed[:, :-1]  # Features (exclude the target 'failure' column)
y_class = data_imputed[:, -1]   # Target for classification

# Define a dummy regression target for expected days (this should come from actual data)
data['expected_days'] = data['failure'].apply(lambda x: 30 if x == 1 else 60)
y_regress = data['expected_days']

# Ensure the target variable is binary (0 or 1) for classification
y_class = y_class.astype(int)

# Fit the scaler on the feature data
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets for classification
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_scaled, y_class, test_size=0.2, random_state=42)

# Define the parameter grid for classification
param_grid_class = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced']
}

# Create a GridSearchCV object for classification
grid_search_class = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid_class, cv=5, n_jobs=-1, verbose=2)

# Train the classification model
grid_search_class.fit(X_train_class, y_train_class)

# Get the best classification model
best_model_class = grid_search_class.best_estimator_

# Predict and evaluate classification
y_pred_class = best_model_class.predict(X_test_class)
print(classification_report(y_test_class, y_pred_class))

# Save the best classification model and scaler
joblib.dump(best_model_class, 'predictive_model_class.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Split the data into training and testing sets for regression
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_scaled, y_regress, test_size=0.15, random_state=24)

# Define the parameter grid for regression
param_grid_reg = {
    'n_estimators': [60, 70, 80],
    'max_depth': [300],
    'max_features': [10],
    'min_samples_split': [5],
    'min_samples_leaf': [50]
}

# Create a GridSearchCV object for regression
grid_search_reg = GridSearchCV(estimator=RandomForestRegressor('friedman_mse', random_state=21), param_grid=param_grid_reg, cv=5, n_jobs=-1, verbose=3)

# Train the regression model
grid_search_reg.fit(X_train_reg, y_train_reg)

#Evaluate the base model
import numpy as np
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy
base_accuracy = evaluate(grid_search_reg, X_test_reg, y_test_reg)

# Get the best regression model
best_model_reg = grid_search_reg.best_estimator_

# Evaluate best regression model
random_accuracy = evaluate(best_model_reg, X_test_reg, y_test_reg)
print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))
# Save the best regression model
joblib.dump(best_model_reg, 'predictive_model_reg.pkl')
