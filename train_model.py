import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
import pickle
import json

# Load data
data = pd.read_csv('sales_data.csv')

# Handle missing values
data['Special_Events'] = data['Special_Events'].fillna('None')

# Convert data types
data['Price'] = data['Price'].astype(float)
data['Competitor_Price'] = data['Competitor_Price'].astype(float)
data['Promotion'] = data['Promotion'].map({'Yes': 1, 'No': 0})
data['Holiday_Flag'] = data['Holiday_Flag'].map({'Yes': 1, 'No': 0})
data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')
data['Month'] = data['Date'].dt.month
data['Day_of_Week'] = data['Date'].dt.day_name()

# Remove outliers
price_mean = data['Price'].mean()
price_std = data['Price'].std()
data = data[(data['Price'] >= price_mean - 3 * price_std) & (data['Price'] <= price_mean + 3 * price_std)]

# Features and target
features = ['Price', 'Units_Sold', 'Promotion', 'Holiday_Flag', 'Competitor_Price', 'Month', 'Product_ID', 'Category', 'Special_Events', 'Day_of_Week']
X = data[features]
y = np.log1p(data['Revenue'])  # Log-transform target

# One-hot encode categorical variables
X = pd.get_dummies(X, columns=['Product_ID', 'Category', 'Special_Events', 'Day_of_Week'])

# Save model columns
model_columns = X.columns.tolist()
with open('model_columns.json', 'w') as f:
    json.dump(model_columns, f)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model with GridSearchCV
param_grid = {
    'n_estimators': [50, 100],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5]
}
model = GradientBoostingRegressor(random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")

# Evaluate model
y_pred = best_model.predict(X_test)
mse = ((np.expm1(y_pred) - np.expm1(y_test)) ** 2).mean()  # Reverse log for MSE
r2 = best_model.score(X_test, y_test)
print(f"Test MSE: {mse:.2f}, R²: {r2:.2f}")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_model.feature_importances_
}).sort_values(by='Importance', ascending=False)
print("Feature Importance:\n", feature_importance.head(10))

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Sample prediction
sample = X_test.iloc[0:1]
pred = np.expm1(best_model.predict(sample))[0]
print(f"Sample prediction: ${pred:.2f}")