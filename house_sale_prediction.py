import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('train.csv')

# Feature Engineering
data['TotalBathrooms'] = data['FullBath'] + 0.5 * data['HalfBath'] + data['BsmtFullBath'] + 0.5 * data['BsmtHalfBath']

# Select features and target variable
features = ['GrLivArea', 'TotalBathrooms', 'GarageCars', 'OverallQual', 'YearBuilt', 'GarageArea']
X = data[features]
y = data['SalePrice']

# Handling Missing Data
X.fillna(0, inplace=True)  # Fill missing values with 0 (you can use more sophisticated methods)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared: {r2:.2f}')

# Visualization (pairplot)
data['SalePrice'] = y  # Add SalePrice back to the dataset for visualization
sns.pairplot(data, x_vars=features, y_vars='SalePrice', height=5, aspect=0.7)
plt.show()

new_data = pd.DataFrame({
    'GrLivArea': [2000],  # Square footage
    'TotalBathrooms': [2.5],  # Total bathrooms
    'GarageCars': [2],  # Number of garage cars
    'OverallQual': [7],  # Overall quality
    'YearBuilt': [1990],  # Year built
    'GarageArea': [500]  # Garage area
})

# Standardize the new data using the same scaler
new_data_std = scaler.transform(new_data)

# Make predictions for the new data
predicted_prices = model.predict(new_data_std)

# Print the predicted prices
print(f'Predicted Prices: {predicted_prices[0]:.2f}')
