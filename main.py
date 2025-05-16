import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv(r"D:\house price prediction\train.csv")

# Select features: square footage (GrLivArea), bedrooms (BedroomAbvGr), and bathrooms (FullBath + HalfBath)
data['TotalBath'] = data['FullBath'] + 0.5 * data['HalfBath']
features = data[['GrLivArea', 'BedroomAbvGr', 'TotalBath']]
target = data['SalePrice']

# Handle missing values (if any)
features = features.fillna(features.mean())
target = target.fillna(target.mean())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print model performance
print("Model Coefficients:")
print(f"GrLivArea: {model.coef_[0]:.2f}")
print(f"BedroomAbvGr: {model.coef_[1]:.2f}")
print(f"TotalBath: {model.coef_[2]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")
print("\nModel Performance:")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R-squared Score: {r2:.2f}")

# Example prediction
example_house = pd.DataFrame({
    'GrLivArea': [2000],
    'BedroomAbvGr': [3],
    'TotalBath': [2.5]
})
predicted_price = model.predict(example_house)
print(f"\nPredicted price for a 2000 sqft house with 3 bedrooms and 2.5 bathrooms: ${predicted_price[0]:,.2f}")