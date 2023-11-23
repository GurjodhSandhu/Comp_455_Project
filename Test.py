# test code this is what I have so far ...
# pip install pandas scikit-learn numpy
# just an idea to get started.. still need code that uses hashmap and some type of other method we learned in class

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the dataset
file_path = 'housing_pricing.csv'
df = pd.read_csv(file_path)

# Features (X) and target variable (y)
X = df.drop('Price', axis=1)
y = df['Price']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error on the test set: {rmse}")

# Take user input for a new house
user_input = {
    'SquareFeet': float(input("Enter the square feet of the house: ")),
    'Bedrooms': int(input("Enter the number of bedrooms: ")),
    'Bathrooms': int(input("Enter the number of bathrooms: ")),
    'Neighborhood': input("Enter the neighborhood: "),
    'YearBuilt': int(input("Enter the year the house was built: "))
}

# Convert user input into a DataFrame for prediction
user_df = pd.DataFrame([user_input])

# Use the trained model to predict the price
predicted_price = model.predict(user_df)
print(f"Estimated price for the input house: ${predicted_price[0]:,.2f}")

