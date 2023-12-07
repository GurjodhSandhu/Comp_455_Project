import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

%matplotlib inline
plt.style.use('seaborn-white')

# Load the dataset
house = pd.read_csv('house.csv')

# Filter the dataset for houses in Ontario
house = house[house['Province'] == "Ontario"]

# User input
city = input("Enter the city: ")
number_beds = int(input("Enter the number of bedrooms: "))
number_baths = int(input("Enter the number of bathrooms: "))

# Create a new DataFrame with user input
user_house = pd.DataFrame({'City': [city],
                           'Number_Beds': [number_beds],
                           'Number_Baths': [number_baths]})

# Merge user input with the dataset
X = pd.concat([house[['Price', 'Number_Beds', 'Number_Baths']], user_house])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X[['Number_Beds', 'Number_Baths']], X['Price'], train_size=0.5, random_state=1)

# Scale the features
X_train_scaled = scale(X_train)
X_test_scaled = scale(X_test)

# Train the linear regression model
linmod2 = LinearRegression()
linmod2.fit(X_train_scaled, y_train)

# Predict the house prices for the user input
user_price = linmod2.predict(scale(user_house[['Number_Beds', 'Number_Baths']]))[0]

# Display the predicted price
print("Estimated price for the given house:", user_price)

# Plot the predicted price against actual prices
plt.scatter(X_test['Price'], linmod2.predict(X_test_scaled), color='g')
plt.scatter(user_price, user_price, color='r', marker='x', label='User Input')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.legend()
plt.show()
