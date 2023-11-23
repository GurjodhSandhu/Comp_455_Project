# test code this is what I have so far ...
# just an idea to get started...

# pip install pandas scikit-learn numpy
#pip install datasketch
# pip install numpy

import pandas as pd
from sklearn.model_selection import train_test_split
from datasketch import MinHashLSHForest, MinHash
import numpy as np

# Load the dataset
file_path = 'housing_pricing_dataset.csv'
df = pd.read_csv(file_path)

# Features (X) and target variable (y)
X = df.drop('Price', axis=1)
y = df['Price']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the dataset to MinHash representation
def convert_to_minhash(row):
    minhash = MinHash()
    for col in row.index:
        minhash.update(str(row[col]).encode('utf-8'))
    return minhash

X_train_minhash = X_train.apply(convert_to_minhash, axis=1)
X_test_minhash = X_test.apply(convert_to_minhash, axis=1)

# Train a MinHash LSH Forest
forest = MinHashLSHForest(num_perm=128)
for i, minhash in enumerate(X_train_minhash):
    forest.add(i, minhash)
forest.index()

# Take user input for a new house
user_input = {
    'SquareFeet': float(input("Enter the square feet of the house: ")),
    'Bedrooms': int(input("Enter the number of bedrooms: ")),
    'Bathrooms': int(input("Enter the number of bathrooms: ")),
    'Neighborhood': input("Enter the neighborhood: "),
    'YearBuilt': int(input("Enter the year the house was built: "))
}

# Convert user input to MinHash for querying
user_minhash = convert_to_minhash(pd.Series(user_input))

# Query the LSH Forest for similar houses
query_result = list(forest.query(user_minhash, 3))  # Get 3 most similar houses

# Calculate the estimated price based on the average price of similar houses
estimated_price = np.mean(y_train.iloc[query_result])
print(f"Estimated price for the input house: ${estimated_price:,.2f}")
