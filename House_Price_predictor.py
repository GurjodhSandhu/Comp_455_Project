import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import scale, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, precision_recall_fscore_support,accuracy_score


%matplotlib inline
plt.style.use('seaborn-white')

house = pd.read_csv('housing_price_dataset.csv')
house.head()

X = house[['SquareFeet','Bedrooms','Bathrooms','YearBuilt']]
y = house['Price']


X_train = X[0:25000]
X_test = X[25000:]

y_train = y[0:25000]
y_test = y[25000:]


linmod2 = LinearRegression()

linmod2.fit(X_train, y_train)

preds = linmod2.predict(X_test)


plt.scatter(y_test, preds,color='g')
plt.xlabel("Actual y Value")
plt.ylabel("Predicted y Value")
plt.show()

mse = mean_squared_error(y_test, preds)
print(mse)
