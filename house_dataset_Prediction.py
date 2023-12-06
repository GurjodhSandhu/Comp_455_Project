# %load ../standard_import.txt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import scale, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, precision_recall_fscore_support,accuracy_score
from sklearn.model_selection import train_test_split


%matplotlib inline
plt.style.use('seaborn-white')


house = pd.read_csv('house.csv')
house.head()

house = house[house['Province'] =="Ontario" ]
house.head()

#print(house)


X = house[['Price','Number_Baths','Number_Beds']]
y = house['Price']


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=1)

linmod2 = LinearRegression()
linmod2.fit(X_train, y_train)
preds = linmod2.predict(X_test)

plt.scatter(y_test, preds,color='g')
plt.xlabel("Actual y Value")
plt.ylabel("Predicted y Value")
plt.show()
