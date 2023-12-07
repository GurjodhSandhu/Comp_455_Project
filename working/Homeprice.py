import pandas as pd #importing libraries
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


#input --------------------------------------------------------------------------------------------------------------------------------------
beds= input("how many bedrooms: ")
baths = input("how many bathrooms: ")
City = input("what city are you in: ") # only limited options
#Province = input() #we need to only allow for limited options in GUI
#City = input()

#input --------------------------------------------------------------------------------------------------------------------------------------

house = pd.read_csv('house.csv') #read in housing dataset
house.head()



#currently hardcoded should take input for province ---------------------------------------------------------------------------------------------

house = house[house['Province'] =="Ontario" ] # restrict dataset to a certain province *change to take input*
house.tail() #display restricted dataset

# ---------------------------------------------------------------------------------------------


user_house = pd.DataFrame({'City': [City],'Number_Beds': [beds], 'Number_Baths': [baths]}) #take user input for data *change to take input*
user_house.head() #display the user inputed values

X = house[['Number_Beds','Number_Baths']] #predictors
y = house['Price'] #price of houses

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=1) #split up the training and testing data

linmod2 = LinearRegression() #use linear regression to create a model
linmod2.fit(X_train, y_train)
preds = linmod2.predict(X_test) #predict house prices of test data using test predictors
#print(X_test)
#print(user_house)

inputX = user_house[['Number_Beds','Number_Baths']]  #get user inputted data
print(user_house[['Number_Beds','Number_Baths','City']]) #display user inputted data
prediction = linmod2.predict(inputX) #predict house prices based on user inputted data

output = prediction[0] #estimated price for house based of factors given
print("value of your house is: $", output) #print out price


plt.scatter(y_test, preds,color='g')
plt.xlabel("Actual price")
plt.ylabel("Predicted price")
plt.show()
