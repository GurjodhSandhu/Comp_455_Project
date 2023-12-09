import pandas as pd #importing libraries
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Mapper function to find mean value of houses
def Mapreduce(province):
    def Mapper(Province, Prices):  # mapper function altered to fulfill my requirements
        size = 1  # this is the amount of times a province repeats in this program
        k = 0
        pairs = []  # initialized an array
        while k < len(Province):  # iterate through file
            pair = (Province[k], size, Prices[k])  # add current province, 1 and price to pair
            pairs.append(pair)  # add the key-value pair to list (and another key in this case)
            k += 1
        return pairs


    def Reducer(oldpairs):
        dictionary = {}  # using dictionary to lower time complexity
        for index in oldpairs:  # iterate through the key-value list
            # since the key pair were initially put into 2d list first index is province second is size 3: price
            provincekey = index[0]
            size = index[1]
            price = index[2]
            if provincekey in dictionary:  # if province is already in the dict
                dictionary[provincekey][0] += price  # add 1 to size(size)
                dictionary[provincekey][1] += size  # creates new list that can be accessed by key (province)

            else:
                dictionary[provincekey] = [0, 0]  # creates new list that can be accessed by key (province)

        return dictionary

    df = pd.read_csv('house.csv')  # read in dataset using panda library
    Prov = df['Province']  # hold the province column
    price = df['Price']  # holding the price column

    itemset = (
        Reducer(Mapper(Prov, price)))  # using both mapper and reducer function and storing in itemset variable

    inputforprovince = province
    mean = itemset[inputforprovince][0] / itemset[inputforprovince][1] #divides the total price calculated of the province by the size of province (mapreduce)
    print("average price of a home in ", inputforprovince, "is $", round(mean,2)) #display data
    return mean

def predict(beds, baths, city, province):
    house = pd.read_csv('house.csv') #read in housing dataset
    house.head()
    house = house[house['Province'] == province ] # restrict dataset to a certain province
    house.tail() #display restricted dataset
    # ---------------------------------------------------------------------------------------------
    user_house = pd.DataFrame({'City': [city],'Number_Beds': [beds], 'Number_Baths': [baths]}) #take user input for data
    user_house.head() #display the user inputed values

    X = house[['Number_Beds','Number_Baths']] #predictors
    y = house['Price'] #price of houses

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=1) #split up the training and testing data

    linmod2 = LinearRegression() #use linear regression to create a model
    linmod2.fit(X_train, y_train)
    preds = linmod2.predict(X_test) #predict house prices of test data using test predictors


    inputX = user_house[['Number_Beds','Number_Baths']]  #get user inputted data
    print(user_house[['Number_Beds','Number_Baths','City']]) #display user inputted data
    prediction = linmod2.predict(inputX) #predict house prices based on user inputted data

    output = prediction[0] #estimated price for house based of factors given
    print("value of your house is: $", output) #print out price


    plt.scatter(y_test, preds,color='g')
    plt.xlabel("Actual price")
    plt.ylabel("Predicted price")
    plt.show()
    return round(output,2)

choose = input("Enter M to retrieve average house price in your province \nEnter P to predict housing price\n")
if(choose == 'M'):
    print("Available provinces: Alberta | British Columbia | Manitoba | New Brunswick | Newfoundland and Labrador \n| Nova Scotia | Ontario | Prince Edward Island | Quebec | Saskatchewan")
    province = input("what province are you in: ")
    Mapreduce(province)                                         #uses function to get mean

if(choose == 'P'):
    beds= input("how many bedrooms: ")                          #number of beds users has
    baths = input("how many bathrooms: ")                       #number bathrooms user has
    city = input("what city are you in: ")
    print("Available provinces: Alberta | British Columbia | Manitoba | New Brunswick | Newfoundland and Labrador \n| Nova Scotia | Ontario | Prince Edward Island | Quebec | Saskatchewan")
    province = input("what province are you in: ")
    predict(beds, baths, city, province)

print("Have a good day.")
