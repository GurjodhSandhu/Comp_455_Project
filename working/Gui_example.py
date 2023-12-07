import tkinter as tk
from tkinter import *
import pandas as pd

OPTIONS = [
"Alberta",
"Ontario",
"British Columbia",
"Manitoba",
"Saskatchewan",
"Newfoundland and Labrador",
"New Brunswick",
"Quebec",
"Nova Scotia",
] #etc

window = Tk()

window.title("average price calculator")
window.resizable(width=False, height=False)
window.geometry("500x310")
window['background'] = '#0f0069'

variable = StringVar(window)
variable.set(OPTIONS[0]) # default value
w = OptionMenu(window, variable, *OPTIONS)

w.pack()

def Mapreduce():
    def Mapper(Province, Prices):  # mapper function altered to fulfill my requirements
        size = 1  # this is the amount of times a country repeats in this program
        k = 0
        pairs = []  # initialized an array
        while k < len(Province):  # iterate through file
            pair = (Province[k], size, Prices[k])  # add current country, 1 and credit status to pair
            pairs.append(pair)  # add the key-value pair to list (and another key in this case)
            k += 1
        return pairs


    def Reducer(oldpairs):
        dictionary = {}  # using dictionary to lower time complexity
        for index in oldpairs:  # iterate through the key-value list
            # since the key pair were initially put into 2d list first index is country second is size 3: price
            countrykey = index[0]
            size = index[1]
            price = index[2]
            if countrykey in dictionary:  # if country is already in the dict
                dictionary[countrykey][0] += price  # add 1 to size(size)
                dictionary[countrykey][1] += size  # creates new list that can be accessed by key (country)

            else:
                dictionary[countrykey] = [0, 0]  # creates new list that can be accessed by key (country)

        return dictionary

    df = pd.read_csv('house.csv')  # read in dataset using panda library
    Prov = df['Province']  # hold the country column
    price = df['Price']  # holding the credit status column

    itemset = (
        Reducer(Mapper(Prov, price)))  # using both mapper and reducer function and storing in itemset variable

    inputforprovince = variable.get()
    mean = itemset[inputforprovince][0] / itemset[inputforprovince][1]
    print("average price of a home in ", inputforprovince, "is $", mean)

button = Button(window, text="Get average price", command=Mapreduce)

button.pack()

mainloop()
