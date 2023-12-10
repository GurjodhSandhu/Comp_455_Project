import tkinter as tk
from tkinter import *
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

window = Tk()
window.title("House Price Estimator")
window.geometry("500x310")
window['background'] = 'white'


beds_label = Label(window, text="Number of Bedrooms:")
beds_label.pack()
beds_entry = Entry(window)
beds_entry.pack()

baths_label = Label(window, text="Number of Bathrooms:")
baths_label.pack()
baths_entry = Entry(window)
baths_entry.pack()

province_label = Label(window, text="Province:")
province_label.pack()

# Drop-down list for province
province_options = ['Alberta', 'British Columbia', 'Manitoba', 'New Brunswick', 'Newfoundland and Labrador',
                    'Nova Scotia', 'Ontario', 'Prince Edward Island', 'Quebec', 'Saskatchewan']
selected_province = StringVar()
selected_province.set(province_options[0])
province_dropdown = OptionMenu(window, selected_province, *province_options)
province_dropdown.pack()

city_label = Label(window, text="City:")
city_label.pack()

# Drop-down list for city
city_options = {
    'Alberta': ['Calgary', 'Edmonton'],
    'British Columbia': ['Vancouver', 'Victoria', 'Abbotsford', 'Kelowna'],  # enter the rest of the city names
    'Manitoba': ['Winnipeg'],
    'New Brunswick': ['Moncton', 'Saint John'],
    'Newfoundland and Labrador': ["St. John's"],
    'Nova Scotia': ['Halifax'],
    'Ontario': ['Toronto', 'Hamilton', 'Ottawa'],
    'Prince Edward Island': ['Charlottetown'],
    'Quebec': ['Montreal', 'Quebec City'],
    'Saskatchewan': ['Regina', 'Saskatoon']
}
selected_city = StringVar()
selected_city.set(city_options[selected_province.get()][0])
city_dropdown = OptionMenu(window, selected_city, *city_options[selected_province.get()])
city_dropdown.pack()


def update_city_dropdown(*args):
    city_dropdown['menu'].delete(0, 'end')
    for city in city_options[selected_province.get()]:
        city_dropdown['menu'].add_command(label=city, command=tk._setit(selected_city, city))


selected_province.trace('w', update_city_dropdown)


def estimate_price():
    beds = beds_entry.get()
    baths = baths_entry.get()
    city = selected_city.get()

    house = pd.read_csv('house.csv')
    house = house[(house['Province'] == selected_province.get()) & (house['City'] == city)]

    user_house = pd.DataFrame({'City': [city], 'Number_Beds': [beds], 'Number_Baths': [baths]})
    X = house[['Number_Beds', 'Number_Baths']]
    y = house['Price']

    if len(X) >= 2:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=1)
        linmod2 = LinearRegression()
        linmod2.fit(X_train, y_train)

        inputX = user_house[['Number_Beds', 'Number_Baths']]
        prediction = linmod2.predict(inputX)
        output = prediction[0]
        output = round(output,2)
        result_label.config(text="Estimated price: $" + str(output), bg = 'grey', font = ('Arial', 12, 'bold'))
        window['background'] = 'green'
    else:
        result_label.config(text="Not enough data to estimate price.")


estimate_button = Button(window, text="Estimate Price", command=estimate_price, bg='grey',font = ('Arial', 12, 'bold'))
estimate_button.pack()

result_label = Label(window)
result_label.pack()

window.mainloop()

result_label = Label(window)
result_label.pack()

window.mainloop()
