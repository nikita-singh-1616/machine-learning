# given prices and area of houses we are going to predict the price of a house based on the given area Linear
# regression plots all the given data points on the graph and then finds the equation of a straight line that has
# mini error with all points eq of line--> y = mx+c --> in our case y is the price x is the area m is the slope and c
# is the intercept price is y because price is the dependent variable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


class LinearRegressionModel:
    def __init__(self):
        pass
    @staticmethod
    def linear_regression_model_with_single_variable():
        data = {
            'Area': [100, 200, 300, 430, 500],
            'Price': [1000, 3500, 4000, 6344, 9899]
        }
        df = pd.DataFrame(data)
        plt.xlabel('Area')
        plt.ylabel('Price')
        plt.scatter(df.Area, df.Price)
        plt.show()

        linear_regression_model = linear_model.LinearRegression()
        linear_regression_model.fit(df[['Area']], df['Price'])
        predicted_value = linear_regression_model.predict([[4500]])

        plt.xlabel('Area')
        plt.ylabel('Price')
        plt.scatter(df.Area, df.Price)
        plt.plot(df.Area, linear_regression_model.predict(df[['Area']]))
        plt.show()

        # there are multiple ways of drawing a line through this set of values but this is the best fit as the mean sqaure
        # value of error is least
        # the mean_square_err = sum((yi-y_calculated_i)**2)//n ---------also called cost function
        # gradient descent is the algo that is used to get the best m and c

        # this model as mentioned above generates an equation y = mx + c , lets get the m and c values
        m = linear_regression_model.coef_
        c = linear_regression_model.intercept_
        # so the eq will be price = m*area + c
        price = m * 4500 + c
        print(f'checking if predicted value({predicted_value}) is equal to the calculated value({price}).')
        return linear_regression_model

    # linear regression for multi variable
    @staticmethod
    def linear_regression_for_multi_variable():
        data = {
            'area': [2600, 3000, 3200, 3600, 4000],
            'bedrooms': [3, 4, None, 3, 5],
            'age': [20, 15, 18, 30, 8],
            'price': [550000, 565000, 610000, 595000, 760000]
        }
        df = pd.DataFrame(data)
        # LINEAR REGRESSION EXP => PRICE = m1*area +m2*bedrooms +m3*age+c
        # always clean data to remove nan values
        bedroom_median = df.bedrooms.median()
        df.bedrooms = df.bedrooms.fillna(bedroom_median)
        reg = linear_model.LinearRegression()
        reg.fit(df[['area', 'bedrooms', 'age']], df.price)
        model_result = reg.predict([[1000, 1, 1]])
        m1 = reg.coef_[0]
        m2 = reg.coef_[1]
        m3 = reg.coef_[2]
        c = reg.intercept_
        price = m1 * 1000 + m2 * 1 + m3 * 1 + c
        print(f'calculated price - {price}\npredicted value - {model_result[0]}')
        return reg
