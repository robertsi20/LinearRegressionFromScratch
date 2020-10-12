"""
Author: Jay Roberts
File: linearRegression.py
Description: This model implements a linear regression from scratch that uses gradient descent as the optimization algorithm minimizing
the mean squared error cost function. Also, comparisons are made to Sci-Kit Learn's and Statsmodels OLS Linear Regresssion models.
"""

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import statsmodels.graphics.api as smg
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import random 


#defines the From Scratch Linear Regression Class 
class Linear_Regression():

    #initiates the instance variables that will be used
    # alpha is the learning rate
    # accepts alpha, number of iterations, and initial constant value
    def __init__(self, alpha, iterations, constant):
        self.alpha = alpha 
        self.iterations = iterations
        self.constant = constant

    # function that trains the model on the training set
    def train(self, data, target):
        #list of costs will be used in a plot to show gradient descent rate of optimization
        cost_list = []

        # stores each column of dataframe into a list
        lyst = list()
        for i in range(data.shape[1]):
                lyst.append(data.iloc[:,i])
        
        #changes list of columns into a numpy array for processing
        arr = np.array(lyst)
        
        #defines an array of random floats between 0 and 1 for the slopes
        slopes = np.random.rand(data.shape[1])
        
        #Gradient Descent process
        for i in range(self.iterations):

            #Uses the vector dot product for linear regression equation 
            # gets into form: y = column0*slopes[0] + column1*slopes[1] + ... + columnN*slopes[N]
            #Uses .T to transpose the matrix to ensure matrix multiplication
            dotprod = np.dot(arr.T, slopes)

            #adds the constant to the linear equation 
            pred_data = dotprod + self.constant

            #calculates the residuals
            errors = target - pred_data

            #calculates the partial derivative with respect to the slope
            # and updates the slope 
            for i in range(arr.shape[0]):
                d_slope = (-2/(arr[i].shape[0])) * sum(arr[i] * errors)
                slopes[i] = slopes[i] - (self.alpha * d_slope)
            
            # calculates the partial derivative with respect to the constant
            # and updates the constant 
            d_constant = (-2/(arr[i].shape[0])) * sum(errors)
            self.constant = self.constant - (self.alpha * d_constant)

            # calls the costfunction to calculate the mean squared error 
            cost = self.costFunction(arr, target, slopes)

            #adds the cost to the cost list
            cost_list.append(cost)
        
        #calls the calculateRSquared function to record the R^2 value of the training set
        rSquared = self.calculateRSquared(pred_data, target)
        
        return slopes, cost_list, cost, rSquared

    #defines the MSE Cost function 
    def costFunction(self, data, Y, slope):

        #Uses .T to transpose the matrix to ensure matrix multiplication
        #calculates the predicted value 
        Y_pred = np.dot(data.T,slope) + self.constant

        #calculates errors, squares them, and then sums them and divides by the length of the data set
        errors = Y - Y_pred
        mse = (errors)**2
        cost = sum(mse)/(data.shape[1])
        return cost

    #defines the prediction method
    #intended to be used after training
    def predict(self, X, slopes):
        y_pred = np.dot(X, slopes) + self.constant
        return y_pred

    #calculates the R^2 value
    def calculateRSquared(self, prediction, target):
        return 1-(np.sum((target - prediction)**2) / np.sum((target - np.mean(target))**2))
    
    #normalizes the inputs by subtracting the mean of the column and dividing by the columns standard deviation
    def normalize(self, data):
        for column in data.columns:
            data[column] = (data[column] - np.mean(data[column])) / np.std(data[column])
        return data
        
#plots correlation matrix from numpy function calls
def plotCorrMatrix(data, target, data_columns):
    
    #creates the correlation matrix from numpy
    corr_matrix = np.corrcoef(target,data.T)
    
    #calls statsmodels.graphics.api to create the image of the correlation matrix
    smg.plot_corr(corr_matrix, xnames=data_columns)
    plt.show()

#defines a plot cost function
#requires a list of costs
def plotCostFunction(costs):
    plt.plot(range(1,len(costs)+1), costs)
    plt.xlabel('Number of Iterations')
    plt.ylabel('Mean Squared Error')
    plt.title('Cost Function')
    plt.show()



def main():

    #list of column names
    column_list = ['crime_rate','res_land_lots>25000','proportion_non-retail_bus_acres',
    'river','nitric_oxide_conc','avg_no_rooms','prop_o_o_units_1940','dist_5_employment_center',
    'index_access_hwy','property_tax/$10000','student/teacher', 'ethnic_demo','%_low_stat_pop',
    'med_val_o_o_homes_in_1000s']

    #creates the pandas dataframe witht the column list as the name of each column
    #and uses the whitespace as a delimiter
    data = pd.read_csv('housing.csv', names = column_list,  delim_whitespace=True)
    
    #grabs the target column
    Y = data.iloc[:,13]

    #drops specific column from the dataframe
    data = data.drop(columns=['med_val_o_o_homes_in_1000s'])
    
    #makes a list of the data columns
    columns = list(data.columns)
    
    #appends the target column to the columns
    columns.append('med_val_o_o_homes_in_1000s')
    
    #calls the plot correlation matrix function
    plotCorrMatrix(data, Y, columns)

    #Initiates the from scratch model
    lin_reg = Linear_Regression(.1, 250, random.random())
    
    #normalizes the data
    normalizedData = lin_reg.normalize(data)
    
    #uses sklearns train_test_split function
    x_train, x_test, y_train, y_test = train_test_split(normalizedData, Y, test_size=0.2)
    
    #calls the train function on the training data
    slopes, cost_list, finalCost, rSquared = lin_reg.train(x_train, y_train)
    print("From Scratch Model coefficients:", slopes)

    plotCostFunction(cost_list)

    #calls the predict method
    y_pred = lin_reg.predict(x_test,slopes)
    
    #returns the rSquared value for the test prediction
    r_squared = lin_reg.calculateRSquared(y_pred, y_test)
    print("From Scratch Model Test Prediction R^2 value:", r_squared)

    #function call to sci-kit learn linear regression model
    model = LinearRegression()
    model.fit(x_train, y_train)
    print("\nSci-Kit Learn Test Prediction R^2 Value:", model.score(x_test,y_test))
    print("\nSci-Kit Learn Coefficients:", model.coef_)

    #function call to statsmodel OLS Regression model
    #adds a column of ones to the data so that the sm model will infer a constant
    data = sm.add_constant(data)
    model = sm.OLS(Y, data)
    result = model.fit()
    print(result.summary())
    

if __name__ == "__main__":
   main()
