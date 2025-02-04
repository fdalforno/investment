import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time

def plot_returns(data, names, flag='Total Return', printFinalVals = False):
    """
    Plots the returns of the given columns in the pandas dataframe data.
    
    Parameters
    ----------
    data : pandas dataframe
        The data to be plotted.
    names : list of strings
        The names of the columns to be plotted.
    flag : string
        The type of plot to be generated. Options are 'Total Return' and 'Return'.
    printFinalVals : boolean
        Whether or not to print the final values of the columns.
    """

    plt.figure(figsize=(12, 8))

    if(type(names) is str):
        names = [names]
    for name in names:
        if(name not in data.columns):
            print ('column ' + name + ' not in pandas df')
            return


    if (flag == 'Total Return'):
        n = data.shape[0]
        totalReturns = np.zeros((n,len(names)))
        totalReturns[0,:] = 1.0

        for i in range(1,n):
            totalReturns[i,:] = np.multiply(totalReturns[i-1,:], (1 + data[names].values[i,:]))
        
        for j in range(len(names)):
            plt.semilogy(data.index, totalReturns[:,j])
        
        plt.title('Total Return Over Time')
        plt.ylabel('Total Return')
        plt.legend(names)
        plt.xlabel('Date')


        # Formattazione dell'asse x per mostrare le date ogni 4 mesi
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)


        if(printFinalVals):
            print(totalReturns[-1])

        plt.show()
    elif (flag == 'Return'):
        for i in range(len(names)):
            plt.plot(data.index, data[names[i]])
        plt.title('Returns Over Time')
        plt.ylabel('Returns')
        plt.legend(names)
        plt.xlabel('Date')
        plt.show()
    else:
        print('Invalid flag')
        return


def display_factor_loadings(intercept, coefs, factorNames,name='Beta'):
    loadings = np.insert(coefs, 0, intercept)
    out = pd.DataFrame(loadings, columns=[name])
    out = out.transpose()
    fullNames = ['Intercept'] + factorNames
    out.columns = fullNames

    return out