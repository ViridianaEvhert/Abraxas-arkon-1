import os
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
from pandas.plotting import autocorrelation_plot
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.ar_model import AR

def main():
    """
    """
    """Set path to the input file."""
    in_file=r'daily-min-temperatures.csv'

    """Create output folder, where the result will be stored."""
    ou_dir=r'Results'
    if(not os.path.exists(ou_dir)):
        os.makedirs(ou_dir)

    """Read input file, specifying that the 1st row contain the column names."""
    df=pd.read_csv(in_file,header=0)

    """Print a summary of the input file. Useful to know what the column names
        are and what data type is stored on each column."""
    print(df.info(),'\n\n')

    """The 'Date' column stores dates, however, they are stored as strings.
        Lets convert them into an special data structure (time stamp), more
        suitable to handle dates."""
    df['Date']=pd.to_datetime(df['Date'],format='%Y-%m-%d')

    """By default, the dataframe index is the row number. For this particular
        project, it is better to set 'Date' as the dataframe index, so that we
        can access temperatures by specifying a particular date instead of a
        row number. When we do so, 'Date' will no longer be a regular column,
        but the dataframe's index."""
    df=df.set_index('Date')
    print(df.info(),'\n\n')

    """Now we will print the daily evolution of temperatures and save it to a
        file (within the output folder)."""
    ou_file=os.path.join(ou_dir,'Fig_Temp_vs_time.png')
    # Here we say "plot each data point in the column 'Temp'"
    df['Temp'].plot(title='Temp vs time')
    plt.savefig(ou_file)
    plt.close('all')

    """After examining the previous plot, we realize that temperatures repeat
        periodically after a 1 years (not surprising, right?). Lets have a
        closer look by plotting the tamporal evolution within the first 3
        years. To specify that we want to plot only specific rows of the column
        'Temp', we need to refer to the name/index of these rows. We do so by
        using the special method '.loc', specifying the range of row indexes
        first and then the name of the column we want to grab the data from."""
    ou_file=os.path.join(ou_dir,'Fig_Temp_vs_time_1st_3_years.png')
    df.loc['1981-01-01':'1984-01-01',['Temp']].plot(title='Temp vs time')
    plt.savefig(ou_file)
    plt.close('all')

    """After looking at the previous plot, we confirm that, indeed, temperatures
        repeat themselves after a one year period. This means that, if we plot
        the temperature values at a given time point vs temperatures a year
        after, we will observe a scatter set of points around the identity line,
        (i.e., we will observe a high correlation (positive slope) in a one
        year period). Such plots are already implemented in Pandas, under the
        name lag_plot, a function which takes a series of values, together with
        a specific lag time (specified as the NUMBER of rows we want to shift).
        Therefore, we need to know of many rows are between two consecutive
        years"""
    # Here we get the number of rows between two consecutive years. If we have
    # one temperature measurement each day, the lag period should be 365 days.
    lag=len(df.loc['1981-01-01':'1982-01-01',['Temp']])
    print('lag between 2 consecutive years:',lag) # just to confirm lag = 365
    # Now that we know the lag period, we can produce the plot.
    ou_file=os.path.join(ou_dir,'Fig_lagplot_1_year.png')
    # Lets specify a level of transparency to each point. This is somehow
    # advanced, but you could use 'lag_plot(df,lag=lag)' instead.
    kwarg={'alpha':0.1}
    lag_plot(df,lag=lag,**kwarg)
    plt.savefig(ou_file)
    plt.close('all')

    """We could even plot the correlation after half a year (366/2), i.e.,
        summer vs winter. In such case, we will observe a negative
        correlation!!!, i.e., a negative slope in the scatter plot.
        """
    ou_file=os.path.join(ou_dir,'Fig_lagplot_6_months.png')
    kwarg={'alpha':0.1}
    lag_plot(df,lag=int(lag/2),**kwarg)
    plt.savefig(ou_file)
    plt.close('all')

    """Based on what has been discussed so far, the autocorrelation plot for
        different lag times should display a fluctuating behavior, alternating
        between positive an negative values, and with a decreasing envelope for
        largetimes."""
    ou_file=os.path.join(ou_dir,'Fig_autocorrelation.png')
    autocorrelation_plot(df)
    plt.savefig(ou_file)
    plt.close('all')

    
    """Now we jump to the persistence prediction model. We whant to develop a model to predict 
	the last 7 days. We could use to make predictions would be to persist the last observation 
	and it provides a baseline of performance for the problem that we can use for comparison 
	with another model. 
        We start by creating a new dataframe with the original and lagged
        temperatures, named 't-1' and 't+1', respectively."""
    df_lag=pd.concat([df.shift(1),df],axis=1)
    df_lag.columns=['t-1','t+1']
    print(df_lag.info())

    """Now we split the data into train and testing subsets."""
    x=df_lag.values
    train=x[1:len(x)-7]
    test=x[len(x)-7:]

    train_x=train[:,0]
    train_y=train[:,1]

    test_x=test[:,0]
    test_y=test[:,1]

    """Now we predict using the simplistic, persistence model."""
    # persistence model
    def model_persistence(x):
        return x

    predictions=[]
    for temp in test_x:
        pred=model_persistence(temp)
        predictions.append(pred)
    score=mean_squared_error(test_y,predictions)
    print('Test MSE:',score)

    ou_file=os.path.join(ou_dir,'Fig_model_persistence.png')
    plt.plot(test_y,c='b',label='true values')
    plt.plot(predictions,c='r',label='predicted values')
    plt.xlabel('Days')
    plt.ylabel('Temperature')
    plt.title('Persistence model, MSE = {:0.3f}'.format(score))
    plt.legend()
    plt.savefig(ou_file)
    plt.close('all')

    """Now the autoregression model from statsmodels.
	The statsmodels library provides an autoregression model
	that automatically selects an appropriate lag value using 
	statistical tests and trains a linear regression model whit 
	AR() and after we can use the model to make a prediction by 
	calling the predict()"""
    """If you need to re-run the thing and still have issues because of the
        statsmodels package, just comment from here and until the line before
        if __name__=='__main__', as well as line 9"""
    x=df.values
    train=x[1:len(x)-7]
    test=x[len(x)-7:]
    model=AR(train)
    model_fit=model.fit()
    print('Lag:',model_fit.k_ar)
    print('Coefficients:',model_fit.params)
    # make predictions
    predictions=model_fit.predict(
        start=len(train),end=len(train)+len(test)-1,dynamic=False
        )
    for i in range(len(predictions)):
        print('predicted={0}, expected={1}'.format(predictions[i],test[i]))
    score=mean_squared_error(test,predictions)
    print('Test MSE:',score)

    ou_file=os.path.join(ou_dir,'Fig_model_autoregression.png')
    plt.plot(test_y,c='b',label='true values')
    plt.plot(predictions,c='r',label='predicted values')
    plt.xlabel('Days')
    plt.ylabel('Temperature')
    plt.title('Persistence model, MSE = {:0.3f}'.format(score))
    plt.legend()
    plt.savefig(ou_file)
    plt.close('all')

if __name__=='__main__':
    main()
