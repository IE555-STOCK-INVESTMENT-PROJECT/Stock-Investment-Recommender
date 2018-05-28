def forecast(stock_ticker, n_stock, n_days):
    from alpha_vantage.timeseries import TimeSeries
    import warnings
    import itertools
    import pandas as pd
    import numpy as np
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    import plotly.plotly as py
    import plotly.tools as tls
    import plotly.graph_objs as go

    tls.set_credentials_file(username = 'Enter Username here', api_key = 'Enter Plotly API here')


    # Retreiving the data from API
    ts = TimeSeries(key='Enter Alphavantage Api here', output_format = 'pandas')
    data, meta_data = ts.get_daily(symbol=stock_ticker, outputsize='full')
    current_price = data.iloc[:,1][-1]
    investment = n_stock * current_price

    data = data.iloc[:,1]
    data = data.fillna(data.bfill())

    ########################################################################################
    #ARIMA model
    ########################################################################################
    ''''
    # Define the p, d and q parameters to take any value between 0 and 2
    p = d = q = range(0, 2)
    
    # Generate all different combinations of p, q and q triplets
    pdq = list(itertools.product(p, d, q))
    
    # Generate all different combinations of seasonal p, q and q triplets
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    
    #print('Examples of parameter combinations for Seasonal ARIMA...')
    #print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
    #print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
    #print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
    #print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
    
    
    count = 1000000
    warnings.filterwarnings("ignore") # specify to ignore warning messages
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(data,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
                
                results = mod.fit()
                if results.aic < count:
                    order1 = param
                    order2 = param_seasonal
                    count = results.aic
                    print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
            except:
                continue

    '''
    mod = sm.tsa.statespace.SARIMAX(data,
                                    order= (1,1,1),
                                    seasonal_order= (1,1,1,12),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    
    results = mod.fit()
    
    #print(results.summary().tables[1])   
    
    results.plot_diagnostics(figsize=(15, 12))
    plt.show()     
    
    #######################################################################################
    # Plotting the time series dynamic = false       
    pred = results.get_prediction(start=pd.to_datetime(data.index[len(data)/2]), dynamic=False)
    pred_ci = pred.conf_int()      
    
    
    pred_ci.index =pred_ci.index.map(lambda t: t.strftime('%Y-%m-%d'))
    prediction = np.array(pred_ci)
    pred_ci['avg'] = pred_ci.mean(axis = 1)
    # Plotly
    timeseries1 = go.Scatter(x = data.index, y = data, name = 'Actual stock price')
    timeseries2 = go.Scatter(x = pred_ci.index, y = pred_ci['avg'], name = 'Test stock price')
    timeseries = [timeseries1, timeseries2]
    layout = dict(title = "Train test plot", xaxis = dict(range = [data.index[0], data.index[len(data)-1]]))
    fig_ts1 = dict(data = timeseries, layout = layout)
    py.plot(fig_ts1, filename = 'timeseres1')
    
    
    # Compute the mean square error
    mse = ((pred_ci['avg'] - data[len(pred_ci):]) ** 2).mean()
    print "\n"
    print "Result:"
    print "\n"
    print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
    
    
    ###############################################################################################
    
    # Producing forecasts
    # Get forecast 10 steps ahead in future
    
    #################################################################
    rng = pd.date_range(data.index[-2], periods=n_days+3, freq='D')
    rng = rng.map(lambda t: t.strftime('%Y-%m-%d'))
    rng = pd.DataFrame(rng)
    index = pd.DataFrame(data.index[:-2])
    index.columns = [0]
    dateindex1 = [index, rng]
    dataindex = pd.concat(dateindex1)
    dataindex.index = dataindex
    #################################################################
    pred_uc = results.get_prediction(end = len(data)+n_days)
    # Get confidence intervals of forecasts
    pred_uc = pred_uc.conf_int()
    pred_uc['avg'] = pred_uc.mean(axis = 1)
    
    # Plotly plots:
    timeseries3 = go.Scatter(x = dataindex.index, y = pred_uc.iloc[14:,0], name = 'Low')
    timeseries4 = go.Scatter(x = dataindex.index, y = pred_uc.iloc[14:,1], name = 'High')
    timeseries5 = [timeseries3, timeseries4]
    layout = dict(title = 'Forecasted stock prices')
    plot = dict(data = timeseries5, layout = layout)
    py.plot(plot)
    ################################################################
    forecasted_stock_price = pred_uc.iloc[:,2][len(pred_uc)-1]
    roi = n_stock * forecasted_stock_price
    strategy = roi - investment
    print "The latest stock price = $%f" %current_price
    print "Current investment = $%f" %investment
    print "Stock price on %s = $%f" %(dataindex[0][-2],forecasted_stock_price)
    print "Return on investment on %s = $%f" %(dataindex[0][-2],roi)
    if (strategy >= 0):
        print "Profit =",strategy
        print "Recommended Strategy:"
        print "Hold or sell"
        
    if (strategy < 0):
        print "Loss =",-strategy
        print "Recommended Strategy:"
        print "Hold or buy"
        
    
    
stock_ticker = raw_input("Enter Stock ticker: \n")
n_stock = int(raw_input("Enter number of stocks to buy: \n"))
n_days = int(raw_input("Enter the number of days after which you want the forecast: \n"))
forecast(stock_ticker, n_stock, n_days)
        
