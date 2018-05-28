# Efficient Market Hypothesis..

from alpha_vantage.timeseries import TimeSeries
from pprint import pprint
import quandl
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.utils.validation import column_or_1d
import sklearn
import math
import plotly.plotly as py
import plotly.tools as tls
import plotly.graph_objs as go

# Plotly Key
tls.set_credentials_file(username = 'Enter Plotly username here', api_key = 'Enter Plotly API Key here')

# Fetching data for Exxon Mobil Corporation 

alpha_key = 'Enter Alphavantage API Key here'
ts = TimeSeries(key= alpha_key , output_format = 'pandas')
data_XOM, meta_data = ts.get_daily(symbol='XOM', outputsize='full')


# Fetching data for Chevron Corporation
ts = TimeSeries(key= alpha_key, output_format = 'pandas')
data_CVX, meta_data = ts.get_daily(symbol='CVX', outputsize='full')


# Fetching data for Occidental Petroleum
ts = TimeSeries(key= alpha_key, output_format = 'pandas')
data_OXY, meta_data = ts.get_daily(symbol='OXY', outputsize='full')
 

# Fetchng data for National Oilwell Varco 
ts = TimeSeries(key= alpha_key, output_format = 'pandas')
data_NOV, meta_data = ts.get_daily(symbol='NOV', outputsize='full')
 

# Fetching data for Halliburton Company
ts = TimeSeries(key= alpha_key, output_format = 'pandas')
data_HAL, meta_data = ts.get_daily(symbol='HAL', outputsize='full')


quandl_key = 'Enter Quandl API Key here'

# Fetching the crude oil prices using the auandl API 
quandl.ApiConfig.api_key = 'Enter Quandl API Key here'
quotes = quandl.get("CHRIS/CME_CL1", returns = 'pandas')

# print quotes

# Taking the recent one year worth of data into consideration

oil = quotes.iloc[-300:,3]
XOM = data_XOM.iloc[-300:,1]
CVX = data_CVX.iloc[-300:,1]
OXY = data_OXY.iloc[-300:,1]
NOV = data_NOV.iloc[-300:,1]
HAL = data_HAL.iloc[-300:,1]
oil.index =oil.index.map(lambda t: t.strftime('%Y-%m-%d'))

data_oil = []
data_XOM = []
data_CVX = []
data_OXY = []
data_NOV = []
data_HAL = []
lag_oil = []
lag_XOM = []
lag_CVX = []
lag_OXY = []
lag_NOV = []
lag_HAL = []
rms_oil = []
rms_lag = []

for i in range(0, len(oil)):
    data_oil.append([oil[i]])
    data_XOM.append([XOM[i]])
    data_CVX.append([CVX[i]])
    data_OXY.append([OXY[i]])
    data_NOV.append([NOV[i]])
    data_HAL.append([HAL[i]])

for i in range(1, len(oil)):
    lag_oil.append([oil[i]])
    lag_XOM.append([XOM[i]])
    lag_CVX.append([CVX[i]])
    lag_OXY.append([OXY[i]])
    lag_NOV.append([NOV[i]])
    lag_HAL.append([HAL[i]])

    
# Fitting Lasso Regression Model..

# Oil
    
# Splitting the data into training and test sets for building a learner     
    
from sklearn.cross_validation import train_test_split
train_oil, test_oil = train_test_split(data_oil, test_size=0.2, random_state=0)
train_lag_oil, test_lag_oil = train_test_split(lag_oil, test_size=0.2, random_state=0)

######################################################################################

# XOM
train_XOM, test_XOM = train_test_split(data_XOM, test_size=0.2, random_state=0)
train_lag_XOM, test_lag_XOM = train_test_split(lag_XOM, test_size=0.2, random_state=0)

# With multiplier alpha as 0.01 and tolerance as 0.00001
Lasso_XOM = linear_model.Lasso(alpha = 0.01, tol = 0.00001)

#Run Lasso regression for stock price vs oil price for the day
Lasso_XOM.fit(train_oil,train_XOM)
XOM_pred_oil = Lasso_XOM.predict(test_oil)

# Root mean squared error for oil
rms_oil.append(math.sqrt(mean_squared_error(test_XOM, XOM_pred_oil)))

#Run Lasso regression for stock price vs previous day price
train_XOM = train_XOM[:-1]
Lasso_XOM.fit(train_lag_XOM,train_XOM)
XOM_pred_prev = Lasso_XOM.predict(test_lag_XOM)

# Root mean squared error for previous day prices
rms_lag.append(math.sqrt(mean_squared_error(test_XOM, XOM_pred_prev)))


# Plotting using plotly

p1 = go.Scatter(x=oil.index[-60:], y = XOM_pred_oil, name='stock prediction from oil prices', line=dict(width=2))
p2 = go.Scatter(x=oil.index[-60:], y = XOM_pred_prev, name='stock prediction from previous day prices', line=dict(width=2))
p3 = go.Scatter(x=oil.index[-60:], y = test_XOM, name='Actual stock price', line=dict(width=2))
p = [p1,p2,p3]
layout = dict(title='Exxon Mobil Corporation', xaxis = dict(title = 'Days'), yaxis = dict(title = 'Stock price'))
fig_XOM = dict(data = p, layout = layout)
py.plot(fig_XOM, filename = 'XOM')


########################################################################################

# CVX
train_CVX, test_CVX = train_test_split(data_CVX, test_size=0.2, random_state=0)
train_lag_CVX, test_lag_CVX = train_test_split(lag_CVX, test_size=0.2, random_state=0)

# With multiplier alpha as 0.01 and tolerance as 0.00001
Lasso_CVX = linear_model.Lasso(alpha = 0.01, tol = 0.00001)

#Run Lasso regression for stock price vs oil price for the day
Lasso_CVX.fit(train_oil,train_CVX)
CVX_pred_oil = Lasso_CVX.predict(test_oil)

# Root mean squared error for oil
rms_oil.append(math.sqrt(mean_squared_error(test_CVX, CVX_pred_oil)))


#Run Lasso regression for stock price vs previous day price
train_CVX = train_CVX[:-1]
Lasso_CVX.fit(train_lag_CVX,train_CVX)
CVX_pred_prev = Lasso_CVX.predict(test_lag_CVX)

# Root mean squared error for previous day prices
rms_lag.append(math.sqrt(mean_squared_error(test_CVX, CVX_pred_prev)))


# Plotting using plotly
p1 = go.Scatter(x=oil.index[-60:], y = CVX_pred_oil, name='stock prediction from oil prices', line=dict(width=2))
p2 = go.Scatter(x=oil.index[-60:], y = CVX_pred_prev, name='stock prediction from previous day prices', line=dict(width=2))
p3 = go.Scatter(x=oil.index[-60:], y = test_CVX, name='Actual stock price', line=dict(width=2))
p = [p1,p2,p3]
layout = dict(title='Chevron Corporation', xaxis = dict(title = 'Days'), yaxis = dict(title = 'Stock price'))
fig_CVX = dict(data = p, layout = layout)
py.plot(fig_CVX, filename = 'CVX')


########################################################################################

# OXY
train_OXY, test_OXY = train_test_split(data_OXY, test_size=0.2, random_state=0)
train_lag_OXY, test_lag_OXY = train_test_split(lag_OXY, test_size=0.2, random_state=0)

# With multiplier alpha as 0.01 and tolerance as 0.00001
Lasso_OXY = linear_model.Lasso(alpha = 0.01, tol = 0.00001)
#Run Lasso regression for stock price vs oil price for the day
Lasso_OXY.fit(train_oil,train_OXY)
OXY_pred_oil = Lasso_OXY.predict(test_oil)

# Root mean squared error for oil
rms_oil.append(math.sqrt(mean_squared_error(test_OXY, OXY_pred_oil)))


#Run Lasso regression for stock price vs previous day price
train_OXY = train_OXY[:-1]
Lasso_OXY.fit(train_lag_OXY,train_OXY)
OXY_pred_prev = Lasso_OXY.predict(test_lag_OXY)

# Root mean squared error for previous day prices
rms_lag.append(math.sqrt(mean_squared_error(test_OXY, OXY_pred_prev)))

# Plotting using plotly
p1 = go.Scatter(x=oil.index[-60:], y = OXY_pred_oil, name='stock prediction from oil prices', line=dict(width=2))
p2 = go.Scatter(x=oil.index[-60:], y = OXY_pred_prev, name='stock prediction from previous day prices', line=dict(width=2))
p3 = go.Scatter(x=oil.index[-60:], y = test_OXY, name='Actual stock price', line=dict(width=2))
p = [p1,p2,p3]
layout = dict(title='Oxydental Petroleum', xaxis = dict(title = 'Days'), yaxis = dict(title = 'Stock price'))
fig_OXY = dict(data = p, layout = layout)
py.plot(fig_OXY, filename = 'OXY')


##########################################################################################
# NOV

train_NOV, test_NOV = train_test_split(data_NOV, test_size=0.2, random_state=0)
train_lag_NOV, test_lag_NOV = train_test_split(lag_NOV, test_size=0.2, random_state=0)

# With multiplier alpha as 0.01 and tolerance as 0.00001
Lasso_NOV = linear_model.Lasso(alpha = 0.01, tol = 0.00001)
#Run Lasso regression for stock price vs oil price for the day
Lasso_NOV.fit(train_oil,train_NOV)
NOV_pred_oil = Lasso_NOV.predict(test_oil)

# Root mean squared error for oil
rms_oil.append(math.sqrt(mean_squared_error(test_NOV, NOV_pred_oil)))

#Run Lasso regression for stock price vs previous day price
train_NOV = train_NOV[:-1]
Lasso_NOV.fit(train_lag_NOV,train_NOV)
NOV_pred_prev = Lasso_NOV.predict(test_lag_NOV)

# Root mean squared error for previous day prices
rms_lag.append(math.sqrt(mean_squared_error(test_NOV, NOV_pred_prev)))

# Plotting using plotly
p1 = go.Scatter(x=oil.index[-60:], y = NOV_pred_oil, name='stock prediction from oil prices', line=dict(width=2))
p2 = go.Scatter(x=oil.index[-60:], y = NOV_pred_prev, name='stock prediction from previous day prices', line=dict(width=2))
p3 = go.Scatter(x=oil.index[-60:], y = test_NOV, name='Actual stock price', line=dict(width=2))
p = [p1,p2,p3]
layout = dict(title='National Oilwell Varco', xaxis = dict(title = 'Days'), yaxis = dict(title = 'Stock price'))
fig_NOV = dict(data = p, layout = layout)
py.plot(fig_NOV, filename = 'NOV')


##########################################################################################

# HAL
train_HAL, test_HAL = train_test_split(data_HAL, test_size=0.2, random_state=0)
train_lag_HAL, test_lag_HAL = train_test_split(lag_HAL, test_size=0.2, random_state=0)

# With multiplier alpha as 0.01 and tolerance as 0.00001
Lasso_HAL = linear_model.Lasso(alpha = 0.01, tol = 0.00001)
#Run Lasso regression for stock price vs oil price for the day
Lasso_HAL.fit(train_oil,train_HAL)
HAL_pred_oil = Lasso_HAL.predict(test_oil)

# Root mean squared error for oil
rms_oil.append(math.sqrt(mean_squared_error(test_HAL, HAL_pred_oil)))

#Run Lasso regression for stock price vs previous day price
train_HAL = train_HAL[:-1]
Lasso_HAL.fit(train_lag_HAL,train_HAL)
HAL_pred_prev = Lasso_HAL.predict(test_lag_HAL)

# Root mean squared error for previous day prices
rms_lag.append(math.sqrt(mean_squared_error(test_HAL, HAL_pred_prev)))

# Plotting using plotly
p1 = go.Scatter(x=oil.index[-60:], y = HAL_pred_oil, name='stock prediction from oil prices', line=dict(width=2))
p2 = go.Scatter(x=oil.index[-60:], y = HAL_pred_prev, name='stock prediction from previous day prices', line=dict(width=2))
p3 = go.Scatter(x=oil.index[-60:], y = test_HAL, name='Actual stock price', line=dict(width=2))
p = [p1,p2,p3]
layout = dict(title='Halliburton Company', xaxis = dict(title = 'Days'), yaxis = dict(title = 'Stock price'))
fig_HAL = dict(data = p, layout = layout)
py.plot(fig_HAL, filename = 'HAL')


# Plotting the Root Mean Square Errors for Predictions 

names = ['Exxon Mobil Corporation', 'Chevron Corporation', 'Occidental Petroleum', 'National Oilwell Varco', 
         'Halliburton Company']

mean_oil = np.mean(rms_oil)
mean_lag = np.mean(rms_lag)

trace1 = go.Bar(x = names, y = rms_oil, name = 'RMS error for oil')
trace2 = go.Bar(x = names, y = rms_lag, name = 'RMS error for previous day')
b = [trace1, trace2]
layout = go.Layout(title = 'RMS Error', barmode = 'group')
fig_bar = go.Figure(data=b, layout = layout)
py.plot(fig_bar, filename = 'RMS Error')



