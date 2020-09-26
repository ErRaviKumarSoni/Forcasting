import pandas as pd
data = pd.read_excel('D:\\LIVE WIRE - DS\\Data Set\\datasetShampoo.xlsx',index_col=0)
data.head()


import matplotlib.pyplot as plt
plt.plot(data)
plt.show()


data.reset_index(inplace=True)
data.index = pd.to_datetime(data.index)


from statsmodels.tsa.seasonal import seasonal_decompose    #trend, seasonality, residual(error/noise)
result = seasonal_decompose(data.Sales, model='additive',freq=12)
result.plot()

#install pip install pyramid-arima for it (support visual basic studio c++)
#from pyramid.arima import auto_arima
from pmdarima.arima import auto_arima
stepwiseM = auto_arima(data.Sales, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
print(stepwiseM.aic())

import numpy as np
data=np.array(data)

train=data[1:30,:]
test=data[30:,:]

dataTrain = pd.DataFrame({'Date':train[:,0],'Sales':train[:,1]}) 
dataTest = pd.DataFrame({'Date':test[:,0],'Sales':test[:,1]}) 

dataTrain.reset_index(inplace=True)
dataTrain.index = pd.to_datetime(dataTrain.index)
stepwiseM.fit(dataTrain.Sales)
predR=stepwiseM.predict(n_periods=6)
print("the six step prdicted result by arima is: ",predR)
