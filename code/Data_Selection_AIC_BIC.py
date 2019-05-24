import itertools
import time
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from tqdm import tnrange, tqdm_notebook


#Read data
SM = pd.read_csv("Cleaned_Data.csv", sep=',', header=0)
SM = SM[SM.AccessCost < 100000]

def createdummy(X, col, n):
    for i in range(n):
        dummy = col
        dummy += str(i + 1)
        X[dummy] = 0
        X[dummy][X[col] == i + 1] = 1
    return

def change2to0(X, col):
    X[col][X[col] == 2] = 0
    return
    
        
X = SM
X['Age'][X['Age'] < 50] = 0
X['Age'][X['Age'] >= 50] = 1



createdummy(X, 'ProvinceResidence', 8)
createdummy(X, 'Destination', 4)
createdummy(X, 'DepartureTime', 4)
createdummy(X, 'SeatClass', 3)
createdummy(X, 'Occupation', 12)
createdummy(X, 'Income', 7)
change2to0(X, 'Gender')
change2to0(X, 'GroupTravel')
change2to0(X, 'Airport')
createdummy(X, 'Airline', 4)
"""
createdummy(X, 'Nationality', 5)
createdummy(X, 'TripPurpose', 4)
createdummy(X, 'ModeTransport', 11)
"""
X['Korean'] = 1
X['Korean'][X['Nationality'] != 1] = 0

X['PleasureTrip'] = 1
X['PleasureTrip'][X['TripPurpose'] != 1] = 0

X['PublicTrans'] = 1
X['PublicTrans'][X['ModeTransport'] == 1] = 0
X['PublicTrans'][X['ModeTransport'] == 2] = 0
X['PublicTrans'][X['ModeTransport'] == 9] = 0
X['PublicTrans'][X['ModeTransport'] == 10] = 0
X['PublicTrans'][X['ModeTransport'] == 11] = 0
X['PublicTrans'][X['ModeTransport'] == 5] = 0
X['const'] = 1


#X = X.drop(['Airline', 'ID', 'Nationality', 'TripPurpose', 'ProvinceResidence', 'Destination', 'DepartureTime', 'SeatClass', 'ModeTransport', 'Occupation', 'Income', 'MileageAirline', 'FrequentDestination7','FrequentDestination6','FrequentDestination5','FrequentDestination4','FrequentDestination3','FrequentDestination2','FrequentDestination1',], axis = 1)  
#X = X.drop(['ID', 'Nationality', 'TripPurpose', 'ProvinceResidence', 'Destination', 'DepartureTime', 'SeatClass', 'ModeTransport', 'Occupation', 'Income', 'MileageAirline', 'FrequentDestination7','FrequentDestination6','FrequentDestination5','FrequentDestination4','FrequentDestination3','FrequentDestination2','FrequentDestination1',], axis = 1)  
X = X.drop(['Airline', 'ID', 'Nationality', 'TripPurpose', 'ProvinceResidence', 'Destination', 'DepartureTime', 'SeatClass', 'ModeTransport', 'Occupation', 'Income', 'MileageAirline', 'FrequentDestination7','FrequentDestination6','FrequentDestination5','FrequentDestination4','FrequentDestination3','FrequentDestination2','FrequentDestination1',], axis = 1)  


X.to_csv("X.csv")
Y = X['Airport']
#Y1 = X['Airline']
#X2 = X.drop(['Airline'], axis=1)
X1 = X.drop(['Airport'], axis = 1)




#new = SM[['Age', 'TripDuration', 'FlyingCompanion', 'NoTripsLastYear', 'TotalDepartureHr', 'Airfare', 'NoTransport', 'AccessCost', 'AccessTime']]
"""

model = sm.Logit(Y, X1.astype(float)).fit()	
predictions = model.predict(X1) 
print_model = model.summary()
print(print_model)
"""

X2 = X1[['FlyingCompanion', 'NoTransport', 'AccessCost','AccessTime','FrequentFlightDestination1','FrequentFlightDestination2','FrequentFlightDestination3','FrequentFlightDestination4','FrequentFlightDestination5','FrequentFlightDestination6', 'FrequentFlightDestination7','Korean']]
"""
X3 = X2[['PleasureTrip', 'Korean', 'NoTransport', 'Airfare', 'TotalDepartureHr', 'FrequentFlightDestination7', 'FrequentFlightDestination6', 'FrequentFlightDestination5','FrequentFlightDestination3','FlyingCompanion', 'Age', 'Airport', 'FrequentFlightDestination2', 'FrequentFlightDestination1', 'FrequentFlightDestination4']]
"""
def fit_log_reg(X,Y):
    #Fit linear regression model and return RSS and R squared values
    X = sm.add_constant(X)
    model_k = sm.Logit(Y, X.astype(float)).fit()  
    AIC = model_k.aic
    BIC = model_k.bic
    LLH = model_k.llf
    R2 = model_k.prsquared
    return AIC, BIC, LLH,R2
    
def MNfit_log_reg(X,Y):
    #Fit linear regression model and return RSS and R squared values
    X = sm.add_constant(X)
    model_k = sm.MNLogit(Y, X.astype(float)).fit()  
    AIC = model_k.aic
    BIC = model_k.bic
    LLH = model_k.llf
    R2 = model_k.prsquared
    return AIC, BIC, LLH,R2


#funcion of tring all combination to find the best model
def try_all(X,Y):
    
    AIC_list, BIC_list,Log_list,RSquare_list, feature_list = [],[], [],[],[]
    numb_features = []

    
    for i in range(len(X.columns)):
        for j in range(i+1, len(X.columns)+1):
            tmp_result = MNfit_log_reg(X.iloc[:,i:j],Y)   
            AIC_list.append(tmp_result[0])                  
            BIC_list.append(tmp_result[1])
            Log_list.append(tmp_result[2])
            RSquare_list.append(tmp_result[3])
            feature_list.append(X.iloc[:,i:j].columns.values)
            numb_features.append(j-i) 
    #Store in DataFrame
    df = pd.DataFrame({'numb_features': numb_features,'AIC': AIC_list, 'BIC':BIC_list,'Loglikelihood':Log_list,'McFaddens R2':RSquare_list,'features':feature_list})
    return(df)
    
"""
model_k = sm.MNLogit(Y1, X2.astype(float)).fit()
print(model_k.summary())
"""

resutl = try_all(X2,Y)
print(resutl.nsmallest(3, ['AIC', 'BIC']))
print(resutl.nlargest(3, ['Loglikelihood', 'McFaddens R2']))

