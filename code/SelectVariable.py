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


def fit_log_reg(X,Y, multi):
    #Fit linear regression model and return AIC, BIC, loglikelihood, and R squared values
    X = sm.add_constant(X)
    model_k = sm.Logit(Y, X.astype(float)).fit()
    if(multi == True):
        model_k=sm.MNLogit(Y, X.astype(float)).fit()    
    AIC = model_k.aic
    BIC = model_k.bic
    LLH = model_k.llf
    R2 = model_k.prsquared
    return AIC, BIC, LLH,R2

#funcion of tring all combination to find the best model
def try_all(dat,  Yname, multi):
    #Initialization variables
    Y = dat[Yname]#dat["Airport"]
    if(Yname =="Airport"):
        Y = Y.replace(regex={1:0, 2:1})
    X = dat.drop(columns = Yname, axis = 1)
    #k = 11
    AIC_list, BIC_list,Log_list,RSquare_list, feature_list = [],[], [],[],[]
    numb_features = []

    #Looping over k = 1 to k snip= 11 features in X
    for k in tnrange(1,len(X.columns) + 1, desc = 'Loop...'):
        #Looping over all possible combinations: from 11 choose k
        for combo in itertools.combinations(X.columns,k):
            tmp_result = fit_log_reg(X.loc[:,list(combo)],Y,multi)   #Store temp result 
            AIC_list.append(tmp_result[0])                  #Append lists
            BIC_list.append(tmp_result[1])
            Log_list.append(tmp_result[2])
            RSquare_list.append(tmp_result[3])
            feature_list.append(combo)
            numb_features.append(len(combo))  
    
    #Store in DataFrame
    df = pd.DataFrame({'numb_features': numb_features,'AIC': AIC_list, 'BIC':BIC_list,'Loglikelihood':Log_list,'McFaddens R2':RSquare_list,'features':feature_list})
    return(df)
