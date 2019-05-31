from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection  import train_test_split
from sklearn import metrics
import statsmodels.api as sm
import sklearn.linear_model as lm
from sklearn.model_selection import KFold
import numpy as np

#k-fold validation Airline
airline = pd.read_csv("Airline_Dataset.csv", sep=',', header=0)
airline = airline.drop(['FrequentFlightDestination7'], axis = 1)
airline['intercept'] = 1
y = airline['Airline']
X = airline.drop(['Airline'], axis = 1)

kf = KFold(n_splits=6)
kf.get_n_splits(X)


accuracy = []
precision = []
for train_index, test_index in kf.split(X):
    
    train_first = train_index[0]
    train_last = train_index[-1] + 1
    test_first = test_index[0]
    test_last = test_index[-1] + 1
    
    X_train = X[train_first : train_last]
    y_train = y[train_first : train_last]
    X_test = X[test_first : test_last]
    y_test = y[test_first : test_last]
    
        
    clf = lm.LogisticRegression(random_state=0, multi_class='multinomial', solver='newton-cg')
    model = clf.fit(X_train, y_train.astype('int'))
    y_pred = model.predict(X_test)
    accuracy.append(metrics.accuracy_score(y_test, y_pred))
print(np.mean(accuracy))


##k-fold validation Airport
airport = pd.read_csv("Airport_Dataset.csv", sep=',', header=0)
airport = airport.drop(['FrequentFlightDestination7'], axis = 1)
airport['intercept'] = 1
y1 = airport['Airport']
X1= airport.drop(['Airport'], axis = 1)

kf1 = KFold(n_splits=6)
kf1.get_n_splits(X1)


accuracy_airport = []
precision_airport = []
recall_airport = []
for train_index, test_index in kf.split(X1):
    
    train_first = train_index[0]
    train_last = train_index[-1] + 1
    test_first = test_index[0]
    test_last = test_index[-1] + 1
    
    X1_train = X1[train_first : train_last]
    y1_train = y1[train_first : train_last]
    X1_test = X1[test_first : test_last]
    y1_test = y1[test_first : test_last]

    logit_model = sm.Logit(y1_train, X1_train).fit()
    predictions_test = logit_model.predict(X1_test)
    predictions_test[predictions_test.astype(float) >= 0.5] = 1
    predictions_test[predictions_test != 1] = 0
    accuracy_airport.append(metrics.accuracy_score(y1_test, predictions_test))
    precision_airport.append(metrics.precision_score(y1_test, predictions_test))
    recall_airport.append(metrics.recall_score(y1_test, predictions_test))

print(np.mean(accuracy_airport))