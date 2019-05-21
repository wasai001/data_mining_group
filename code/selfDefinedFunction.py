import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Pre-define Functions
def sumMissing(df):
    col_name_1 = str(df)+"Count"
    col_name_2 = str(df)+"Percentage"
    countMissing = df.isnull().sum()
    percentMissing = df.isnull().mean().round(4) * 100
    result = pd.concat([countMissing, percentMissing], axis=1)
    #result = result.rename(columns={0: col_name_1, 1: col_name_2})
    return(result)

def horzBarPlot(df,x,y):
    a4_dims = (11.7, 8.27)
    fig, ax = plt.subplots(figsize=a4_dims)
    ax = sns.barplot(x=x, y=y, data=df)

def compareDF(df1,df2):
    return(df1.join(df2.set_index('index'), on='index'))

def fillAirfare(df,separateCol,groupCol):
    # Split the dataframe
    nullAirfare = df[df['Airfare'].isnull()]
    df_splited = df[df['Airfare'].notnull()]
    
    # Calculate the median airfare
    medianAirfare = df[separateCol].groupby(groupCol).median().reset_index()
    
    # Merge the missing airfare with the median airfare
    fillMissingAirfare = pd.merge(nullAirfare, medianAirfare,  how='left', left_on=groupCol, right_on = groupCol)
    fillMissingAirfare = fillMissingAirfare.drop(columns=['Airfare_x'])
    fillMissingAirfare = fillMissingAirfare.rename(columns={'Airfare_y': 'Airfare'})
    
    df = df_splited.append(fillMissingAirfare,sort=False)
    df = df.sort_values('ID')
    df = df.reset_index()
    df = df.drop(columns = ['index'])
    return(df)

def fillAccessCost(df,separateCol,groupCol):
    # Split the dataframe
    nullAccessCost = df[df['AccessCost'].isnull()]
    df_splited = df[df['AccessCost'].notnull()]
    # Calculate the median access cost
    medianAccessCost = df[separateCol].groupby(groupCol).median().reset_index()
    # Merge the missing access cost with the median access cost
    fillMissingAccessCost = pd.merge(nullAccessCost, medianAccessCost,  how='left', left_on=groupCol, right_on=groupCol)
    fillMissingAccessCost = fillMissingAccessCost.drop(columns=['AccessCost_x'])
    fillMissingAccessCost = fillMissingAccessCost.rename(columns={'AccessCost_y': 'AccessCost'})
    
    df = df_splited.append(fillMissingAccessCost,sort=False)
    df = df.sort_values('ID')
    df = df.reset_index()
    df = df.drop(columns = ['index'])
    return(df)

def fillAccessTime(df,separateCol,groupCol):
    # Split the dataframe
    nullAccessTime = df[df['AccessTime'].isnull()]
    df_splited = df[df['AccessTime'].notnull()]
    # Calculate the median access Time
    medianAccessTime = df[separateCol].groupby(groupCol).median().reset_index()
    # Merge the missing access Time with the median access Time
    fillMissingAccessTime = pd.merge(nullAccessTime, medianAccessTime,  how='left', left_on=groupCol, right_on=groupCol)
    fillMissingAccessTime = fillMissingAccessTime.drop(columns=['AccessTime_x'])
    fillMissingAccessTime = fillMissingAccessTime.rename(columns={'AccessTime_y': 'AccessTime'})
    
    df = df_splited.append(fillMissingAccessTime,sort=False)
    df = df.sort_values('ID')
    df = df.reset_index()
    df = df.drop(columns = ['index'])
    return(df)

def fillIncome(df,separateCol,groupCol):
    # Split the dataframe
    nullIncome = df[df['Income'].isnull()]
    df_splited = df[df['Income'].notnull()]
    # Calculate the median access Time
    medianIncome = df[separateCol].groupby(groupCol).median().reset_index()
    # Merge the missing access Time with the median access Time
    fillMissingIncome = pd.merge(nullIncome, medianIncome,  how='left', left_on=groupCol, right_on=groupCol)
    fillMissingIncome = fillMissingIncome.drop(columns=['Income_x'])
    fillMissingIncome = fillMissingIncome.rename(columns={'Income_y': 'Income'})
    
    df = df_splited.append(fillMissingIncome,sort=False)
    df = df.sort_values('ID')
    df = df.reset_index()
    df = df.drop(columns = ['index'])
    return(df)