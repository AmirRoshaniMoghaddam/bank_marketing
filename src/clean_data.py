import pandas as pd
import numpy as np

def read_data(data_path):
    """reading the data

    Args:
        data_path (_type_): path to the input data, csv
    """
    data = pd.read_csv(data_path,sep = ';')
    return data

def clean_data(df):
    """
    clean the data
    
    Args:
    
    inputs:
        df: pd.DataFrame
    
    Returns:
        X: features, pd.DataFame
        y: target, pd.PandasSeries
    """
    df = df.drop_duplicates()
    # Creating lists for numeical and catergorical variables.

    numerics = ['age', 'campaign', 'previous', 'emp.var.rate',
                'cons.price.idx','cons.conf.idx', 'euribor3m', 'nr.employed']
    categoricals = ['job', 'marital', 'education', 'default', 'housing',
                        'loan', 'contact', 'month', 'day_of_week', 'poutcome']
    
    df = df.drop('duration', axis = 1) # duration is known only after the call, it must be removed 
    
    # Creating lists for numeical and catergorical variables.

    numerics = ['age', 'campaign', 'previous', 'emp.var.rate',
                'cons.price.idx','cons.conf.idx', 'euribor3m', 'nr.employed']
    categoricals = ['job', 'marital', 'education', 'default', 'housing',
                        'loan', 'contact', 'month', 'day_of_week', 'poutcome']
    
    df['y'] = np.where(df['y'] == 'yes', 1, 0)
    df['campaign'] = np.where(df['campaign'] > 9, 9, df['campaign'])
    
    df = pd.get_dummies(df, columns=categoricals, drop_first=True)
    
    # creating vectors of independent vars and dependent var.
    X = df.drop('y', axis = 1) 
    y = df.y
    
    return X,y