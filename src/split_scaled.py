import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def double_split_scale(X,y):
    """spliting the data in two steps into train, test and validation datasets.

    Args:
    inputs:
        X: the vector of features
        y: the series of the response variable
        
    outputs:
        X_t_scaled , y_t: the data that we use for training the models
        X_v_scaled, y_v: the data that we use to validate our models.
        X_test_scaled, y_test: the data that we use to report the performance of the model.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=12, stratify=y)
    X_t, X_v, y_t, y_v = train_test_split(X_train, y_train, test_size=0.20, random_state=12, stratify=y_train)
    
    scalar = StandardScaler()
    X_t_scaled = X_t.copy()
    
    numerics = ['age', 'campaign', 'previous', 'emp.var.rate',
                'cons.price.idx','cons.conf.idx', 'euribor3m', 'nr.employed']
    categoricals = ['job', 'marital', 'education', 'default', 'housing',
                        'loan', 'contact', 'month', 'day_of_week', 'poutcome']
    
    X_t_scaled[numerics] = scalar.fit_transform(X_t_scaled[numerics])
    
    X_v_scaled = X_v.copy()
    X_v_scaled[numerics] = scalar.transform(X_v_scaled[numerics])
    
    X_test_scaled = X_test.copy()
    X_test_scaled[numerics] = scalar.transform(X_test_scaled[numerics])
    
    
    return X_t_scaled, X_v_scaled, X_test_scaled
    
    
    
    
    