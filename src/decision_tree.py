import numpy as np
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier



def dt_fit(X_t_scaled, y_t):
    tree_best = DecisionTreeClassifier(class_weight='balanced', max_depth=3,
                       max_features='auto', random_state=12)
    tree_best.fit(X_t_scaled, y_t)
    joblib.dump(tree_best, 'tree_best.joblib')
    
def dt_predictor(X):
    tree_best = joblib.load('tree_best.joblib')
    pred = tree_best.predict(X)
    return pred
    
def dt_predict_proba(X):
    tree_best = joblib.load('tree_best.joblib')
    pred_proba = tree_best.predict_proba(X)[:,1]
    return pred_proba