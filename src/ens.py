import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import VotingClassifier

def ensemble_fit(X_t_scaled, y_t):
    LR2 = joblib.load('LR.joblib')
    svc = joblib.load('svc.joblib')
    tree_best = joblib.load('tree_best')
    vot = VotingClassifier(
    estimators=[('lr',LR2), ('svc', svc), ('tree', tree_best)],
    voting = 'hard')
    vot.fit(X_t_scaled, y_t)
    joblib.dump(vot, 'vot.joblib')
    
def ensemble_predictor(X):
    vot = joblib.load('vot.joblib')
    pred = vot.predict(X)
    return pred
    
    
def ensemble_predict_proba(X):
    vot = joblib.load('vot.joblib')
    pred_proba = vot.predict_proba(X)[:,1]
    return pred_proba