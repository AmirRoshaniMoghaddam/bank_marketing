import numpy as np
import pandas as pd
from sklearn.svm import SVC
import joblib

def svc_fit(X_t_scaled, y_t):
    svc = SVC(random_state=12, class_weight='balanced')
    svc.fit(X_t_scaled, y_t)
    joblib.dump(svc, 'svc.joblib')
    
    
    
def svc_predictor(X):
    svc = joblib.load('svc.joblib')
    pred = svc.predict(X)
    return pred

def svc_predict_proba(X):
    svc = joblib.load('svc.joblib')
    pred_proba = svc.predict_proba(X)[:,1]
    return pred_proba