from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def logistic_fit(X_t, y_t):
 
    LR1=LogisticRegression(C=1.9655172413793105, class_weight='balanced',
                   max_iter=1000000, penalty='l1', random_state=12,
                   solver='saga')
    pipe = Pipeline([('scaler', StandardScaler()), ('LR', LR1)])
    print('logistic regression is done on the training set')
    pipe.fit(X_t, y_t)
    joblib.dump(pipe, 'LR.joblib')
    
    
    
def logistic_predictor(s):
    LR2 = joblib.load('LR.joblib')
    pred_lr2 = LR2.predict(s)
    return pred_lr2

def logistic_predict_proba(s):
    LR2 = joblib.load('LR.joblib')
    pred_proba = LR2.predict_proba(s)[:,1]
    return pred_proba