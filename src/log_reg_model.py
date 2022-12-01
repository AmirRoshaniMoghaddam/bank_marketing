from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, balanced_accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import joblib

def logistic_fit(X_t_scaled, y_t):
    
    LR2=LogisticRegression(random_state= 12 , max_iter=1_000_000, class_weight = 'balanced')
    print('logistic regression is done on the training set')
    LR2.fit(X_t_scaled, y_t)
    joblib.dump(LR2, 'LR.joblib')
    
    
    
def logistic_predictor(s):
    LR2 = joblib.load('LR.joblib')
    pred_lr2 = LR2.predict(s)
    return pred_lr2

def logistic_predict_proba(s):
    LR2 = joblib.load('LR.joblib')
    pred_proba = LR2.predict_proba(s)[:,1]
    return pred_proba