from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, balanced_accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import joblib


def predictor(X):
    X = X.values.reshape(1,-1)
    LR2 = joblib.load('LR.joblib')
    pred_lr2 = LR2.predict(X)
    return pred_lr2