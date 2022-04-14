from machine_learning.dataset import get_data
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

def accuracy_score(truth, pred):
    """ Returns accuracy score for input truth and predictions. """
    if len(truth) == len(pred):
        return (truth == pred).mean() * 100
    else:
        return 0

if __name__ =="__main__":
    data_type="DEAP_A"
    if data_type=="SEED":
        total = 15

        mean_acc=0.
        param_grid = {"criterion":["gini", "entropy"],
                      "max_depth":[1,3,5],
                      "max_features": ["auto","sqrt","log2"],
                      "n_estimators":[10,50,100]
                      }

        X_train,y_train,X_test,y_test=get_data(data_type,0)
        X=np.concatenate([X_train,X_test],axis=0)
        y=np.concatenate([y_train,y_test],axis=0).astype(int)
        kf = KFold(n_splits=total)
        svm=RandomForestClassifier()
        scoring_fnc = make_scorer(accuracy_score)
        grid=GridSearchCV(svm,param_grid=param_grid,cv=kf,n_jobs=8,verbose=3)
        grid.fit(X, y)
        print("最好的准确率:",grid.best_score_)
        print("最好的参数:",grid.best_params_)
        print("每次交叉验证后的准确率结果：\n", grid.cv_results_)
        """
        best acc: 0.7807407407407408
        best params: {'C': 0.025, 'decision_function_shape': 'ovr', 'gamma': 'scale', 'kernel': 'linear'}
        best index: 1
        """
    else:
        total = 32

        mean_acc=0.
        param_grid = {"criterion":["gini", "entropy"],
                      "max_depth":[1,3,5],
                      "max_features": ["auto","sqrt","log2"],
                      "n_estimators":[10,50,100]
                      }

        X_train,y_train,X_test,y_test=get_data(data_type,0)
        X=np.concatenate([X_train,X_test],axis=0)
        y=np.concatenate([y_train,y_test],axis=0).astype(int)
        kf = KFold(n_splits=total)
        svm=RandomForestClassifier()
        scoring_fnc = make_scorer(accuracy_score)
        grid=GridSearchCV(svm,param_grid=param_grid,cv=kf,n_jobs=8,verbose=3)
        grid.fit(X, y)
        print("最好的准确率:",grid.best_score_)
        print("最好的参数:",grid.best_params_)
        print("每次交叉验证后的准确率结果：\n", grid.cv_results_)






