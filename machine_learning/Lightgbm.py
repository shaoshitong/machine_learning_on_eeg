import lightgbm as lgb
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
    data_type="DEAP_V"
    if data_type=="SEED":
        total = 15
    else:
        total=32
    print('设置参数')
    params = {
        'objective': 'multiclass',  # 目标函数
        'metric':['multi_error','multiclass'],
        'num_thread':8,
        'device':"gpu",
        'learning_rate': 0.1,
        'num_leaves': 100,
        'min_data_in_leaf':1,
        'max_depth': 3,
        # 'max_bin':8,
        'num_class': 3 if data_type=="SEED" else 2,
        'lambda_l1': 0.01,
        'lambda_l2': 0.01,
    }
    total_acc=[]
    for num_leaves in range(20, 100, 20):
        for max_depth in range(3, 7, 1):
            params['num_leaves'] = num_leaves
            params['max_depth'] = max_depth
            acc = []
            for i in range(total):
                X_train, y_train, X_test, y_test = get_data(data_type, i)
                y_train=y_train.astype(int)
                y_test=y_test.astype(int)
                lgb_train = lgb.Dataset(X_train, y_train,free_raw_data=True)
                lgb_eval = lgb.Dataset(X_test,y_test,reference=True)
                print("开始训练")
                gbm = lgb.train(params,  # 参数字典
                                lgb_train,  # 训练集
                                num_boost_round=200,  # 迭代次数
                                verbose_eval=1)
                y_pred = gbm.predict(X_test)
                y_pred = [list(x).index(max(x)) for x in y_pred]
                test_acc=accuracy_score(y_test, y_pred)
                print(f"第{i}折的准确率是{test_acc}")
                acc.append(test_acc)
            acc=np.array(acc)
            print("="*30+f" {total}折测试结束准确率报告 "+"="*30)
            print(f"最大深度是 {max_depth}，叶子数量是 {num_leaves}")
            print(f"准确率的标准差是 {acc.std(axis=0)}")
            print(f"准确率的均值是 {acc.mean(axis=0)}")
            total_acc.append([num_leaves,max_depth,acc.std(axis=0),acc.mean(axis=0)])
    print(total_acc)
    total_acc=sorted(total_acc,key=lambda x:x[3],reverse=True)
    print("=====================result==============================")
    print(total_acc[0])
    print(total_acc)