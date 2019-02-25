### 网格搜索训练lgb模型
import pandas as pd
import numpy as np
import time
import datetime
import warnings
import lightgbm as lgb
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings("ignore")

lgb_train = lgb.Dataset('./data/train.bin')
lgb_val = lgb.Dataset('./data/val.bin')

parameters = {
    'max_depth': [15, 20, 25, 30, 35],
    'max_depth':[5,6],
    'learning_rate': [0.01, 0.02, 0.05, 0.1],
    'feature_fraction': [0.6, 0.7, 0.8, 0.9, 0.95],
    'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 0.95],
    'bagging_freq': [2, 4, 5, 6, 8],
    'lambda_l1': [0, 0.1, 0.4, 0.5, 0.6],
    'lambda_l2': [0, 10, 15, 35, 40],
    'cat_smooth': [1, 10, 15, 20, 35]
}
gbm = lgb.LGBMClassifier(boosting_type='gbdt',
                         objective = 'binary',
                         metric = 'auc',
                         verbose = 0,
                         learning_rate = 0.01,
                         num_leaves = 35,
                         feature_fraction=0.8,
                         bagging_fraction= 0.9,
                         bagging_freq= 8,
                         lambda_l1= 0.6,
                         lambda_l2= 0)
# 有了gridsearch我们便不需要fit函数
gsearch = GridSearchCV(gbm, param_grid=parameters, scoring='accuracy', cv=3)
gsearch.fit(train_x, train_y)

print("Best score: %0.3f" % gsearch.best_score_)
print("Best parameters set:")
best_parameters = gsearch.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))


# 过拟合
print("调参2：max_bin,min_data_in_leaf...")
# 建议使用较小的 max_bin来获得更快的速度,
# 工具箱的最大数特征值决定了容量,工具箱的最小数特征值可能会降低训练的准确性, 但是可能会增加一些一般的影响
for max_bin in [40, 50]:
    for min_data_in_leaf in [10,20]:
        params['max_bin'] = max_bin
        params['min_data_in_leaf'] = min_data_in_leaf

        cv_results = lgb.cv(
            params,
            lgb_train,
            seed=2018,
            nfold=3,
            metrics=['auc'],
            early_stopping_rounds=10,
            verbose_eval=False
        )

        mean_merror = pd.Series(cv_results['auc-mean']).min()
        boost_rounds = pd.Series(cv_results['auc-mean']).idxmin()

        if mean_merror < min_merror:
            min_merror = mean_merror
            best_params['max_bin'] = max_bin
            best_params['min_data_in_leaf'] = min_data_in_leaf

params['min_data_in_leaf'] = best_params['min_data_in_leaf']
params['max_bin'] = best_params['max_bin']
print(best_params)
print('调参2：over')

print("feature_fraction,bagging_fraction,bagging_freq...")
for feature_fraction in [0.7]:
    for bagging_fraction in [0.7, 0.8]:
        for bagging_freq in [2,4,6]:
            params['feature_fraction'] = feature_fraction
            params['bagging_fraction'] = bagging_fraction
            params['bagging_freq'] = bagging_freq

            cv_results = lgb.cv(
                params,
                lgb_train,
                seed=2018,
                nfold=3,
                metrics=['auc'],
                early_stopping_rounds=10,
                verbose_eval=False
            )

            mean_merror = pd.Series(cv_results['auc-mean']).min()
            boost_rounds = pd.Series(cv_results['auc-mean']).idxmin()

            if mean_merror < min_merror:
                min_merror = mean_merror
                best_params['feature_fraction'] = feature_fraction
                best_params['bagging_fraction'] = bagging_fraction
                best_params['bagging_freq'] = bagging_freq

params['feature_fraction'] = best_params['feature_fraction']
params['bagging_fraction'] = best_params['bagging_fraction']
params['bagging_freq'] = best_params['bagging_freq']
print(best_params)
print('调参3：over')

print("调参4：lambda_l1,lambda_l2...")
for lambda_l1 in [1e-5, 1e-2, 0.1, 1,2,2.5,3]:
    for lambda_l2 in [1e-5, 1e-2, 0.1, 1,2,2.5,3]:
        params['lambda_l1'] = lambda_l1
        params['lambda_l2'] = lambda_l2

        cv_results = lgb.cv(
            params,
            lgb_train,
            seed=2018,
            nfold=3,
            metrics=['auc'],
            early_stopping_rounds=10,
            verbose_eval=False
        )

        mean_merror = pd.Series(cv_results['auc-mean']).min()
        boost_rounds = pd.Series(cv_results['auc-mean']).idxmin()

        if mean_merror < min_merror:
            min_merror = mean_merror
            best_params['lambda_l1'] = lambda_l1
            best_params['lambda_l2'] = lambda_l2

params['lambda_l1'] = best_params['lambda_l1']
params['lambda_l2'] = best_params['lambda_l2']
print('调参5：over')

print("调参5：learning_rate")
for learning_rate in [0.01, 0.02, 0.2]:
    params['learning_rate'] = learning_rate

    cv_results = lgb.cv(
        params,
        lgb_train,
        seed=2018,
        nfold=3,
        metrics=['auc'],
        early_stopping_rounds=10,
        verbose_eval=False
    )

    mean_merror = pd.Series(cv_results['auc-mean']).min()
    boost_rounds = pd.Series(cv_results['auc-mean']).idxmin()

    if mean_merror < min_merror:
        min_merror = mean_merror
        best_params['learning_rate'] = learning_rate

params['learning_rate'] = best_params['learning_rate']
print('调参5：over')

print('最佳参数为：')
print(best_params)

# # =======
# # 本次搜索出来的最优参数
# params = {
#     'boosting_type': 'gbdt',
#     'objective': 'binary',
#     'metric': 'auc',
#     'num_thread': -1,
#     'n_jobs': -1,
#     'num_leaves':50,
#     'max_depth':1,
#     'max_bin':40,
#     'min_data_in_leaf':20,
#     'feature_fraction': 0.7,
#     'bagging_fraction': 0.8,
#     'bagging_freq': 1,
#     'lambda_l1': 0.0,
#     'lambda_l2': 0.0
# }
# # =======

# 训练

model = lgb.train(
    params,
    lgb_train,
    valid_sets=lgb_val,
    early_stopping_rounds=30
)
model.save_model('./model/model_4.txt',num_iteration=model.best_iteration)
