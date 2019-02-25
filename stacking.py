### stacking demo (非原创)
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

def get_stacking(clf, x_train, y_train, x_test, n_folds=5):
    """
    这个函数是stacking的核心，使用交叉验证的方法得到次级训练集
    x_train, y_train, x_test 的值应该为numpy里面的数组类型 numpy.ndarray .
    如果输入为pandas的DataFrame类型则会把报错"""
    train_num, test_num = x_train.shape[0], x_test.shape[0]
    second_level_train_set = np.zeros((train_num,))
    second_level_test_set = np.zeros((test_num,))
    test_nfolds_sets = np.zeros((test_num, n_folds))
    
    kf = KFold(n_splits=n_folds)

    for i,(train_index, test_index) in enumerate(kf.split(x_train)):
        x_tra, y_tra = x_train[train_index], y_train[train_index]
        x_tst, y_tst =  x_train[test_index], y_train[test_index]

        clf.fit(x_tra, y_tra)

        second_level_train_set[test_index] = clf.predict(x_tst)
        test_nfolds_sets[:,i] = clf.predict(x_test)

    second_level_test_set[:] = test_nfolds_sets.mean(axis=1)
    return second_level_train_set, second_level_test_set

# 我们这里使用5个分类算法，为了体现stacking的思想，就不加参数了
lgb_model_1 = lgb.Booster(model_file='./lgb_model_1.txt')
lgb_model_2 = lgb.Booster(model_file='./lgb_model_2.txt')
lgb_model_3 = lgb.Booster(model_file='./lgb_model_3.txt')

### test 读取方法
print('reading testset...')
test_df =  pd.read_csv('./test_final_data.csv')
### train 读取方法
print('reading trainset...')
train_df =  pd.read_csv('./train_final_data.csv',index_col=0)
train_shape = train_df.shape[0]

target = 'whether_to_keep'

print('divide train_x,train_y...')
train_df_X = train_df.drop(target,1)
train_df_y = train_df[target]

# 最终数据
print('concating all the data...')
all_data_df = pd.concat([train_df_X,test_df])
del train_df_X

# 读取所有的embedding特征
print('reading embedding...')
class_creator_tag =  pd.read_csv('./class_creator_tag.csv')

# 构成最终数据
all_data_df_new = all_data_df.merge(class_creator_tag,on='uid',how='left')
del class_creator_tag

print('Getting all the feature...')
lgb_feature_list = list(all_data_df_new.columns.drop(['uid']))
lgb_df_all = all_data_df_new[lgb_feature_list].copy()
del all_data_df_new

print('Divide trainset and testset...')
train = lgb_df_all[:train_shape]
test = lgb_df_all[train_shape:]
print(test.shape)
del lgb_df_all

# 划分数据
print('Dividing dataset to trainset and valset...')
train_X, val_X, train_y, val_y = train_test_split(train,train_df_y,test_size=0.2,random_state=42)

train_sets = []
test_sets = []

for clf in [lgb_model_1, lgb_model_2, lgb_model_3]:
    train_set, test_set = get_stacking(clf, train_X, train_y, train_X)
    train_sets.append(train_set)
    test_sets.append(test_set)

meta_train = np.concatenate([result_set.reshape(-1,1) for result_set in train_sets], axis=1)
meta_test = np.concatenate([y_test_set.reshape(-1,1) for y_test_set in test_sets], axis=1)

# 使用决策树作为我们的次级分类器
dt_model = DecisionTreeClassifier()
dt_model.fit(meta_train, train_y)
df_predict = dt_model.predict(meta_test)

# 汇总测试结果
submission = test_df[['uid']]
del test
submission['preds'] = df_predict

# 保存测试结果
submission.to_csv('./sub_last.csv', index=None, header=False, encoding='utf-8', float_format='%.4f') 

print(df_predict)