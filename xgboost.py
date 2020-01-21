
# coding: utf-8

# In[26]:


import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4


# In[27]:


train = pd.read_csv('train.csv')


# In[28]:


train.columns


# In[29]:



train = train.drop(columns = 'Unnamed: 0')
train = train.drop(columns = 'date_of_game')
train = train.drop(columns = 'match_event_id')
train = train.drop(columns = 'year')
train = train.drop(columns = 'day')
train = train.drop(columns = 'month')
train = train.drop(columns = 'm_id_null')
train = train.drop(columns = 'l_x_null')
train = train.drop(columns = 'l_y_null')
train = train.drop(columns = 'rem_min_null')
train = train.drop(columns = 'rem_sec_null')
train = train.drop(columns = 'p_shot_null')
train = train.drop(columns = 'd_shot_null')
train = train.drop(columns = 'a_shot_null')
train = train.drop(columns = 'b_shot_null')
train = train.drop(columns = 'r_shot_null')
train = train.drop(columns = 't_shot_null')
train = train.drop(columns = 'c_shot_null')


# In[30]:


train['area_of_shot'].replace(np.nan,'NaN',inplace=True)
train['shot_basics'].replace(np.nan,'NaN',inplace=True)
train['range_of_shot'].replace(np.nan,'NaN',inplace=True)
train['type_of_shot'].replace(np.nan,'NaN',inplace=True)
train['type_of_combined_shot'].replace(np.nan,'NaN',inplace=True)


# In[31]:


from sklearn.preprocessing import LabelEncoder
#game_season, area_of_shot, shot_basics, range_of_shot, date_of_game, type_of_shot, type_of_combined_shot
label_en_gs = LabelEncoder()
label_en_as = LabelEncoder()
label_en_bs = LabelEncoder()
label_en_rs = LabelEncoder()
label_en_dg = LabelEncoder()
label_en_ts = LabelEncoder()
label_en_cs = LabelEncoder()


# In[32]:


train['game_season'] = label_en_gs.fit_transform(train['game_season'])


# In[34]:


#train['date_of_game'] = label_en_dg.fit_transform(train['date_of_game'])


# In[35]:


train['area_of_shot'] = label_en_as.fit_transform(train['area_of_shot'])
train['shot_basics'] = label_en_bs.fit_transform(train['shot_basics'])
train['range_of_shot'] = label_en_rs.fit_transform(train['range_of_shot'])
train['type_of_shot'] = label_en_ts.fit_transform(train['type_of_shot'])
train['type_of_combined_shot'] = label_en_cs.fit_transform(train['type_of_combined_shot'])


# In[36]:


print(label_en_as.transform(['NaN']))
print(label_en_bs.transform(['NaN']))
print(label_en_rs.transform(['NaN']))
print(label_en_ts.transform(['NaN']))
print(label_en_cs.transform(['NaN']))


# In[37]:


train['area_of_shot'].replace(4,np.nan,inplace=True)
train['shot_basics'].replace(5,np.nan,inplace=True)
train['range_of_shot'].replace(5,np.nan,inplace=True)
train['type_of_shot'].replace(0,np.nan,inplace=True)
train['type_of_combined_shot'].replace(0,np.nan,inplace=True)


# In[42]:


def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='mae', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
    print(cvresult.shape[0])
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['is_goal'],eval_metric='mae')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
        
    #Print model report:
    print ("\nModel Report")
    print ("Accuracy : %.4g" % metrics.accuracy_score(dtrain['is_goal'].values, dtrain_predictions))
    print ("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['is_goal'], dtrain_predprob))
    print(alg.get_booster())                
    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')


# In[44]:


#Choose all predictors except target & IDcols
target = 'is_goal'
IDcol = 'shot_id_number'
predictors = [x for x in train.columns if x not in [target, IDcol]]
xgb1 = XGBClassifier(
 learning_rate =0.8,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb1, train, predictors)


# In[45]:


param_test0 = {
 'eval_metric':['rmse', 'mae', 'logloss', 'error']
}
gsearch0 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.8, n_estimators=373, max_depth=5,
 min_child_weight=1,eval_metric = 'error', gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test0, scoring='roc_auc',n_jobs=4,iid=False, cv=5)


# In[46]:


gsearch0.fit(train[predictors],train[target])
gsearch0.grid_scores_, gsearch0.best_params_, gsearch0.best_score_


# In[47]:


param_test1 = {
 'max_depth':[1,3,5,7],
 'min_child_weight':[1,3,5]
}


# In[48]:


gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.8, n_estimators=365, max_depth=5,
 min_child_weight=1, gamma=0,eval_metric = 'mae',subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)


# In[49]:


gsearch1.fit(train[predictors],train[target])


# In[76]:


gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_


# In[ ]:


# param_test2 = {
#  'max_depth':[2,3,4],
#  'min_child_weight':[2,3,4]
# }
# gsearch2 = GridSearchCV(estimator = XGBClassifier( learning_rate=0.8, n_estimators=365, max_depth=1,
#  min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
#  objective= 'binary:logistic', eval_metric='mae',nthread=4, scale_pos_weight=1,seed=27), 
#  param_grid = param_test2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)


# In[37]:


# gsearch2.fit(train[predictors],train[target])
# gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_


# In[48]:


param_test2b = {
 'max_depth':[1,2]
}
gsearch2b = GridSearchCV(estimator = XGBClassifier( learning_rate=0.1, n_estimators=140, max_depth=4,
 min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test2b, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch2b.fit(train[predictors],train[target])


# In[49]:


modelfit(gsearch2b.best_estimator_, train, predictors)
gsearch2b.grid_scores_, gsearch2b.best_params_, gsearch2b.best_score_


# In[77]:


param_test3 = {
 'gamma':[i/10.0 for i in range(0,5)]
}
gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.8, n_estimators=365, max_depth=1,
 min_child_weight=1, gamma=0,eval_metric = 'mae', subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)


# In[78]:


gsearch3.fit(train[predictors],train[target])
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_


# In[79]:


xgb2 = XGBClassifier(
 learning_rate =0.8,
 n_estimators=1000,
 max_depth=1,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb2, train, predictors)


# In[80]:


param_test4 = {
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}
gsearch4 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.8, n_estimators=410, max_depth=1,
 min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch4.fit(train[predictors],train[target])
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_


# In[ ]:


param_test5 = {
 'subsample':[0.65, 0.70, 0.75],
 'colsample_bytree':[0.85, 0.9, 0.95]
}
gsearch5 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.8, n_estimators=410, max_depth=1,
 min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
 param_grid = param_test4, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch4.fit(train[predictors],train[target])
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_

