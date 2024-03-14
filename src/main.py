#!/usr/bin/env python
# coding: utf-8

# # Supervised Machine Learning competition
# ## Multi-class classification

# In[18]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import sklearn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

from flaml import AutoML
from catboost import CatBoostClassifier, Pool

from hyperopt import hp, fmin, tpe, STATUS_OK, Trials


# Import data and store in DataFrames

# In[2]:


df_feat_test = pd.read_csv('../data/raw/features_test.csv')
df_targ_train = pd.read_csv('../data/raw/target_train.csv')
df_feat_train = pd.read_csv('../data/raw/features_train.csv')


# In[4]:


X_test_raw = df_feat_test
X_train_raw = df_feat_train
y_train = df_targ_train['Expected']

X_train = X_train_raw.drop(columns = ['Id'])
X_test = X_test_raw.drop(columns = ['Id'])


# Data inspection

# In[5]:


X_test.shape, X_train.shape, y_train.shape


# In[7]:


X_train.info()


# In[6]:


X_train.describe()


# In[8]:


X_train.isnull().sum().unique()


# In[14]:


sns_plot = sns.clustermap(X_train.corr(), cmap="rocket_r", figsize=(7,7))


# In[15]:


X_train.corr().unstack().sort_values(ascending=False).drop_duplicates()


# In[ ]:


model_Cat = CatBoostClassifier(
         early_stopping_rounds = 10,
         learning_rate = 0.06692273485930686,
         n_estimators = 200,
         thread_count = -1,
         verbose = False,
         random_seed = 10242048,
         #od_type = "Iter",
         #od_wait = 10
)


# In[ ]:


feats = model_Cat.select_features(X_train, y_train,
                         features_for_select=list(range(120)),
                         num_features_to_select=16)


# In[ ]:


X_train_red = X_train.iloc[:,feats['selected_features']]
X_train_red.shape


# In[ ]:


automl = AutoML()


# In[ ]:


automl_settings = {
    "time_budget": 1200, # 20 mins to try and select best model
    "metric": 'macro_f1',
    "task": 'classification',
    "log_file_name": 'mylog.log',
    "eval_method": 'cv',
    "n_splits": 5
}


# In[ ]:


automl.fit(X_train=X_train_red, y_train=y_train.values,
           **automl_settings)


# In[ ]:


automl.best_estimator


# In[ ]:


automl.best_config


# In[ ]:


automl.model.get_params()


# In[ ]:


predictions = automl.predict(X_train_red)
cf = confusion_matrix(y_train, predictions)
print(classification_report(y_train, predictions))
sns.heatmap(cf, annot=True);


# # Use FLAML ensemble approach

# In[ ]:


automl_ens = AutoML()

automl_ens.fit(X_train=X_train_red, y_train=y_train.values, ensemble=True,
           **automl_settings)


# In[ ]:


automl_ens.best_estimator


# In[ ]:


automl_ens.model


# In[ ]:


predictions = automl_ens.model.predict(X_train_red)
cf = confusion_matrix(y_train, predictions)
print(classification_report(y_train, predictions))
sns.heatmap(cf, annot=True);


# ## Continue with CatBoost algorithm

# In[99]:


model_Cat = CatBoostClassifier(
         early_stopping_rounds = 13,
         learning_rate = 0.04171721859304757,
         n_estimators = 2500,
         thread_count = -1,
         verbose = False,
         random_seed = 10242048,
         #od_type = "Iter",
         #od_wait = 10
)


# In[100]:


model_Cat.fit(X_train_red, y_train)


# In[101]:


cv_score = cross_val_score(model_Cat, X_train_red, y_train,
                           cv=5, scoring='f1_macro')
print(cv_score)
print(np.mean(cv_score))


# In[ ]:


list_feat_imp = model_Cat.get_feature_importance(data=Pool(X_train_red, label=y_train))


# In[ ]:


plt.hist(list_feat_imp);


# In[ ]:


list_feat_imp[list_feat_imp>1]


# ## Finetune CatBoost using GridSearchCV

# In[ ]:


from sklearn.model_selection import GridSearchCV

catboost_tune = CatBoostClassifier(random_seed = 10242048,
                                   verbose = False,
                                   thread_count = -1)
                                  

grid_search = {
            'early_stopping_rounds': [10, 13],
            'min_data_in_leaf': [500],
            'learning_rate': [0.175],
            'n_estimators': [150],
            'l2_leaf_reg': [4]}


grid_search_obj = GridSearchCV(estimator=catboost_tune,
                               param_grid=grid_search,
                               scoring='f1_macro', cv=3, verbose=5, n_jobs=-1)

#grid_search_obj.fit(X_Train,Y_Train)



# In[ ]:


grid_search_obj.fit(X_train_red,y_train)


# In[ ]:


predictioncat = grid_search_obj.best_estimator_.predict(X_train_red)
print(confusion_matrix(y_train,predictioncat))
print(classification_report(y_train,predictioncat))


# In[ ]:


pd.set_option('display.max_colwidth', None)

cv_result_df = pd.DataFrame({
    'Model Rank': grid_search_obj.cv_results_['rank_test_score'],
    'Model Hyperparams': grid_search_obj.cv_results_['params'],
    'Avg CV F1-macro': grid_search_obj.cv_results_['mean_test_score'],
    'Std Dev CV F1-macro': grid_search_obj.cv_results_['std_test_score'],
    'CV Fold 1 F1-macro': grid_search_obj.cv_results_['split0_test_score'],
    'CV Fold 2 F1-macro': grid_search_obj.cv_results_['split1_test_score'],
    'CV Fold 3 F1-macro': grid_search_obj.cv_results_['split2_test_score']
})


cv_result_df.sort_values(by=['Model Rank'], ascending=True)


# ## Use Hyperopt for Bayesian hyperparameter tuning

# In[ ]:


from hyperopt import hp, fmin, tpe, STATUS_OK, Trials


# In[ ]:


space = {
    'learning_rate':  hp.choice('learning_rate', [0.04, 0.06]),
    'n_estimators': hp.choice('n_estimators', [600, 860, 1000]),
    'l2_leaf_reg' : hp.choice('l2_leaf_reg', [30, 100])
}

def objective(space):
    cat_model_opt = CatBoostClassifier(

                                        learning_rate = space['learning_rate'],
                                        n_estimators = space['n_estimators'],
                                        l2_leaf_reg = space['l2_leaf_reg'],
                                        min_data_in_leaf = 300,
                                        early_stopping_rounds = 13,
                                        random_seed = 10242048,
                                        verbose = False,
                                        thread_count = -1)
    
   
       
    f1 = cross_val_score(cat_model_opt, X_train_red, y_train, cv=5, scoring='f1_macro').mean()

    # We aim to maximize accuracy, therefore we return it as a negative value
    return {'loss': -f1, 'status': STATUS_OK }

trials = Trials()

best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest, # Tree parzen estimator
            max_evals=10,
            trials=trials)
best


# In[ ]:


depth = {0: 6}
lrate = {0: 0.02, 1: 0.04, 2:0.06}
n_est = {0: 600, 1:860, 2:1000}
l2_reg = {0: 10, 1: 30, 2: 100}

optimal_cat = CatBoostClassifier(
                                       learning_rate = lrate[best['learning_rate']],
                                       l2_leaf_reg = l2_reg[best['l2_leaf_reg']],
                                       n_estimators = n_est[best['n_estimators']],
                                       min_data_in_leaf = 500,
                                        early_stopping_rounds = 5,
                                        random_seed = 10242048,
                                        verbose = False,
                                        thread_count = -1,
                                      ).fit(X_train_red,y_train)


# In[ ]:


prediction_best_cat = optimal_cat.predict(X_train_red)
print(confusion_matrix(y_train,prediction_best_cat))
print(classification_report(y_train,prediction_best_cat))


# In[ ]:


X_test_red = X_test.iloc[:,feats['selected_features']]
X_train_red.shape


# In[ ]:


y_pred_automl = automl.predict(X_test_red)


# In[ ]:


df_pred_automl = pd.DataFrame()
df_pred_automl['Id']=X_test_raw['Id']
df_pred_automl.set_index('Id', inplace=True)
df_pred_automl['Predicted'] = y_pred_automl.ravel()


# In[ ]:


df_pred_automl.to_csv('df_pred_automl_2.csv')


# In[102]:


y_pred_cat = model_Cat.predict(X_test_red)


# In[103]:


df_pred_cat = pd.DataFrame()
df_pred_cat['Id']=X_test_raw['Id']
df_pred_cat.set_index('Id', inplace=True)
df_pred_cat['Predicted'] = y_pred_cat.ravel()


# In[104]:


df_pred_cat.to_csv('df_pred_cat4.csv')


# In[ ]:


y_pred_opt_cat = optimal_cat.predict(X_test_red)


# In[ ]:


comp = pd.DataFrame(red_prediction == red_prediction_2)


# In[ ]:


comp.sum()

