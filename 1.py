 # -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 18:21:53 2019

@author: Hassan
"""
 
import pandas as pd  
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn import linear_model 
from sklearn.tree import DecisionTreeRegressor


da=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\Number 1\\All metrics.csv")
Estimated=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\Number 1\\Estimated range.csv")
Est=np.transpose(Estimated)
Est.index=range(1024)
Est.columns=['estimated']
actual=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\Number 1\\Actual range.csv")
actu=np.transpose(actual)
actu.index=range(1024)
actu.columns=['actual range']
data=pd.concat([da,actu],axis=1)


error_of_est=np.ones((5,1),dtype=float)
i=0
while i<5:
 idx = np.random.randint(1024, size=103)
 Es_sample=Est.iloc[idx,:]
 actu_sample=actu.iloc[idx,:]
 error_of_est[i]=mean_squared_error(Es_sample,actu_sample)
 i=i+1
#----------------------------- Error of Estimation --------------------------------------------------
 
error_of_estimation=np.mean(error_of_est)
 
#------------------------------ SVR Regression ----------------------------------------------------------------------
error_of_pred_sv=np.ones((5,1),dtype=float)


#------------------------------ computing Error five time --------------------------------------------
i=0

while i<5:
    
    var_train,var_test,targ_train,targ_test=train_test_split(data.loc[:,['energy','max','risetime','MED','RMS-DS','kurtosis']],data.loc[:,['actual range']].values,train_size=0.9,random_state=None)
    
    #------------------------------ Standardizing the features -------------------------------------------
    
    rbX = StandardScaler()
    var_train1= rbX.fit_transform(var_train)
    
    
    rbY = StandardScaler()
    targ_train1= rbY.fit_transform(targ_train)
    
    svregressor=SVR(kernel='rbf')
    grid_param = {
        
        'C': [100,10,15,1,0.1,0.01],
        'gamma': [0.01,0.1,1,10,100],
        } 
    clf=GridSearchCV(estimator=svregressor,param_grid=grid_param,cv=5,n_jobs=-1)
    clf.fit(var_train1,targ_train1)
    best_parameters = clf.best_params_  
    print(best_parameters)
    
    ##------------------------------train error------------------------------------
    
    svregressor_=SVR(kernel='rbf',C=best_parameters['C'],gamma=best_parameters['gamma'])
    svregressor_.fit(var_train1, targ_train1) 
    y_pred_tra = svregressor_.predict(var_train1)
    y_pred_train=rbY.inverse_transform(y_pred_tra)
    error_on_training=mean_squared_error(targ_train,y_pred_train)
    
    #-----------------------------test error-------------------------------------
    
    y_pred_te_sv = svregressor_.predict(rbX.transform(var_test))
    y_pred_test_svr = rbY.inverse_transform(y_pred_te_sv)
    error_of_pred_sv[i]=mean_squared_error(targ_test,y_pred_test_svr)
    i=i+1

error_of_prediction_SVR=np.mean(error_of_pred_sv)

#--------------------------linear Regression ------------------------------------------------------
error_of_pred_lin=np.ones((5,1),dtype=float)
i=0
while i<5:
 var_train,var_test,targ_train,targ_test=train_test_split(data.loc[:,['energy','max','risetime','MED','RMS-DS','kurtosis']],data.loc[:,['actual range']].values,train_size=0.9,random_state=None)

#------------------------------ Standardizing the features -------------------------------------------

 rbX = StandardScaler()
 var_train1= rbX.fit_transform(var_train)


 rbY = StandardScaler()
 targ_train1= rbY.fit_transform(targ_train)


 reg=linear_model.LinearRegression()
 reg.fit(var_train1,targ_train1)
 y_pred_lin=reg.predict(rbX.transform(var_test))
 y_pred_linear=rbY.inverse_transform(y_pred_lin)
 error_of_pred_lin[i]=mean_squared_error(targ_test,y_pred_linear)
 i=i+1

error_of_prediction_linear=np.mean(error_of_pred_lin)

#------------------------------- Decision Tree Regression ------------------------------------------------

error_of_pred_tr=np.ones((5,1),dtype=float)
i=0
while i<5:
 var_train,var_test,targ_train,targ_test=train_test_split(data.loc[:,['energy','max','risetime','MED','RMS-DS','kurtosis']],data.loc[:,['actual range']].values,train_size=0.9,random_state=None)

#------------------------------ Standardizing the features -------------------------------------------

 rbX = StandardScaler()
 var_train1= rbX.fit_transform(var_train)


 rbY = StandardScaler()
 targ_train1= rbY.fit_transform(targ_train)
 
 trclassifier=DecisionTreeRegressor()
 grid_param = {
    
    'max_depth': [1,2,3,4,5,6,7,8,9,10],
    } 
 clftr=GridSearchCV(estimator=trclassifier,param_grid=grid_param,cv=5,n_jobs=-1)
 clftr.fit(var_train1,targ_train1)
 best_parameters_tr = clftr.best_params_  
 print(best_parameters_tr)

##------------------------------train error------------------------------------

 trclassifier=DecisionTreeRegressor(max_depth=best_parameters_tr['max_depth'])
 trclassifier.fit(var_train1,targ_train1) 
 y_pred_tra_tr = trclassifier.predict(var_train1)
 y_pred_train_tree=rbY.inverse_transform(y_pred_tra_tr)
 error_on_training_tree=mean_squared_error(targ_train,y_pred_train_tree)


#--------------------------------test error --------------------------------------------
 y_pred_tr=trclassifier.predict(rbX.transform(var_test))
 y_pred_tree=rbY.inverse_transform(y_pred_tr)
 error_of_pred_tr[i]=mean_squared_error(targ_test,y_pred_tree)
 i=i+1

error_of_prediction_tree=np.mean(error_of_pred_tr)

#-------------------------------- Ridge Regression ----------------------------------------------------

i=0

error_of_pred_ri=np.ones((5,1),dtype=float)
while i<5:
    
 var_train,var_test,targ_train,targ_test=train_test_split(data.loc[:,['energy','max','risetime','MED','RMS-DS','kurtosis']],data.loc[:,['actual range']].values,train_size=0.9,random_state=None)

#------------------------------ Standardizing the features -------------------------------------------

 rbX = StandardScaler()
 var_train1= rbX.fit_transform(var_train)


 rbY = StandardScaler()
 targ_train1= rbY.fit_transform(targ_train)

 riclassifier=linear_model.Ridge(alpha=2)
 grid_param = {
    
    'alpha': [0.001,0.1,0,1,2,3,4,5,6,7,8,9,10,11],
    } 
 clfr=GridSearchCV(estimator=riclassifier,param_grid=grid_param,cv=5,n_jobs=-1)
 clfr.fit(var_train1,targ_train1)
 best_parameters_r = clfr.best_params_  
 print(best_parameters_r)

##------------------------------train error------------------------------------

 riclassifier=linear_model.Ridge(alpha=best_parameters_r['alpha'])
 riclassifier.fit(var_train1,targ_train1) 
 y_pred_tra_ri = riclassifier.predict(var_train1)
 y_pred_train_ridge=rbY.inverse_transform(y_pred_tra_ri)
 error_on_training_ridge=mean_squared_error(targ_train,y_pred_train_ridge)

#-----------------------------test error-------------------------------------

 y_pred_te_ri = riclassifier.predict(rbX.transform(var_test))
 y_pred_test_ridge = rbY.inverse_transform(y_pred_te_ri)
 error_of_pred_ri[i]=mean_squared_error(targ_test,y_pred_test_ridge)
 i=i+1

 error_of_prediction_ridge=np.mean(error_of_pred_ri)
 
 