#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 10:24:34 2019

@author: labadmin
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 02 21:05:32 2019

@author: Hassan
"""


import pandas as pd  
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier as GBC
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SMOTENC
 
 
data_ben1=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\bending1\\dataset1.csv",skiprows=4)
data_ben2=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\bending1\\dataset2.csv",skiprows=4)
data_ben3=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\bending1\\dataset3.csv",skiprows=4)
data_ben4=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\bending1\\dataset4.csv",skiprows=4)
data_ben5=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\bending1\\dataset5.csv",skiprows=4)
data_ben6=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\bending1\\dataset6.csv",skiprows=4)
data_ben7=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\bending1\\dataset7.csv",skiprows=4)
frames_ben1 = [data_ben1,data_ben2,data_ben3,data_ben4,data_ben5,data_ben6,data_ben7]
result_ben1 = pd.concat(frames_ben1)
result_ben1.index=range(3360)
df_ben1 = pd.DataFrame({'label': [1]},index=range(0,3360))
dat_ben1=pd.concat([result_ben1,df_ben1],axis=1)

#-------------------------------------------------------------------------------------------------

data__ben1=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\bending2\\dataset1.csv",skiprows=4)
data__ben2=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\bending2\\dataset2.csv",skiprows=4)
data__ben3=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\bending2\\dataset3.csv",skiprows=4)
data__ben4=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\bending2\\dataset4.csv",skiprows=4)
data__ben4=data__ben4['# Columns: time'].str.split(expand=True)
data__ben4.columns=['# Columns: time','avg_rss12','var_rss12','avg_rss13','var_rss13','avg_rss23','var_rss23']
data__ben5=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\bending2\\dataset5.csv",skiprows=4)
data__ben6=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\bending2\\dataset6.csv",skiprows=4)
frames_ben2 = [data__ben1,data__ben2,data__ben3,data__ben4,data__ben5,data__ben6]
result_ben2 = pd.concat(frames_ben2)
result_ben2.index=range(2880)
df_ben2 = pd.DataFrame({'label': [2]},index=range(0,2880))
dat__ben2=pd.concat([result_ben2,df_ben2],axis=1)

#-----------------------------------------------------------------------------------------------------

data_cyc1=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\cycling\\dataset1.csv",skiprows=4)
data_cyc2=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\cycling\\dataset2.csv",skiprows=4)
data_cyc3=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\cycling\\dataset3.csv",skiprows=4)
data_cyc4=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\cycling\\dataset4.csv",skiprows=4)
data_cyc5=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\cycling\\dataset5.csv",skiprows=4)
data_cyc6=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\cycling\\dataset6.csv",skiprows=4)
data_cyc7=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\cycling\\dataset7.csv",skiprows=4)
data_cyc8=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\cycling\\dataset8.csv",skiprows=4)
data_cyc9=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\cycling\\dataset99.csv",skiprows=4)
data_cyc10=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\cycling\\dataset10.csv",skiprows=4)
data_cyc11=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\cycling\\dataset11.csv",skiprows=4)
data_cyc12=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\cycling\\dataset12.csv",skiprows=4)
data_cyc13=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\cycling\\dataset13.csv",skiprows=4)
data_cyc14=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\cycling\\dataset144.csv",skiprows=4)
data_cyc15=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\cycling\\dataset15.csv",skiprows=4)
frames_cyc = [data_cyc1,data_cyc2,data_cyc3,data_cyc4,data_cyc5,data_cyc6,data_cyc7,data_cyc8,data_cyc9,data_cyc10,data_cyc11,data_cyc12,data_cyc13,data_cyc14,data_cyc15]
result_cyc = pd.concat(frames_cyc)
result_cyc.index=range(7200)
df_cyc = pd.DataFrame({'label': [3]},index=range(0,7200))
data_cyc=pd.concat([result_cyc,df_cyc],axis=1)

#----------------------------------------------------------------------------------------------

data_ly1=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\lying\\dataset1.csv",skiprows=4)
data_ly2=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\lying\\dataset2.csv",skiprows=4)
data_ly3=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\lying\\dataset3.csv",skiprows=4)
data_ly4=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\lying\\dataset4.csv",skiprows=4)
data_ly5=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\lying\\dataset5.csv",skiprows=4)
data_ly6=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\lying\\dataset6.csv",skiprows=4)
data_ly7=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\lying\\dataset7.csv",skiprows=4)
data_ly8=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\lying\\dataset8.csv",skiprows=4)
data_ly9=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\lying\\dataset9.csv",skiprows=4)
data_ly10=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\lying\\dataset10.csv",skiprows=4)
data_ly11=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\lying\\dataset11.csv",skiprows=4)
data_ly12=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\lying\\dataset12.csv",skiprows=4)
data_ly13=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\lying\\dataset13.csv",skiprows=4)
data_ly14=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\lying\\dataset14.csv",skiprows=4)
data_ly15=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\lying\\dataset15.csv",skiprows=4)
frames_ly = [data_ly1,data_ly2,data_ly3,data_ly4,data_ly5,data_ly6,data_ly7,data_ly8,data_ly9,data_ly10,data_ly11,data_ly12,data_ly13,data_ly14,data_ly15]
result_ly = pd.concat(frames_ly)
result_ly.index=range(7200)
df_ly = pd.DataFrame({'label': [4]},index=range(0,7200))
data_ly=pd.concat([result_ly,df_ly],axis=1)

#-------------------------------------------------------------------------------------------------

data_sit1=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\sitting\\dataset1.csv",skiprows=4)
data_sit2=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\sitting\\dataset2.csv",skiprows=4)
data_sit3=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\sitting\\dataset3.csv",skiprows=4)
data_sit4=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\sitting\\dataset4.csv",skiprows=4)
data_sit5=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\sitting\\dataset5.csv",skiprows=4)
data_sit6=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\sitting\\dataset6.csv",skiprows=4)
data_sit7=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\sitting\\dataset7.csv",skiprows=4)
data_sit8=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\sitting\\dataset8.csv",skiprows=4)
data_sit9=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\sitting\\dataset9.csv",skiprows=4)
data_sit10=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\sitting\\dataset10.csv",skiprows=4)
data_sit11=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\sitting\\dataset11.csv",skiprows=4)
data_sit12=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\sitting\\dataset12.csv",skiprows=4)
data_sit13=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\sitting\\dataset13.csv",skiprows=4)
data_sit14=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\sitting\\dataset14.csv",skiprows=4)
data_sit15=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\sitting\dataset15.csv",skiprows=4)
frames_sit= [data_sit1,data_sit2,data_sit3,data_sit4,data_sit5,data_sit6,data_sit7,data_sit8,data_sit9,data_sit10,data_sit11,data_sit12,data_sit13,data_sit14,data_sit15]
result_sit = pd.concat(frames_sit)
result_sit.index=range(7199)
df_sit= pd.DataFrame({'label': [5]},index=range(0,7199))
data_sit=pd.concat([result_sit,df_sit],axis=1)

#----------------------------------------------------------------------------------------------------

data_sta1=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\standing\\dataset1.csv",skiprows=4)
data_sta2=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\standing\\dataset2.csv",skiprows=4)
data_sta3=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\standing\\dataset3.csv",skiprows=4)
data_sta4=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\standing\\dataset4.csv",skiprows=4)
data_sta5=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\standing\\dataset5.csv",skiprows=4)
data_sta6=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\standing\\dataset6.csv",skiprows=4)
data_sta7=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\standing\\dataset7.csv",skiprows=4)
data_sta8=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\standing\\dataset8.csv",skiprows=4)
data_sta9=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\standing\\dataset9.csv",skiprows=4)
data_sta10=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\standing\\dataset10.csv",skiprows=4)
data_sta11=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\standing\\dataset11.csv",skiprows=4)
data_sta12=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\standing\\dataset12.csv",skiprows=4)
data_sta13=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\standing\\dataset13.csv",skiprows=4)
data_sta14=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\standing\\dataset14.csv",skiprows=4)
data_sta15=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\standing\dataset15.csv",skiprows=4)
frames_sta= [data_sta1,data_sta2,data_sta3,data_sta4,data_sta5,data_sta6,data_sta7,data_sta8,data_sta9,data_sta10,data_sta11,data_sta12,data_sta13,data_sta14,data_sta15]
result_sta = pd.concat(frames_sta)
result_sta.index=range(7200)
df_sta= pd.DataFrame({'label': [6]},index=range(0,7200))
data_sta=pd.concat([result_sta,df_sta],axis=1)

#---------------------------------------------------------------------------------------------------------------

data_wa1=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\walking\\dataset1.csv",skiprows=4)
data_wa2=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\walking\\dataset2.csv",skiprows=4)
data_wa3=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\walking\\dataset3.csv",skiprows=4)
data_wa4=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\walking\\dataset4.csv",skiprows=4)
data_wa5=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\walking\\dataset5.csv",skiprows=4)
data_wa6=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\walking\\dataset6.csv",skiprows=4)
data_wa7=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\walking\\dataset7.csv",skiprows=4)
data_wa8=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\walking\\dataset8.csv",skiprows=4)
data_wa9=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\walking\\dataset9.csv",skiprows=4)
data_wa10=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\walking\\dataset10.csv",skiprows=4)
data_wa11=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\walking\\dataset11.csv",skiprows=4)
data_wa12=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\walking\\dataset12.csv",skiprows=4)
data_wa13=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\walking\\dataset13.csv",skiprows=4)
data_wa14=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\walking\\dataset14.csv",skiprows=4)
data_wa15=pd.read_csv("F:\\Projects\\Master\\Statistical  learning\\project\\walking\dataset15.csv",skiprows=4)
frames_wa= [data_wa1,data_wa2,data_wa3,data_wa4,data_wa5,data_wa6,data_wa7,data_wa8,data_wa9,data_wa10,data_wa11,data_wa12,data_wa13,data_wa14,data_wa15]
result_wa = pd.concat(frames_wa)
result_wa.index=range(7200)
df_wa= pd.DataFrame({'label': [7]},index=range(0,7200))
data_wa=pd.concat([result_wa,df_wa],axis=1)


#----------------------------------------------------------------------------------------------------

da=[dat_ben1,dat__ben2,data_cyc,data_ly,data_sit,data_sta,data_wa]
data=pd.concat(da)
data.index=range(42239)

#-------------------------------------------------------------------------------------------------

features = ['# Columns: time', 'avg_rss12', 'var_rss12', 'avg_rss13','var_rss13','avg_rss23','var_rss23']

# Separating out the features
x = data.loc[:, features].values

# Separating out the target
y = data.loc[:,['label']].values

# Standardizing the features
x = StandardScaler().fit_transform(x)

pca = PCA(n_components=4)

principalComponents = pca.fit_transform(x)

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2', 'principal component 3', 'principal component 4'])

finalDf = pd.concat([principalDf, data[['label']]], axis = 1)

#------------------------------------ SVM ----------------------------------------------------

var_train,var_test,targ_train,targ_test=train_test_split(finalDf.loc[:,['principal component 1', 'principal component 2']].values,finalDf.loc[:,['label']].values,train_size=0.9,random_state=0)
svclassifier=SVC(kernel='rbf')
grid_param = {  
        'C': [100,10,1,0.1,0.01,0.001],
    'gamma': [0.001,0.01,0.1,1,10,100],
    } 
clf=GridSearchCV(estimator=svclassifier,param_grid=grid_param,scoring='accuracy',cv=5,n_jobs=-3)
clf.fit(var_train,targ_train)
best_parameters = clf.best_params_  
print(best_parameters)

##------------------------------train error------------------------------------

svclassifier=SVC(kernel='rbf',C=best_parameters['C'],gamma=best_parameters['gamma'])
svclassifier.fit(var_train, targ_train) 
y_pred = svclassifier.predict(var_train)
#print(confusion_matrix(targ_train,y_pred))  
print(classification_report(targ_train,y_pred))

#-----------------------------test error-------------------------------------

y_pred_svm = svclassifier.predict(var_test)
#print(confusion_matrix(targ_test,y_pred_svm))  
print(classification_report(targ_test,y_pred_svm))

#----------------------------- Logostic --------------------------------------------------------------------

var_train,var_test,targ_train,targ_test=train_test_split(finalDf.loc[:,['principal component 1', 'principal component 2', 'principal component 3', 'principal component 4']].values,finalDf.loc[:,['label']].values,train_size=0.9,random_state=0)
logclassifier=LogisticRegression()
grid_param = {  
    'C': [10,30,50,70,100,200],
    } 
clf1=GridSearchCV(estimator=logclassifier,param_grid=grid_param,scoring='accuracy',cv=5,n_jobs=-3)
clf1.fit(var_train,targ_train)
best_parameters = clf1.best_params_  
print(best_parameters)

#-----------------------------test error--------------------------------------------------------------------------
logclassifier=SVC(C=best_parameters['C'])
logclassifier.fit(var_train, targ_train) 
y_pred_log = logclassifier.predict(var_test)
#print(confusion_matrix(targ_test,y_pred))  
print(classification_report(targ_test,y_pred_log))

#--------------------------- LDA -----------------------------------------------------------------------------

var_train,var_test,targ_train,targ_test=train_test_split(finalDf.loc[:,['principal component 1', 'principal component 2', 'principal component 3', 'principal component 4']].values,finalDf.loc[:,['label']].values,train_size=0.9,random_state=0)
LDAclassifier=LDA()
LDAclassifier.fit(var_train,targ_train.ravel())
y_pred_LDA=LDAclassifier.predict(var_test)
print(classification_report(targ_test,y_pred_LDA))

#-------------------------- QDA ---------------------------------------------------------------------------------

var_train,var_test,targ_train,targ_test=train_test_split(finalDf.loc[:,['principal component 1', 'principal component 2', 'principal component 3', 'principal component 4']].values,finalDf.loc[:,['label']].values,train_size=0.9,random_state=0)
QDAclassifier=QDA()
QDAclassifier.fit(var_train,targ_train)
y_pred_QDA=QDAclassifier.predict(var_test)
print(classification_report(targ_test,y_pred_QDA))

#----------------------- Decision Tree -------------------------------------------------------------------------------

var_train,var_test,targ_train,targ_test=train_test_split(finalDf.loc[:,['principal component 1', 'principal component 2', 'principal component 3', 'principal component 4']].values,finalDf.loc[:,['label']].values,train_size=0.9,random_state=0)
Treeclassifier=DecisionTreeClassifier(max_depth=16)
Treeclassifier.fit(var_train,targ_train)
y_pred_Tree=Treeclassifier.predict(var_test)
print(classification_report(targ_test,y_pred_Tree))

#-------------------- Adaboost ------------------------------------------------------------------------------------------

var_train,var_test,targ_train,targ_test=train_test_split(finalDf.loc[:,['principal component 1', 'principal component 2', 'principal component 3', 'principal component 4']].values,finalDf.loc[:,['label']].values,train_size=0.9,random_state=0)
Adaclassifier=AdaBoostClassifier(n_estimators=30)
Adaclassifier.fit(var_train,targ_train.ravel())
y_pred_Ada=Adaclassifier.predict(var_test)
print(classification_report(targ_test,y_pred_Ada))

#============================================= Random Forest =========================================
var_train,var_test,targ_train,targ_test=train_test_split(finalDf.loc[:,['principal component 1', 'principal component 2', 'principal component 3', 'principal component 4']].values,finalDf.loc[:,['label']].values,train_size=0.9,random_state=0)
clf = RandomForestClassifier()
grid_param = {  
    'max_depth': [15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30],
    'n_estimators':[100,150,200],
    } 
clf=GridSearchCV(estimator=clf,param_grid=grid_param,scoring='accuracy',cv=5,n_jobs=-1)
clf.fit(var_train,targ_train)
best_parameters = clf.best_params_  
print(best_parameters)


clf = RandomForestClassifier(max_depth=best_parameters['max_depth'],n_estimators=best_parameters['n_estimators'])
clf.fit(var_train,targ_train.ravel())
y_pred_Ra=clf.predict(var_test)
print(classification_report(targ_test,y_pred_Ra))


#==================================== K_nearest==========================================================
var_train,var_test,targ_train,targ_test=train_test_split(finalDf.loc[:,['principal component 1', 'principal component 2', 'principal component 3', 'principal component 4']].values,finalDf.loc[:,['label']].values,train_size=0.9,random_state=0)

near = KNeighborsClassifier(n_neighbors=10)

near.fit(var_train,targ_train.ravel())

y_pred_near=near.predict(var_test)
print(classification_report(targ_test,y_pred_near))


#===================================Gradient Boosting================================================

var_train,var_test,targ_train,targ_test=train_test_split(finalDf.loc[:,['principal component 1', 'principal component 2', 'principal component 3', 'principal component 4']].values,finalDf.loc[:,['label']].values,train_size=0.9,random_state=0)

GBC_1= GBC(n_estimators=1000,max_depth=5,max_features='sqrt')

GBC_1.fit(var_train,targ_train.ravel())

y_pred_GBC=GBC_1.predict(var_test)
print(classification_report(targ_test,y_pred_GBC))

#==================================== over_Sampling ======================================
ros=RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_resample(finalDf.loc[:,['principal component 1', 'principal component 2', 'principal component 3', 'principal component 4']].values, finalDf.loc[:,['label']].values)
var_train,var_test,targ_train,targ_test=train_test_split(X_resampled,y_resampled.ravel(),train_size=0.9,random_state=0)

#------------------------------------ Random Forest after Over_Sampling -----------------

clf = RandomForestClassifier(max_depth=100,n_estimators=300)
clf.fit(var_train,targ_train.ravel())
y_pred_Ra=clf.predict(var_test)
print(classification_report(targ_test,y_pred_Ra))

#------------------------------------ K_nearest after Over_Sampling -----------------

near = KNeighborsClassifier(n_neighbors=10)

near.fit(var_train,targ_train.ravel())

y_pred_near=near.predict(var_test)
print(classification_report(targ_test,y_pred_near))

#------------------------------------ Adaboost after Over_Sampling -----------------

Adaclassifier=AdaBoostClassifier(n_estimators=30)
Adaclassifier.fit(var_train,targ_train.ravel())
y_pred_Ada=Adaclassifier.predict(var_test)
print(classification_report(targ_test,y_pred_Ada))

#-------------------------------------- Decision Tree after Over_Sampling--------------------------------------

Treeclassifier=DecisionTreeClassifier(max_depth=16)
Treeclassifier.fit(var_train,targ_train)
y_pred_Tree=Treeclassifier.predict(var_test)
print(classification_report(targ_test,y_pred_Tree))

#--------------------------------------- QDA after Over_Sampling---------------------------------------------

QDAclassifier=QDA()
QDAclassifier.fit(var_train,targ_train)
y_pred_QDA=QDAclassifier.predict(var_test)
print(classification_report(targ_test,y_pred_QDA))

#------------------------------------- LDA after Over_Sampling------------------------------------------

LDAclassifier=LDA()
LDAclassifier.fit(var_train,targ_train.ravel())
y_pred_LDA=LDAclassifier.predict(var_test)
print(classification_report(targ_test,y_pred_LDA))

#-------------------------------------- Logistic after Over_Sampling -------------------

logclassifier=SVC(C=200)
logclassifier.fit(var_train, targ_train) 
y_pred_log = logclassifier.predict(var_test)
print(classification_report(targ_test,y_pred_log))