# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 19:54:30 2017

@author: fatemeh
"""

import pandas as pd 
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
import numpy as np
import time 
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import KFold
# read csv file  
data=pd.read_csv('ourdata.csv',encoding='latin1')

# read csv file  
data1=pd.read_csv('tmdb_5000_movies.csv')
# drop out nan data and  clean data 
data1=data1.dropna()
#data=data.drop(data.index[[1322,1482]])

data1=data1[data1['title']!='I Want Your Money']
data1=data1[data1['title']!="2016: Obama's America"]
data1=data1[data1['title']!="Love Him, Hate Him, You Don't Know Him"]
data1=data1[data1['title']!="Tiny Furniture"]
data1=data1[data1['title']!="Bending Steel"]  

id_data=data1['id']
# our labeles to predict imdb rate 
label=data1[['vote_average']]


# ADD UNIQUE NUMBER TO EACH NAME OF CHARECTER AND DIRECTER FOR DATA ANAYLYSIS

name_direc=pd.unique(data['directer'].values.ravel())
data['directer']=data['directer'].replace(name_direc,np.arange(0,924))


name_char=pd.unique(data['char1'].values.ravel())
data['char1']=data['char1'].replace(name_char,np.arange(0,788))


# our labeles to predict imdb rate 
label=data1[['vote_average']]

# our numeric features to predict data
feature=data[['budget','popularity','revenue','runtime','vote_count','char1','directer']]


############################
# label of  vote_averages 
for i in label['vote_average']: 
    
     if i <= 2.4:
        label=label.replace (i,10)
     elif 2.4<i<=5:
        label=label.replace(i,20)
     elif 5.1<=i<7.4:
        label=label.replace(i,30)
     elif 7.4<=i<10 :
        label=label.replace(i,40)
        
label=label.replace([10, 20, 30, 40],[1, 2, 3, 4])
 

#n_estimators = 10
start_time = time.time()


# K-Fold Cross validation, divide data into K subsets
k_fold = KFold(10)     

i = 1

acc_list = []

# Train and test LSTSVM K times
for train_index, test_index in k_fold.split(feature):
    
    # Extract data based on index created by k_fold
    X_train = np.take(feature, train_index, axis=0) 
    X_test = np.take(feature, test_index, axis=0)
    
    X_train_label = np.take(label, train_index, axis=0)
    X_test_label = np.take(label, test_index, axis=0)
    
    result2=  OneVsRestClassifier(SVC(kernel='rbf'))
    result2.fit(X_train,X_train_label)
    result_df=result2.predict(X_test)
    
    acc_sc = accuracy_score(X_test_label, result_df)
    
    acc_list.append(acc_sc)
    
    print("Fold: %d" % i)
    
    i = i + 1
    

t_time=time.time() - start_time
print('Test accuracy: %f' % (np.mean(acc_list)))

print("Finished %.2f Seconds" % (time.time() - start_time))
