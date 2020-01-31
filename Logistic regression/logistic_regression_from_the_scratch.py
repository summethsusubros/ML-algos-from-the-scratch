# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 19:49:38 2019

@author: SU SU BRO'S
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('Social_Network_Ads.csv')
dataset.columns
dataset.shape
train=dataset.iloc[:320,:]
test=dataset.iloc[320:,:]

x_train=train[['Age', 'EstimatedSalary']]
x_train['bias']=1
y_train=train[['Purchased']]

x=x_train.values
y=y_train.values
m=len(y)

x=x.astype('float64')
y=y.astype('float64')
x[:,0]=np.divide((x[:,0]-np.average(x[:,0])),(max(x[:,0])-min(x[:,0])))
x[:,1]=np.divide((x[:,1]-np.average(x[:,1])),(max(x[:,1])-min(x[:,1])))


theta0=1
theta1=0
theta2=0

theta=[[theta2],[theta1],[theta0]]
theta=np.array(theta)

z=x @ theta
h=1/(1+np.exp(1)**((-1)*z))
d=h-y
j_train=(-1)*(sum(y*(np.log(h))+(1-y)*(np.log(1-h))))/m
alpha=0.01

for i in range(1000):
    utheta0=theta0-(alpha*sum(d))/m
    utheta1=theta1-(alpha*sum(np.multiply(d,(x[:,0].reshape(m,1)))))/m
    utheta2=theta2-(alpha*sum(np.multiply(d,(x[:,1].reshape(m,1)))))/m
  
    theta0=utheta0
    theta1=utheta1
    theta2=utheta2
    
    theta=[theta2,theta1,theta0]
    theta=np.array(theta)

    z=x @ theta
    h=1/(1+np.exp(1)**((-1)*z))
    d=h-y
    j_train=(-1)*(sum(y*(np.log(h))+(1-y)*(np.log(1-h))))/m

print(theta)
print(j_train)


x_test=test[['Age', 'EstimatedSalary']]
x_test['bias']=1
y_test=test[['Purchased']]

X=x_test.values
Y=y_test.values
X=X.astype('float64')
Y=Y.astype('float64')
X[:,0]=np.divide((X[:,0]-np.average(X[:,0])),(max(X[:,0])-min(X[:,0])))
X[:,1]=np.divide((X[:,1]-np.average(X[:,1])),(max(X[:,1])-min(X[:,1])))


y_pred=1/(1+np.exp(1)**((-1)*(X @ theta)))
J_test=(-1)*(sum(Y*(np.log(y_pred))+(1-Y)*(np.log(1-y_pred))))/len(y_test)

print(J_test)



