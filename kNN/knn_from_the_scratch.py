import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('Social_Network_Ads.csv')
dataset.columns
X=dataset[['Age', 'EstimatedSalary']]
Y=dataset[['Purchased']]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.25,random_state=0)

x_train=x_train.values
x_test=x_test.values
y_train=y_train.values
y_test=y_test.values

x_train=x_train.astype('float64')
y_train=y_train.astype('float64')
x_test=x_test.astype('float64')
y_test=y_test.astype('float64')

x_train[:,0]=np.divide((x_train[:,0]-np.average(x_train[:,0])),(max(x_train[:,0])-min(x_train[:,0])))
x_train[:,1]=np.divide((x_train[:,1]-np.average(x_train[:,1])),(max(x_train[:,1])-min(x_train[:,1])))
x_test[:,0]=np.divide((x_test[:,0]-np.average(x_test[:,0])),(max(x_test[:,0])-min(x_test[:,0])))
x_test[:,1]=np.divide((x_test[:,1]-np.average(x_test[:,1])),(max(x_test[:,1])-min(x_test[:,1])))



def Distance(test_point):
    squares=(x_train-test_point)**2
    distance=np.sqrt(squares[:,0]+squares[:,1])
    return distance
    

def Predict(K,distance_array,y_train_):
    count1=0
    count0=0
    v=0
    if(min(distance_array)==0):
         v=np.argmin(distance_array)
         if(y_train_[v]==1):
             return 1 
         else:
             return 0 
    else:
        for i in range(K):
            v=np.argmin(distance_array)
            if(y_train_[v]==1):
                count1=count1+1
            else:
                count0=count0+1
            distance_array[v]=5
        if(count1>count0):
             return 1  
        else:
             return 0
       
y_pred=[]
for i in range(len(x_test)):
    y_pred=y_pred+[[Predict(5,Distance(x_test[i]),y_train)]]
y_pred=np.array(y_pred)
y_pred=y_pred.astype('float64')  

no_of_correct=0
for j in range(len(x_test)): 
    if(y_pred[j]==y_test[j]):
        no_of_correct= no_of_correct+1
accuracy= no_of_correct/len(x_test)
print(accuracy)

































