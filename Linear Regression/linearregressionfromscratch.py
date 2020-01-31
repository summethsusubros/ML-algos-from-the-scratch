#importing libraries
import pandas  as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt

#importing dataset and spilting it accordingly
dataset= pd.read_csv("Salary_Data.csv")
train=dataset.iloc[:25]
test=dataset.iloc[25:]

train['bais']=1
X_train=train[['YearsExperience','bais']]
y_train=train[['Salary']]


X=X_train.values
Y=y_train.values

#creating theta
theta1=1
theta0=0

#tdf=pd.DataFrame(theta)
theta=[[theta1],[theta0]]
theta=np.array(theta)

#finding the hypothesis
T=theta.T
H=X @ theta

#required values for cost function
D=H-Y
m=len(y_train)

#finding the cost function
J=np.sum((D**2))/m
j=J
alpha=0.005

#creating a function for gradiemt decent
utheta1=1
utheta0=0
#def greadient_descent(alpha=0.01,theta0,theta1):
for i in range(1000):
    utheta0=theta0-(alpha*sum(D))/m
    utheta1=theta1-(alpha*sum(np.multiply(D,(X[:,0].reshape(m,1)))))/m
    theta1=utheta1
    theta0=utheta0
    theta=[theta1,theta0]
    theta=np.array(theta)
    H=X @ theta
    D=H-Y      
    J=np.sum((D**2))/m
    plt.scatter(J,i)
  
print(theta)       
print(J)       


#test set prediction
X_test=test[['YearsExperience']].values
y_test=test[['Salary']].values
y_pred=theta0+theta1*X_test
￼

#test cost 
Jtest=(sum((y_pred - y_test)**2))/m

print(Jtest)

plt.scatter(train[['YearsExperience']],Y)
plt.plot(train[['YearsExperience']],H)
plt.scatter(X_test,y_test)
plt.scatter(X_test,y_pred)
plt.show()

'''from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure()
Axes3D.plot(theta0,theta1,H)'''


plt.plot(J,theta0)
plt.plot(J,theta1)





























