#K-Nearesst Neighbour 

#importing the Librares
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#set the index value as index_col 0
data=pd.read_csv('./Dataset/Classified Data',index_col=0)

#standardize the values
from sklearn.preprocessing import StandardScaler

#Create a object for StandardScaler
scaler=StandardScaler()

#fit and tranform the value to the standard values 
scaler.fit(data.drop('TARGET CLASS',axis=1))
scaled_features=scaler.transform(data.drop('TARGET CLASS',axis=1))

#This scaled data doesn't have index name and column name 
#store the scaled feature in a dataset

#columns will fil the columns index 
#and also neglecting the target class because that is
#independent feature 
df_feat=pd.DataFrame(scaled_features,columns=data.columns[:-1])

#sns.pairplot(data,hue='TARGET CLASS')

#Separate the train and test data 
from sklearn.model_selection import train_test_split
#first is independent feature ->input dependent feature -> output 
x_train,x_test,y_train,y_test=train_test_split(scaled_features,data['TARGET CLASS'])

#k_nearest Neighbour 
from sklearn.neighbors import KNeighborsClassifier
#Giving K=1
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train,y_train)


pred=knn.predict(x_test)

#use the confusion matrix to get the what are the values correctly perdicted 
#and unpredicted

from sklearn.metrics import classification_report,confusion_matrix

#here you can see which are all the values g
print(confusion_matrix(pred,y_test))

#Give reports accuracy and other scores 
print(classification_report(pred,y_test))

#Run the error rate find the point falls below and predict the K value 

error_rate=[]

for i in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    pred=knn.predict(x_test)
    error_rate.append(np.mean(pred!=y_test))
    
#plot the error rate vs K values for the error rate calculated
plt.figure(figsize=(10,6))

plt.plot(range(1,40),error_rate,linestyle="dashed",marker="o",markersize=10,
         markerfacecolor="red")
    
plt.title("Error Rate Graph")
plt.xlabel("K-value")
plt.ylabel("Error_Rate")

#here you can see that after k=24 it never touches back so choose that value

knn=KNeighborsClassifier(n_neighbors=24)
knn.fit(x_train,y_train)
pred=knn.predict(x_test)

#Confusion Matric
print(confusion_matrix(y_test,pred))
#classification report
print(classification_report(y_test,pred))

#You can see that accuracy has increased 














    
































