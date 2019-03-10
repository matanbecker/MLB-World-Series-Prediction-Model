import pandas as pd
import numpy as np

df=pd.read_csv("C:\Users\Matan\Desktop\MSA-MRM\MSA 8010\The-Players-master\project_data_team_all.csv")
df1=pd.read_csv("C:\Users\Matan\Desktop\MSA-MRM\MSA 8010\The-Players-master\project_data_team_all.csv")

df1.columns

bins=[-1,2,6,10,11]
group_names=['0','1','2','3']
df1['performance']=pd.cut(df['playoff_wins'],bins,labels=group_names)
df1[['playoff_wins','performance']]
#print(df['playoff_wins','performance'])

df1.columns

#df2=df[['playoff_wins','performance']]

#df1.drop(["round"],axis=1,inplace=True)
#df1
df2=df1["performance"]
df2
#df1.columns
#df3=df1["performance",{"excellent":4,"good":3,"average":2,"low":1}]
#df3
#df['performance']=df['performance'].convert_objects(convert_numeric=True).df3


df1.drop(["performance"],axis=1,inplace=True)
X,y=df1,df2
b=np.array(y)
c=b.ravel()
df3=df1.as_matrix()
c

#X_new=SelectKBest(chi2, k=2).fit_transform(X, c)

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
name=df1.columns.values
name

#use linear regression as the model


lr = LinearRegression()
#rank all features, i.e continue the elimination until the last one
rfe = RFE(lr, n_features_to_select=1)
rfe.fit(df3,c)
print ("Features sorted by their rank:")
print (sorted(zip(map(lambda x: round(x, 4), rfe.ranking_),name)))

# Feature Importance
from sklearn import datasets
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
# load the iris datasets

# fit an Extra Trees model to the data
model = ExtraTreesClassifier()
model.fit(df3,c)
# display the relative importance of each attribute
print(model.feature_importances_)

#Applying K- Nearest Neighbour Algorithm on our top 10 features
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=1)

print(knn)

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(df3,c,test_size=0.4)

from sklearn import metrics
knn=KNeighborsClassifier(n_neighbors=23)
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)
print(metrics.accuracy_score(y_test,y_pred))

from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg=LogisticRegression()
logreg.fit(x_train,y_train)
y_pred=logreg.predict(x_test)
print(metrics.accuracy_score(y_test,y_pred))

k_range=range(1,100)
scores=[]
for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    y_pred=knn.predict(x_test)
    scores.append(metrics.accuracy_score(y_test,y_pred))
    
import matplotlib.pyplot as plt
%matplotlib inline
plt.plot(k_range,scores)
plt.xlabel('value of k for knn')
plt.ylabel('Testing accuracy')

#Applying knn on few features
columns=['hit_BA','hit_OPS','hit_SLG','pitch_WHIP','pitch_BB9']
features=df1[list(columns)].values
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(features,c,test_size=0.4)
from sklearn import metrics
knn=KNeighborsClassifier(n_neighbors=13)
knn.fit(features,c)
y_pred=knn.predict(x_test)
print(metrics.accuracy_score(y_test,y_pred))

## from sklearn import metrics
from sklearn.cross_validation import cross_val_score
knn=KNeighborsClassifier(n_neighbors=13)
columns=['hit_BA','hit_OPS','hit_SLG','pitch_WHIP','pitch_BB9','pitch_H9','pitch_WL','pitch_HR9','pitch_FIP']
features=df1[list(columns)].values
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(features,c,test_size=0.4)
from sklearn import metrics
knn=KNeighborsClassifier(n_neighbors=13)
knn.fit(features,c)
print (cross_val_score(knn,features,c,cv=10,scoring="accuracy").mean())

from sklearn.linear_model import LogisticRegression
columns=['hit_BA','hit_OPS','hit_SLG']
features=df1[list(columns)].values
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(features,c)
logreg=LogisticRegression()
logreg.fit(features,c)
y_pred=logreg.predict(x_test)
print (cross_val_score(logreg,features,c,cv=10,scoring="accuracy").mean())

c

k_range=range(1,120)
k_scores=[]
for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    scores=cross_val_score(knn,df3,c,cv=10,scoring="accuracy")   
    k_scores.append(scores.mean())
    
import matplotlib.pyplot as plt
%matplotlib inline
plt.plot(k_range,k_scores)
plt.xlabel('value of k for knn')
plt.ylabel('Testing accuracy')

Correlation=df1.corr()
Correlation.to_csv('CorrelationMatrix.csv')
df2



