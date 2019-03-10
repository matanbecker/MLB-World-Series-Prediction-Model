
# coding: utf-8

# In[249]:


import pandas as pd
import numpy as np


# In[250]:


df=pd.read_csv("/home/rsingla1/project_data_latest.csv")
df1=pd.read_csv("/home/rsingla1/project_data_latest.csv")
df1.to_csv("dataframe")
#df1.columns


# In[252]:


bins=[-1,2,6,10,11]
group_names=['0','1','2','3']
df1['performance']=pd.cut(df['playoff_wins'],bins,labels=group_names)
df1[['performance']]
df1[["playoff_wins","performance"]]
df2=df1[["performance"]]
df1.to_csv('out.csv')
#print(df['playoff_wins','performance'])


# In[253]:


#df1.drop(["playoff_wins"],axis=1,inplace=True)
df1.columns


# In[254]:


df1.drop(["round","playoff_wins","performance","Tm","world_series_ind"],axis=1,inplace=True)
df1.columns


# In[263]:


#Converting dataframe df1 and df2 into array df3 and df4
df3=df1.as_matrix()
b=np.array(df2)
df4=b.ravel()
df2


# In[2044]:


SelectKBest(chi2, k=2).fit_transform(df1,c)



# In[260]:


#use linear regression as the model for feature selection
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
name=df1.columns.values
name
lr = LinearRegression()
#rank all features, i.e continue the elimination until the last one
rfe = RFE(lr, n_features_to_select=1)
rfe.fit(df3,df4)
print ("Features sorted by their rank:")
print (sorted(zip(map(lambda x: round(x, 4), rfe.ranking_),name)))


# In[140]:



# Feature Selection using Extra Tree Classifier
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
# fit an Extra Trees model to the data
model = ExtraTreesClassifier()
model.fit(df3,df4)
#y=k_scores.append(x.mean())
# display the relative importance of each attribute
print (sorted(zip(map(lambda x: round(x, 4), model.feature_importances_),name), 
             reverse=True))


# In[124]:



from sklearn.ensemble import RandomForestRegressor
import numpy as np

X = df1
Y = df2

rf = RandomForestRegressor()
rf.fit(X, df4)
print ("Features sorted by their score:")
print (sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), name), 
             reverse=True))


# In[261]:


from sklearn import metrics
knn=KNeighborsClassifier(n_neighbors=13)
knn.fit(x_train,y_train)
y_pred=knn.predict(x_test)
print(metrics.accuracy_score(y_test,y_pred))


# In[370]:


#df5=df2[["performance"]]
#df5=df[["playoff_wins"]]
columns=['hit_OBP','hit_OPS','hit_SLG','hit_BA','pitch_WHIP','pitch_BB9','pitch_H9','pitch_HR9','pitch_FIP','pitch_WL',
         'hit_RG','pitch_ERA','pitch_SOW','pitch_SO9','pitch_W','pitch_GS','pitch_G','hit_G','pitch_RAG','pitch_PAge',
         'pitch_CG','pitch_GF','hit_BatAge','pitch_ER','pitch_H','pitch_BB','pitch_IP','pitch_cSho','pitch_BK','hit_H','hit_AB']
features=df1[list(columns)]
#from sklearn.cross_validation import train_test_split
#train_idx1,test_idx1=df1[:166],df1[166:]
x_train,x_test=features[:167],features[167:]
#x_train,x_test,y_train,y_test=train_test_split(features,df4,test_size=0.3)
y_train,y_test=df2[:167],df2[167:]
y_test


# In[371]:


new_x_train=x_train
new_x_test=x_test
new_y_train=y_train
new_y_test=y_test


# In[180]:


columns=['hit_OBP','hit_BA','hit_OPS','hit_SLG','pitch_WHIP']
features=df1[list(columns)].values
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(features,df4,test_size=0.3)
df1


# In[15]:


new_x_train=x_train
new_x_test=x_test
new_y_train=y_train
new_y_test=y_test


# In[155]:


from sklearn.neighbors import KNeighborsClassifier
import numpy as np
k_range=np.linspace(start=1,stop=15, num=10)

scores=[]
for k in k_range:

    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(new_x_train,new_y_train)
    y_pred=knn.predict(new_x_test)
    scores.append(metrics.accuracy_score(new_y_test,y_pred))
    
    
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(k_range,scores)
plt.xlabel('value of k for knn')
plt.ylabel('Testing accuracy')


# In[372]:


from sklearn.metrics import recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
logreg=LogisticRegression()
logreg.fit(new_x_train,new_y_train)
y_pred=logreg.predict(new_x_test)
print(metrics.accuracy_score(new_y_test,y_pred))
print(metrics.precision_score(new_y_test,y_pred,average='macro'))
print(metrics.recall_score(new_y_test, y_pred, average='macro'))
print(metrics.f1_score(new_y_test, y_pred, average='macro'))
print(new_y_test)
print(y_pred)



# In[374]:


#Applying knn on few features
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


from sklearn import metrics
knn=KNeighborsClassifier(n_neighbors=10)
knn.fit(new_x_train,new_y_train)
y_pred=knn.predict(new_x_test)
print(metrics.accuracy_score(new_y_test,y_pred))
print(metrics.precision_score(new_y_test,y_pred,average='macro'))
print(metrics.recall_score(new_y_test, y_pred, average='macro'))
print(metrics.f1_score(new_y_test, y_pred, average='macro'))
print(new_y_test)
print(y_pred)


# In[268]:


name1=df2.columns.values
name1
import itertools
from sklearn import svm
from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

   
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
plt.figure()
cnf_matrix = confusion_matrix(new_y_test, y_pred)
np.set_printoptions(precision=2)
plot_confusion_matrix(cnf_matrix, classes=[0,1,2,3],
                      title='Confusion matrix, without normalization')


# In[121]:


name1=df2.columns
name1


# In[179]:


## from sklearn import metrics applying knn with k-fold cross validation
from sklearn.metrics import precision_score
from sklearn.cross_validation import cross_val_score
columns=['hit_OBP','hit_BA','hit_OPS','hit_SLG','pitch_WHIP']
features1=df1[list(columns)].values
#from sklearn.cross_validation import train_test_split
#x_train,x_test,y_train,y_test=train_test_split(features1,c,test_size=0.3)
from sklearn import metrics
knn=KNeighborsClassifier(n_neighbors=60)
knn.fit(features1,df4)
print (cross_val_score(knn,features1,df4,cv=10,scoring="accuracy").mean())
print (cross_val_score(knn,features1,df4,cv=10,scoring="precision_macro").mean())


# In[1026]:


features1


# In[88]:


from sklearn.linear_model import LogisticRegression
columns=['hit_OBP','hit_BA','hit_OPS','hit_SLG','pitch_WHIP']
features=df1[list(columns)].values
#from sklearn.cross_validation import train_test_split
#x_train1,x_test1,y_train1,y_test1=train_test_split(features,c)
logreg=LogisticRegression()
logreg.fit(features,df4)
#y_pred=logreg.predict(x_test1)
print (cross_val_score(logreg,features,df4,cv=10,scoring="accuracy").mean())
print (cross_val_score(logreg,features,df4,cv=10,scoring="precision_macro").mean())


# In[869]:


c


# In[101]:


columns=['hit_OBP','hit_BA','hit_OPS','hit_SLG','pitch_WHIP']
features=df1[list(columns)].values
k_range=range(1,140)
k_scores=[]
for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    scores=cross_val_score(knn,features,df4,cv=10,scoring="accuracy")   
    k_scores.append(scores.mean())
    
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(k_range,k_scores)
plt.xlabel('value of k for knn')
plt.ylabel('Testing accuracy')


# In[374]:


Correlation=df1.corr()
Correlation.to_csv('CorrelationMatrix.csv')
df2


# In[762]:


from sklear.metrics import precision_score
precision_score(df3,vc)


# In[1318]:


from sklearn.metrics import confusion_matrix
confusion_matrix=confusion_matrix(y_test2.value,y_pred2)


precision_score(y_test2.values,y_pred,average='macro')
#it can be macro or weighted


# In[ ]:


import itertools
from sklearn import svm
from sklearn.metrics import confusion_matrix

# import some data to play with
iris = datasets.load_iris()

class_names = iris.target_names

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(d, y, random_state=0)

# Run classifier, using a model that is too regularized (C too low) to see
# the impact on the results
classifier = svm.SVC(kernel='linear', C=0.01)
y_pred = classifier.fit(X_train, y_train).predict(X_test)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

