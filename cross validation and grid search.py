#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


df= pd.read_csv("cancer.csv")


# In[4]:


df


# In[52]:


#x=df.drop(['diagnosis','id'],axis=1)#
x=df.iloc[:,[2,3,4,5,6]]
type(x)


# In[53]:


y=df['diagnosis']


# In[65]:


y


# In[55]:


from sklearn.model_selection import train_test_split


# In[56]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[57]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


# In[124]:


def summarize_classification(y_test,y_pred):
    acc=accuracy_score(y_test,y_pred,normalize=True)
    num_acc=accuracy_score(y_test,y_pred,normalize=False)
    prec=precision_score(y_test,y_pred, pos_label='M')
    
    recall=recall_score(y_test,y_pred, pos_label='M')
    print("test data count",len(y_test))
    print("accuracy_count",num_acc)
    print("accuracy_score",acc)
    print("precision_score",prec)
    print("recall_score",recall)


# In[125]:


from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier


# In[126]:


parameters={'max_depth':[1,2,3,4,5,6,7,8,9,10,11,12]}
grid_search=GridSearchCV(DecisionTreeClassifier(),parameters,cv=3,return_train_score=True)
grid_search.fit(x_train,y_train)
grid_search.best_params_


# In[127]:


decision_tree_model=DecisionTreeClassifier(max_depth=grid_search.best_params_['max_depth']).fit(x_train,y_train)


# In[128]:


y_pred=decision_tree_model.predict(x_test)
#print(type(y_pred),type(y_test))
y_test.values


# In[129]:


summarize_classification(y_test.values,y_pred)


# In[130]:


from sklearn.model_selection import cross_validate


# In[132]:


classifer= DecisionTreeClassifier()
classifer.fit(x_train,y_train)
results=cross_validate(classifer,x_train,y_train,cv=10)
results


# In[135]:


print('test_mean_score',results['test_score'].mean())


# In[ ]:




