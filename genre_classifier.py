#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import io
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from google.colab import files
uploaded = files.upload()


# In[ ]:


#Read in dataset
songs = pd.read_csv(io.BytesIO(uploaded['song_data.csv']))


# In[ ]:


# Split data into train/test
X = songs.drop('Genre', axis = 1)
y = songs['Genre']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

# Scale and transform data
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# ## Hyperparameter tuning with GridSearchCV
# 

# In[ ]:


grid_param = {'C': [0.01, 0.1, 1, 10, 100], 'gamma': [0.1, 0.01, 0.001], 'kernel': ['rbf']}
grid = GridSearchCV(SVC(), grid_param, refit = True, verbose = 2, n_jobs = -1, cv = 5)
grid.fit(X_train, y_train)

y_pred = grid.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# ## Implementing the default SVC using a linear kernel
# 
# 

# In[ ]:


vanilla_clf = svm.SVC(kernel = 'linear', verbose = True)
vanilla_clf.fit(X_train, y_train)

y_pred = vanilla_clf.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


# ## Building a random forest 

# In[ ]:


clf = RandomForestClassifier(min_samples_leaf = 3, max_depth = 10)
clf.fit(X_train, y_train)
print(clf.feature_importances_)


# In[ ]:


y_pred = clf.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

