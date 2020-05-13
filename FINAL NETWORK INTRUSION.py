#!/usr/bin/env python
# coding: utf-8

# In[1]:



get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
import imblearn

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')


# In[3]:


train = pd.read_csv("Train_data.csv")
test = pd.read_csv("Test_data.csv")


# In[5]:


train.columns


# In[6]:



train.dtypes


# In[7]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
cols = train.select_dtypes(include=['float64','int64']).columns
sc_train = scaler.fit_transform(train.select_dtypes(include=['float64','int64']))
sc_test = scaler.fit_transform(test.select_dtypes(include=['float64','int64']))
sc_traindf = pd.DataFrame(sc_train, columns = cols)
sc_testdf = pd.DataFrame(sc_test, columns = cols)


# In[8]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
cattrain = train.select_dtypes(include=['object']).copy()
cattest = test.select_dtypes(include=['object']).copy()
traincat = cattrain.apply(encoder.fit_transform)
testcat = cattest.apply(encoder.fit_transform)
enctrain = traincat.drop(['class'], axis=1)
cat_Ytrain = traincat[['class']].copy()


# In[9]:


train_x = pd.concat([sc_traindf,enctrain],axis=1)
train_y = train['class']
train_x.shape


# In[10]:


test_df = pd.concat([sc_testdf,testcat],axis=1)
test_df.shape


# In[11]:


train_x


# In[12]:


import pandas as pd
import numpy as np
import seaborn as sns
X = train_x.iloc[:,0:20]  #independent columns
y = train_x.iloc[:,-1]    #target column i.e price range
#get correlations of each features in dataset
corrmat = train_x.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(train_x[top_corr_features].corr(),annot=True,cmap="Blues")


# In[13]:


from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(train_x,train_y,train_size=0.70, random_state=2)


# In[14]:



Model_Score = []
Model_Name = []
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC 
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB 
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score ,precision_score,f1_score


# ## KNN

# In[15]:


KNN_Classifier = KNeighborsClassifier(n_jobs=-1)
KNN_Classifier.fit(X_train, Y_train); 

knn_pred = KNN_Classifier.predict(X_test)
print("-----------------------------------------KNN Classifier--------------------------------------")
print(confusion_matrix(Y_test,knn_pred))
print(classification_report(Y_test,knn_pred))
print(accuracy_score(Y_test,knn_pred))

Model_Name.append('KNN Classifier')
Model_Score.append((accuracy_score(Y_test, knn_pred)))


# # LOGISTIC REGRESSSION

# In[17]:



LGR_Classifier = LogisticRegression(n_jobs=-1, random_state=0)
LGR_Classifier.fit(X_train, Y_train);


# In[93]:


lr_pred = LGR_Classifier.predict(X_test)
print("-----------------------------------------Logistic Regression--------------------------------------")
print(confusion_matrix(Y_test,lr_pred))
print(classification_report(Y_test,lr_pred))
print(accuracy_score(Y_test,lr_pred))

Model_Name.append('Logistic Regression')
Model_Score.append((accuracy_score(Y_test, lr_pred)))


# # MULTILAYER PERCEPTRON

# In[18]:



MLP = MLPClassifier(alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
MLP.fit(X_train,Y_train)

mlp_pred = MLP.predict(X_test)
print("-----------------------------------------MLP--------------------------------------")
print(confusion_matrix(Y_test,mlp_pred))
print(classification_report(Y_test,mlp_pred))
print(accuracy_score(Y_test,mlp_pred))

Model_Name.append('MLP')
Model_Score.append((accuracy_score(Y_test, mlp_pred)))


# # NAIVE BAYES

# In[19]:



BNB_Classifier = BernoulliNB()
BNB_Classifier.fit(X_train, Y_train)

bnb_pred = BNB_Classifier.predict(X_test)
print("-----------------------------------------Naive Bayes Classifier--------------------------------------")
print(confusion_matrix(Y_test,bnb_pred))
print(classification_report(Y_test,bnb_pred))
print(accuracy_score(Y_test,bnb_pred))

Model_Name.append('Naive Bayes')
Model_Score.append((accuracy_score(Y_test, bnb_pred)))


# In[20]:



DTC_Classifier = tree.DecisionTreeClassifier(criterion='entropy', random_state=0)
DTC_Classifier.fit(X_train, Y_train)

dtc_pred = DTC_Classifier.predict(X_test)
print("-----------------------------------------Decision Tree Classifier--------------------------------------")
print(confusion_matrix(Y_test,dtc_pred))
print(classification_report(Y_test,dtc_pred))
print(accuracy_score(Y_test,dtc_pred))

Model_Name.append('Decision Tree')
Model_Score.append((accuracy_score(Y_test, dtc_pred)))


# In[21]:



SVM_Classifier = svm.SVC(kernel='linear') # Linear Kernel
SVM_Classifier.fit(X_train, Y_train)

svm_pred = SVM_Classifier.predict(X_test)
print("-----------------------------------------SVM Classifier--------------------------------------")
print(confusion_matrix(Y_test,svm_pred))
print(classification_report(Y_test,svm_pred))
print(accuracy_score(Y_test,svm_pred))

Model_Name.append('SVM Classifier')
Model_Score.append((accuracy_score(Y_test, svm_pred)))


# In[23]:


RF_Classifier = RandomForestClassifier(n_estimators=50)
RF_Classifier.fit(X_train, Y_train); 

rfc_pred = RF_Classifier.predict(X_test)
print("-----------------------------------------RFC Classifier--------------------------------------")
print(confusion_matrix(Y_test,rfc_pred))
print(classification_report(Y_test,rfc_pred))
print(accuracy_score(Y_test,rfc_pred))

Model_Name.append('RF Classifier')
Model_Score.append((accuracy_score(Y_test, rfc_pred)))


# In[24]:


plt.figure(figsize = (15, 10))
plt.plot(Model_Name,Model_Score, marker = 'o', color = 'red')
plt.title('Comparison of different models')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.ylim(0.9, 1.0)
plt.grid()
plt.savefig('Model_compare.jpeg')
plt.show()

