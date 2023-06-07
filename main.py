#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.metrics import accuracy_score


# Defining the features

# In[3]:


data = pd.read_csv('data.csv')

X_questions = data.iloc[:, 2:182]
X_questions = X_questions.drop('date_visit', axis=1)

X_drugs = data.iloc[:, 185:]

X = np.concatenate((X_questions, X_drugs), axis=1)
X = pd.DataFrame(X)

print(X_questions.shape)
print(X_drugs.shape)
print(X.shape)


# Defining the target

# In[36]:


Y = data.iloc[:, 182:185]

y_vas30 = Y.iloc[:, 0:1].values.ravel()
y_vas50 = Y.iloc[:, 1:2].values.ravel()
y_gic = Y.iloc[:, 2:3].values.ravel()

y_perceived = np.logical_or(y_vas30, y_vas50) # Perceived improvement by the patient is defined as either VAS30 or VAS50

# The target is defined as the intersection of perceived improvement and GIC,
# this is because the patient must perceive improvement and the doctor must agree
y = np.logical_and(y_perceived, y_gic)

print(f"The percentage of ones inside y_gic is {(np.sum(y_gic)/y_gic.shape[0])*100:.2f}%")
print(f"The percentage of ones inside y_vas30 is {(np.sum(y_vas30)/y_vas30.shape[0])*100:.2f}%")
print(f"The percentage of ones inside y_vas50 is {(np.sum(y_vas50)/y_vas50.shape[0])*100:.2f}%")
print()
print(f"The percentage of ones inside y_perceived is {(np.sum(y_perceived)/y_perceived.shape[0])*100:.2f}%")
print(f"The percentage of ones inside y is {(np.sum(y)/y.shape[0])*100:.2f}%")
# Faz sentido a porcentagem de 1's ser baixa em y, pois como dito em nossa reunião a maior parte
# dos pacientes que sofrem com dor crônica não apresentam melhora.


# In[38]:


similarity_vas50 = accuracy_score(y_gic, y_vas50) # y_gic is 'true', how similar is y_vas50 to it?
similarity_vas50 *= 100
print(f"The similarity percentage between y_gic and y_vas50 is {similarity_vas50:.2f}%")

similarity_vas30 = accuracy_score(y_gic, y_vas30) # y_gic is 'true', how similar is y_vas30 to it?
similarity_vas30 *= 100
print(f"The similarity percentage between y_gic and y_vas30 is {similarity_vas30:.2f}%")

similarity_perceived = accuracy_score(y_gic, y_perceived) # y_gic is 'true', how similar is y_perceived to it?
similarity_perceived *= 100
print(f"The similarity percentage between y_gic and y_perceived is {similarity_perceived:.2f}%")


# In[41]:


similarity_gic = accuracy_score(y, y_gic) # y is 'true', how similar is y_gic to it?
similarity_gic *= 100
print(f"The similarity percentage between y and y_gic is {similarity_gic:.2f}%")

similarity_vas50 = accuracy_score(y, y_vas50) # y is 'true', how similar is y_vas50 to it?
similarity_vas50 *= 100
print(f"The similarity percentage between y and y_vas50 is {similarity_vas50:.2f}%")

similarity_vas30 = accuracy_score(y, y_vas30) # y is 'true', how similar is y_vas30 to it?
similarity_vas30 *= 100
print(f"The similarity percentage between y and y_vas30 is {similarity_vas30:.2f}%")

similarity_perceived = accuracy_score(y, y_perceived) # y is 'true', how similar is y_perceived to it?
similarity_perceived *= 100
print(f"The similarity percentage between y and y_perceived is {similarity_perceived:.2f}%")


# Encoding the Boolean variables

# In[5]:


le = LabelEncoder()
for col in X_questions.columns:
    if X_questions[col].dtype == 'bool':
        X_questions[col] = le.fit_transform(X_questions[col])

for col in X_drugs.columns:
    if X_drugs[col].dtype == 'bool':
        X_drugs[col] = le.fit_transform(X_drugs[col])

for col in X.columns:
    if X[col].dtype == 'bool':
        X[col] = le.fit_transform(X[col])


# Imputing missing values

# In[6]:


imp = SimpleImputer(strategy='mean')
imp.fit(X_questions)
X_questions = imp.transform(X_questions)

imp = SimpleImputer(strategy='mean')
imp.fit(X_drugs)
X_drugs = imp.transform(X_drugs)

imp = SimpleImputer(strategy='mean')
imp.fit(X)
X = imp.transform(X)


# In[46]:


def trainReferenceMLP(X, x_name, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # 153 is the minimum number of features
    clf = MLPClassifier(hidden_layer_sizes=(153,), solver='sgd', learning_rate_init=0.001,
                        max_iter=400, random_state=42, verbose=False)
    clf.fit(X_train, y_train)

    score = clf.score(X_test, y_test)
    print(f'Test accuracy for y using {x_name}: {score*100:.2f}%')


# In[47]:


trainReferenceMLP(X_questions, 'X_questions', y)
trainReferenceMLP(X_drugs, 'X_drugs', y)
trainReferenceMLP(X, 'X_questions and X_drugs', y)

