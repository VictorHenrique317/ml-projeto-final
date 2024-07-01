import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
import numpy as np
from IPython.core.display import Image
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import xgboost as xgb
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from matplotlib.image import imread
from sklearn.model_selection import cross_val_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

data = pd.read_csv('data.csv')

X_questions = data.iloc[:, 2:182] # Id e data de nascimento são irrelevantes
X_questions = X_questions.drop('date_visit', axis=1) # Data de visita não é relevante
X_questions = X_questions.drop(X_questions.columns[[46, 133, 158, 161]], axis=1) # Essas colunas são constantes

X_drugs = data.iloc[:, 185:]
X_drugs = X_drugs.drop(X_drugs.columns[[50,51,61,92,101,111,114,121,137,140,141,
                                        142,143,148,151,152]], axis=1) # Essas colunas são constantes

X_random = np.random.rand(X_questions.shape[0], 1) # Para comparar a perfomance do modelo
X_random = pd.DataFrame(X_random)
X = np.concatenate((X_questions, X_drugs), axis=1) # A junção das duas tabelas
X = pd.DataFrame(X)

Y = data.iloc[:, 182:185]

y_vas30 = pd.DataFrame(Y.iloc[:, 0:1].values.ravel())
y_vas50 = pd.DataFrame(Y.iloc[:, 1:2].values.ravel())
y_gic = pd.DataFrame(Y.iloc[:, 2:3].values.ravel())

y_perceived = np.logical_or(y_vas30, y_vas50)
y_perceived = y_perceived.astype(int)
y_perceived = pd.DataFrame(y_perceived)

y = np.logical_and(y_perceived, y_gic)
y = y.astype(int)
y = pd.DataFrame(y)

np.random.seed(42) 

zero_rows = y.index[(y == 0).all(axis=1)]
# delete_rows = np.random.choice(zero_rows, size=int(len(zero_rows)/1000), replace=False)
delete_rows = np.random.choice(zero_rows, size=int(len(zero_rows)/1.25), replace=False)
# delete_rows = np.random.choice(zero_rows, size=int(len(zero_rows)/1.04), replace=False)

X = X.drop(delete_rows)
X_drugs = X_drugs.drop(delete_rows)
X_questions = X_questions.drop(delete_rows)
X_random = np.delete(X_random, delete_rows, axis=0)

y_gic = y_gic.drop(delete_rows)
y_vas30 = y_vas30.drop(delete_rows)
y_vas50 = y_vas50.drop(delete_rows)
y_perceived = y_perceived.drop(delete_rows)
y = y.drop(delete_rows)

# Codificando as variáveis categóricas
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

# Imputando os valores que faltam
imp = SimpleImputer(strategy='mean')
imp.fit(X_questions)
X_questions = imp.transform(X_questions)

imp = SimpleImputer(strategy='mean')
imp.fit(X_drugs)
X_drugs = imp.transform(X_drugs)

imp = SimpleImputer(strategy='mean')
imp.fit(X)
X = imp.transform(X)

# Normalizando os dados
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_drugs = scaler.fit_transform(X_drugs)
X_questions = scaler.fit_transform(X_questions)
X_random = scaler.fit_transform(X_random)

X = pd.DataFrame(X)
X_drugs = pd.DataFrame(X_drugs)
X_questions = pd.DataFrame(X_questions)
X_random = pd.DataFrame(X_random)

y_gic = pd.DataFrame(y_gic)
y_vas30 = pd.DataFrame(y_vas30)
y_vas50 = pd.DataFrame(y_vas50)
y_perceived = pd.DataFrame(y_perceived)
y = pd.DataFrame(y)

data = pd.concat([X, y], axis=1)
print(data)
data.to_csv('preprocessed_data.csv', index=False)
