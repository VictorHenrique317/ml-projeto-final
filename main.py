import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
import numpy as np

data = pd.read_csv('data.csv')

# Defining the features and the target
X_questions = data.iloc[:, 2:182]
X_questions = X_questions.drop('date_visit', axis=1)

X_drugs = data.iloc[:, 185:]

Y = data.iloc[:, 182:185]
y_vas30 = Y.iloc[:, 0:1]
y_vas50 = Y.iloc[:, 1:2]
y_gic = Y.iloc[:, 2:3].values.ravel()

# Encoding the Boolean variables
le = LabelEncoder()
for col in X_questions.columns:
    if X_questions[col].dtype == 'bool':
        X_questions[col] = le.fit_transform(X_questions[col])

for col in X_drugs.columns:
    if X_drugs[col].dtype == 'bool':
        X_drugs[col] = le.fit_transform(X_drugs[col])

# Imputing missing values
imp = SimpleImputer(strategy='mean')
imp.fit(X_questions)
X_questions = imp.transform(X_questions)

# Training the reference model using X_questions
X_train, X_test, y_train, y_test = train_test_split(X_questions, y_gic, test_size=0.25, random_state=42)

clf = MLPClassifier(hidden_layer_sizes=(179), solver='sgd', learning_rate_init=0.001, 
                    max_iter=200, random_state=42, verbose=False)
clf.fit(X_train, y_train)

score = clf.score(X_test, y_test)
print(f'Test accuracy for y_gic using X_questions: {score:.2f}')

# Training the reference model using X_drugs
X_train, X_test, y_train, y_test = train_test_split(X_drugs, y_gic, test_size=0.25, random_state=42)

clf = MLPClassifier(hidden_layer_sizes=(179), solver='sgd', learning_rate_init=0.001,
                    max_iter=200, random_state=42, verbose=False)
clf.fit(X_train, y_train)

score = clf.score(X_test, y_test)
print(f'Test accuracy for y_gic using X_drugs: {score:.2f}')

# Training the reference model using X_questions and X_drugs
X = np.concatenate((X_questions, X_drugs), axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y_gic, test_size=0.25, random_state=42)

clf = MLPClassifier(hidden_layer_sizes=(358), solver='sgd', learning_rate_init=0.001,
                    max_iter=200, random_state=42, verbose=False)
clf.fit(X_train, y_train)

score = clf.score(X_test, y_test)
print(f'Test accuracy for y_gic using X_questions and X_drugs: {score:.2f}')
