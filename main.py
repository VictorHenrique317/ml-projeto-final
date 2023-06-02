import pandas as pd
from sklearn.neural_network import MLPClassifier

data = pd.read_csv('data.csv')

X_questions = data.iloc[:, 2:182]
print(X_questions)
X_drugs = data.iloc[:, 185:]

y_vas30 = data.iloc[:, 182:185]
y = data.iloc[:, 182:185]
y = data.iloc[:, 182:185]

# clf = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000)
# clf.fit(X_questions, y)
