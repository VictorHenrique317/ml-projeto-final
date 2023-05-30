import pandas as pd
from sklearn.neural_network import MLPClassifier

# Load data from CSV file
data = pd.read_csv('data.csv')

# Split data into features and target
X = data.drop('target', axis=1)
y = data['target']

# Create MLP classifier
clf = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000)

# Train the classifier
clf.fit(X, y)
