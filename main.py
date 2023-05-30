import pandas as pd
from sklearn.neural_network import MLPClassifier



# Load data from CSV file
data = pd.read_csv('data.csv')
vas_30_index = data.columns.get_loc("vas_30")
print(vas_30_index)

# # Split data into features and target
X_questions = data.iloc[:, :182]
X_drugs = data.iloc[:, 185:]
# y_vas30 = data.iloc[:, 182:185]
# y = data.iloc[:, 182:185]
y = data.iloc[:, 182:185]

print(X_questions.iloc[0])
# # Create MLP classifier
# clf = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000)

# # Train the classifier
# clf.fit(X, y)
