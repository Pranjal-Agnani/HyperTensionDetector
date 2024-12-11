import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv("/hypertension_data.csv")

data['target'] = data['target'].astype('category')

def normalize(column):
    return (column - column.min()) / (column.max() - column.min())

data_normalized = data.copy()
for col in data.columns[:-1]:  # Assuming the target is the last column
    data_normalized[col] = normalize(data[col])

X = data_normalized.iloc[:, :-1]  # Features
y = data_normalized.iloc[:, -1]   # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

knn_predictions = knn.predict(X_test)

accuracy = accuracy_score(y_test, knn_predictions)
print(f"Accuracy: {accuracy * 100:.2f} %")

conf_matrix = confusion_matrix(y_test, knn_predictions)
print("\nConfusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(classification_report(y_test, knn_predictions))
