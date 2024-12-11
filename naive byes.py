import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv("/hypertension_data.csv")

data['target'] = data['target'].astype('category')

X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

nb_predictions = nb_model.predict(X_test)

accuracy = accuracy_score(y_test, nb_predictions)
print(f"Accuracy: {accuracy * 100:.2f} %")

conf_matrix = confusion_matrix(y_test, nb_predictions)
print("\nConfusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(classification_report(y_test, nb_predictions))
