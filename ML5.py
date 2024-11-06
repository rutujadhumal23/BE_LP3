# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score


# Step 2: Load the dataset

columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv('diabetes.csv', names=columns)


# Step 3: Convert all columns to numeric (since they are read as 'object' or strings)
df = df.apply(pd.to_numeric, errors='coerce')


# Step 4: Check for missing values and handle them (e.g., fill NaNs with the mean of the column)
df.fillna(df.mean(), inplace=True)


# Step 5: Feature set and target set
X = df.drop('Outcome', axis=1)  # Features
y = df['Outcome'].astype(int)  # Ensure the target 'Outcome' is integer (0 or 1)


# Step 6: Split the dataset into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Step 7: Normalize the train and test data using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Step 8: Initialize the KNN model and train it
k = 5  # Number of neighbors
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)  # Ensure y_train is categorical (0 or 1)


# Step 9: Predict on the test set
y_pred = knn.predict(X_test)


# Step 10: Compute Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)


# Step 11: Compute Accuracy, Error Rate, Precision, and Recall
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)


# Print evaluation metrics
print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'Error Rate: {error_rate * 100:.2f}%')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
