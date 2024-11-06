# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import seaborn as sns
import matplotlib.pyplot as plt


# Step 2: Load the dataset

df = pd.read_csv('Churn_Modelling.csv')


# Optional: Drop unnecessary columns (CustomerId, Surname, etc.)
df.drop(['CustomerId', 'Surname'], axis=1, inplace=True)


# Step 3: Feature set and target set
X = df[['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']]
y = df['Exited']  # Target variable


# Convert categorical data (Geography, Gender) to numeric using one-hot encoding
X = pd.get_dummies(X, columns=['Geography', 'Gender'], drop_first=True)


# Step 4: Split the dataset into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Step 5: Normalize the train and test data using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Step 6: Initialize and build the neural network model
model = Sequential()


# Input layer
model.add(Dense(units=64, activation='relu', input_shape=(X_train.shape[1],)))


# Hidden layers
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.5))  # Dropout for regularization


model.add(Dense(units=16, activation='relu'))


# Output layer
model.add(Dense(units=1, activation='sigmoid'))  # Binary classification


# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 7: Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)


# Step 8: Evaluate the model
y_pred = (model.predict(X_test) > 0.5).astype("int32")


# Step 9: Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')


# Step 10: Print confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)


# Step 11: Visualize the confusion matrix (optional)



sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# Step 12: Count customers predicted to leave
count_leave = np.sum(y_pred)
print(f"Count of customers predicted to leave the bank: {count_leave}")



