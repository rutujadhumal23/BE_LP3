import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix


# Load the dataset

data = pd.read_csv('emails.csv')

# Data preprocessing
print(data.shape)
print(data.head())

# Separate features and labels
X = data.iloc[:, 1:-1].values  # All columns except the first (email name) and the last (label)
y = data.iloc[:, -1].values    # Last column is the label


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the data (important for SVM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# create object K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)

ConfusionMatrixDisplay.from_predictions(y_test,y_pred_knn)
plt.show()

# Support Vector Machine
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)


# Evaluate the models
print("K-Nearest Neighbors (KNN) Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_knn)}")
print("Classification Report:\n", classification_report(y_test, y_pred_knn))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))


print("\nSupport Vector Machine (SVM) Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_svm)}")
print("Classification Report:\n", classification_report(y_test, y_pred_svm))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))


# Compare the models
if accuracy_score(y_test, y_pred_knn) > accuracy_score(y_test, y_pred_svm):
    print("\nKNN performed better.")
else:
    print("\nSVM performed better.")
