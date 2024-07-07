import numpy as np
import pandas as pd

# KNN Classifier
class KNNclassifier:
    def __init__(self, k=5):
        self.k = k
    
    def fit(self, x, y):
        self.x_train = x
        self.y_train = y
    
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))
    
    def predict(self, X):
        y_pred = [self._predict_single(x) for x in X]
        return np.array(y_pred)
    
    def _predict_single(self, x):
        distances = [self.euclidean_distance(x, x_train) for x_train in self.x_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common

# Load and preprocess the data
data_path = r'D:\KNNapplication\breast-cancer-wisconsin.data.txt'
column_names = [
    "ID", "Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape",
    "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei",
    "Bland Chromatin", "Normal Nucleoli", "Mitoses", "Class"
]
data = pd.read_csv(data_path, header=None, names=column_names)

# Replace missing values represented by '?' with NaN and drop rows with NaN values
data.replace('?', np.nan, inplace=True)
data.dropna(inplace=True)

# Convert data to numeric
data = data.apply(pd.to_numeric)

# Split features and labels
X = data.drop(columns=['ID', 'Class']).values
y = data['Class'].values

# Convert class labels to binary (2: benign, 4: malignant)
y = (y == 4).astype(int)

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the KNN classifier
knn = KNNclassifier(k=5)
knn.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = np.sum(y_pred == y_test) / len(y_test)
print(f"Accuracy: {accuracy:.2f}")
