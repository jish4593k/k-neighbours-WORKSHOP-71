import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.decomposition import PCA

# Importing the dataset
datasets = pd.read_csv('Social_Network_Ads.csv')
X = datasets.iloc[:, [2, 3]].values
Y = datasets.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.25, random_state=0)

# Feature Scaling
sc_X = StandardScaler()
X_Train = sc_X.fit_transform(X_Train)
X_Test = sc_X.transform(X_Test)

# Fitting the classifier into the Training set (K Nearest Neighbors)
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_Train, Y_Train)

# Predicting the Test set results
Y_Pred = classifier.predict(X_Test)

# Making the Confusion Matrix
cm = confusion_matrix(Y_Test, Y_Pred)
print("Confusion Matrix:\n", cm)

# Print Classification Report
print("Classification Report:\n", classification_report(Y_Test, Y_Pred))

# Visualizing the Confusion Matrix with Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Not Purchased', 'Purchased'],
            yticklabels=['Not Purchased', 'Purchased'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Principal Component Analysis (PCA) for Dimensionality Reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
X_Train_pca, X_Test_pca, _, _ = train_test_split(X_pca, Y, test_size=0.25, random_state=0)

# Visualizing the Dataset after PCA
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=Y, palette='viridis')
plt.title('Dataset after PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Neural Network Classifier using TensorFlow and Keras
model = Sequential([
    Dense(units=6, activation='relu', input_dim=2),
    Dense(units=1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_Train_pca, Y_Train, epochs=50, batch_size=10, verbose=0)

# Evaluating the Neural Network on Test set
_, accuracy = model.evaluate(X_Test_pca, Y_Test)
print(f'Neural Network Accuracy: {accuracy * 100:.2f}%')
