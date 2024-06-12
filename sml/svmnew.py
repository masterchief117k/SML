#Name:T.Vinita
#Regd no=2241022012

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import svm

# Read the dataset
df = pd.read_csv('Iris.csv')

# Print the first few rows of the DataFrame
print("First few rows of the DataFrame:")
print(df.head())

# Rename the columns
df.columns = ['ID', 'sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

# Display statistical insights
print("\nStatistical insights of the dataset:")
print(df.describe())

# Visualize 'sepal-length' vs 'class' using boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(x='class', y='sepal_length', data=df)
plt.title("Boxplot between the class and sepal length")
plt.show()

# Visualize 'sepal-width' vs 'class' using boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(x='class', y='sepal_width', data=df)
plt.title("Boxplot between the class and sepal width")
plt.show()

# Plot a bar graph representing the distribution of classes
plt.figure(figsize=(8, 6))
df['class'].value_counts().plot.bar()
plt.title("Bar plot depicting the distribution of the target variable")
plt.show()

# Separate features and target variable
y = df['class']
X = df.drop(columns=['class'])

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Define a function to train and evaluate SVM models
def train_and_evaluate_svm(kernel):
    model = svm.SVC(kernel=kernel)
    model.fit(X_train, y_train)
    y_predicted = model.predict(X_test)
    print(f"\nConfusion Matrix for {kernel.capitalize()} Kernel SVM:")
    print(confusion_matrix(y_test, y_predicted))

# Train and evaluate SVM models with different kernels
train_and_evaluate_svm('linear')
train_and_evaluate_svm('poly')
train_and_evaluate_svm('rbf')