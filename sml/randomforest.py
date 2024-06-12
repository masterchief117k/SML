from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
df = pd.read_csv('https://gist.githubusercontent.com/tijptjik/9408623/raw/b237fa5848349a14a14e5d4107dc7897c21951f5/wine.csv')
print(df.head())
attributes = ["Class", "Alcohol", "Malic acid", "Ash","Alcalinity of ash", "Magnesium", "Total phenols","Flavanoids", "Nonflavanoid phenols", "Proanthocyanins","Color intensity", "Hue", "OD280/OD315 of diluted wines","Proline"]
df.columns = attributes
print(df.head())
print(df.dtypes)
print(df.describe())
plt.hist(df['Alcohol'])
plt.show()
plt.hist(df['Ash'])
plt.show()
plt.hist(df['Class'])
plt.show()
df_n=df[["Class", "Alcohol", "Malic acid", "Flavanoids","Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue", "OD280/OD315 of diluted wines","Proline"]]
sns.pairplot(df_n, height=4, kind="reg",markers=".")
plt.show()
corr = df.corr()
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr,cmap=cmap,vmax=.3,square=True,linewidths=6, cbar_kws={"shrink": .5})
colormap = plt.cm.viridis
plt.show()
plt.figure(figsize=(12,12))
plt.title('Pearson Correlation of Features', y=1.05,
size=15)
sns.heatmap(df.corr(),linewidths=0.1,vmax=1.0, square=True,
cmap=colormap, linecolor='white',annot=True)
plt.show()
df = df.drop(columns=['Flavanoids'])
Y = df['Class']
X = df.drop(columns=['Class'])
X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.33, random_state=42)
classifier = RandomForestClassifier(n_jobs=2,random_state=42)
classifier.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = classifier.predict(X_test)
# Model Accuracy
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
confusion_matrix(y_test, y_pred, labels=None,sample_weight=None)
print(confusion_matrix)