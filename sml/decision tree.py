#T.Vinita
#2241022012
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics # Import scikit-learn metrics module for accuracy calculation
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
df =pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
print(df.head())
print(df.columns)
print(df.describe())
print(df.dtypes)
print(df['Name'].value_counts())
print(df['Sex'].value_counts())
print(df['Ticket'].value_counts())
print(df['Cabin'].value_counts())
print(df['Embarked'].value_counts())
plt.hist(df['Survived'])
plt.show()
df_n=df[['PassengerId', 'Survived', 'Pclass', 'Name','Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin','Embarked']]
sns.pairplot(df_n, height=4, kind="reg",markers=".")
plt.show()
df = df.drop(columns = ['Cabin', 'Name', 'PassengerId','Ticket', 'SibSp'])
df.fillna(method='ffill', inplace=True)
df.fillna(method='ffill', inplace=True)
lab_enc = preprocessing.LabelEncoder()
df['Sex'] = lab_enc.fit_transform(df['Sex'])
df['Fare'] = lab_enc.fit_transform(df['Fare'])
df['Embarked'] = lab_enc.fit_transform(df['Embarked'])
corr = df.corr()
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr,cmap=cmap,
vmax=.3,square=True,linewidths=6, cbar_kws={"shrink": .5})
colormap = plt.cm.viridis
plt.show()
y = df['Survived']
X = df.drop(columns = ['Survived'])
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.33, random_state=42)
# Create Decision Tree classifier object
clf = DecisionTreeClassifier()
# Train Decision Tree Classifier
clf = clf.fit(X_train,y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


