import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df=pd.read_csv("swedish_insurance.csv")
print(df.info())
x=np.array(df['X'])
y=np.array(df['Y'])
X=x.reshape((-1,1))
Y=x.reshape((-1,1))
plt.scatter(X,Y)
x_train,x_test,y_train,y_test = train_test_split(X,Y, train_size=0.75 , test_size=0.25, random_state=4)
model=LinearRegression()
model.fit(x_train,y_train)
y_prediction = model.predict(x_test)
plt.plot(x_train,y_prediction)