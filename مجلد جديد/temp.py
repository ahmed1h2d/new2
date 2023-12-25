# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv('economic_data.csv')
print(data)

print(data.describe())  

plt.scatter(data['Year'],data['GDP'])
plt.show()
#y=mx+b
print(data.head())
x=data.iloc[:,:1]
y=data.iloc[:,1]

print(x)
#
print(y)

from sklearn.linear_model import LinearRegression

model=LinearRegression()

model.fit(x,y)

print(model.coef_)
print(model.intercept_)

plt.scatter(x,y)
plt.plot(x,model.predict(x),'g')

model.predict([[2009]])
model.predict([[2014]])
model.predict([[2080]])
model.predict([[3000]])
model.score(x,y)








