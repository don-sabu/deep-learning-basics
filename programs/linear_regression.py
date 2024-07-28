import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv('height-weight.csv')
x = data['Height']
y = data['Weight']

xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size=0.8, random_state=42)

xtrain = np.array(xtrain).reshape(-1, 1)
xtest = np.array(xtest).reshape(-1, 1)
ytrain = np.array(ytrain).reshape(-1, 1)
ytest = np.array(ytest).reshape(-1, 1)

model = LinearRegression()
model.fit(xtrain, ytrain)

prediction = model.predict(xtest)
mse = round(mean_squared_error(ytest, prediction),4)

plt.scatter(xtrain, ytrain, label="Data Points")
plt.plot(xtest, prediction, c='#FF5733', label="Regression Line")
plt.title("Linear Regression")
plt.xlabel("Height")
plt.ylabel("Weight")
plt.annotate(f'Mean Squared Error: {mse}', xy=(0.02, 0.8), xycoords='axes fraction', fontsize=10)
plt.legend()
plt.savefig("Linear_regression_output.png")
plt.show()

