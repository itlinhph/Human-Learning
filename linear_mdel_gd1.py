from sklearn.linear_model import LinearRegression
import numpy as np

X = np.random.rand(1000)
y = 4 + 3*X + 0.5*np.random.randn(1000)

# print (X)
# print (y)

model = LinearRegression()
model.fit(X.reshape(-1, 1), y.reshape(-1, 1) )

w, b = model.coef_[0][0], model.intercept_[0]

sol_sklearn = np.array([b, w])
print(sol_sklearn)