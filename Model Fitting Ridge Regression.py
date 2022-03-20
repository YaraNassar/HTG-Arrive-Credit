import pandas as pd
from numpy import arange
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedKFold
from sklearn.kernel_ridge import KernelRidge

x = np.linspace(0, 2, 100).reshape(-1, 1)
y = 3*x**2 + 2*x + np.random.normal(scale=0.2, size=(100,1))

#find the optimal alpha value 

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
model = RidgeCV(alphas=arange(0, 1, 0.01), cv=cv, scoring='neg_mean_absolute_error')
model.fit(x, y)
al = model.alpha_

#display lambda that produced the lowest test MSE
print(model.alpha_)

#Find the best fit using Ridge Regression
krr = KernelRidge(alpha=al, kernel='poly', degree=5, gamma=1, coef0=1)
Kernel = krr.fit(x, y)

#Plot the actual data points and the fit
plt.plot(x.ravel(), y.ravel(), 'o', color='skyblue', label='Data')
plt.plot(x.ravel(), krr.predict(x).ravel(), '-', label='Ridge', lw=3)
plt.grid()
plt.legend()
plt.show()
