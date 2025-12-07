import numpy as np

from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler


class MyLinearRegressionWithRegularization:
    def __init__(self, alpha=0.01, regularization_type='L2', p=2, l1_ratio=0.5, lambda_reg=0.1, iterations=1000):
        self.alpha = alpha
        self.regularization_type = regularization_type
        self.p = p
        self.l1_ratio = l1_ratio
        self.lambda_reg = lambda_reg
        self.iterations = iterations
        self.weights = None
        self.bias = 0
        self.scaler = StandardScaler()

    def fit(self, X, y):
        X_norm = self.scaler.fit_transform(X)
        m, n = X_norm.shape

        if self.regularization_type == 'L2':
            I = np.eye(n)
            XTX = X_norm.T @ X_norm
            XTy = X_norm.T @ y
            self.weights = np.linalg.inv(XTX + self.lambda_reg * I) @ XTy
            self.bias = np.mean(y) - np.mean(X_norm @ self.weights)
            return

        self.weights = np.zeros(n)
        self.bias = 0

        for _ in range(self.iterations):
            preds = X_norm @ self.weights + self.bias
            errors = preds - y

            dw = (2 / m) * (X_norm.T @ errors)
            db = (2 / m) * np.sum(errors)

            if self.regularization_type == 'L1':
                dw += self.lambda_reg * np.sign(self.weights)
            elif self.regularization_type == 'Lp':
                dw += self.lambda_reg * self.p * np.sign(self.weights) * np.abs(self.weights) ** (self.p - 1)
            elif self.regularization_type == 'L1+L2':
                dw += self.lambda_reg * (self.l1_ratio * np.sign(self.weights) + (1 - self.l1_ratio) * 2 * self.weights)

            self.weights -= self.alpha * dw
            self.bias -= self.alpha * db

    def predict(self, X):
        X_norm = self.scaler.transform(X)
        return X_norm @ self.weights + self.bias


np.random.seed(42)
X = np.random.rand(100, 3)
y = X.dot([2, -3, 1]) + 5 + np.random.randn(100) * 0.1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lambda_reg = 1.0
iterations = 100
alpha = 0.1

l1_model = MyLinearRegressionWithRegularization(regularization_type='L1', lambda_reg=lambda_reg, alpha=alpha,
                                                iterations=iterations)
l2_model = MyLinearRegressionWithRegularization(regularization_type='L2', lambda_reg=lambda_reg)
lp_model = MyLinearRegressionWithRegularization(regularization_type='Lp', lambda_reg=lambda_reg, p=3, alpha=alpha,
                                                iterations=iterations)
l1_l2_model = MyLinearRegressionWithRegularization(regularization_type='L1+L2', lambda_reg=lambda_reg, alpha=alpha,
                                                   l1_ratio=0.5, iterations=iterations)

l1_model.fit(X_train, y_train)
l2_model.fit(X_train, y_train)
lp_model.fit(X_train, y_train)
l1_l2_model.fit(X_train, y_train)

l1_pred = l1_model.predict(X_test)
l2_pred = l2_model.predict(X_test)
lp_pred = lp_model.predict(X_test)
l1_l2_pred = l1_l2_model.predict(X_test)

lasso_model = Lasso(alpha=lambda_reg)
ridge_model = Ridge(alpha=lambda_reg)
elasticnet_model = ElasticNet(alpha=lambda_reg, l1_ratio=0.5)

lasso_model.fit(X_train, y_train)
ridge_model.fit(X_train, y_train)
elasticnet_model.fit(X_train, y_train)

lasso_pred = lasso_model.predict(X_test)
ridge_pred = ridge_model.predict(X_test)
elasticnet_pred = elasticnet_model.predict(X_test)

print("MSE (L1 - My Implementation):", mean_squared_error(y_test, l1_pred))
print("MSE (L2 - My Implementation):", mean_squared_error(y_test, l2_pred))
print("MSE (Lp - My Implementation):", mean_squared_error(y_test, lp_pred))
print("MSE (L1+L2 - My Implementation):", mean_squared_error(y_test, l1_l2_pred))

print("\nMSE (L1 - Sklearn Lasso):", mean_squared_error(y_test, lasso_pred))
print("MSE (L2 - Sklearn Ridge):", mean_squared_error(y_test, ridge_pred))
print("MSE (L1+L2 - Sklearn ElasticNet):", mean_squared_error(y_test, elasticnet_pred))
