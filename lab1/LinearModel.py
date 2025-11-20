import numpy as np

class LinearReg:
    @staticmethod
    def mse(y_true, y_pred):
        loss = 0
        for i in range(len(y_true)):
            y = y_true[i]
            pred = y_pred[i]
            loss += (y-pred)**2
        loss /= len(y_true)
        return loss
    
    def __init__(self, lr = 0.01, iters = 2000):
        self.lr = lr
        self.iters = iters
        self.w = None
        
    def fit(self, X_train, y_train):
        samples, features = X_train.shape
        self.w = np.zeros(features)
        self.b = 0
        
        for _ in range(self.iters):
            preds = np.dot(X_train, self.w) + self.b
            
            dw = (1 / samples) * np.dot(X_train.T, (preds - y_train))
            db = (1 / samples) * np.sum(preds-y_train)
            
            self.w = self.w - self.lr * dw
    def predict(self, X_test):
        preds = np.dot(X_test, self.w) + self.b
        return preds