import numpy as np

class Lin_reg_sgd:
    @staticmethod
    def mse(y_true, y_pred):
        loss = 0
        for i in range(len(y_true)):
            y = y_true[i]
            pred = y_pred[i]
            loss += (y-pred)**2
        loss /= len(y_true)
        return loss
    
    def __init__(self, lr = 0.1, iters = 2000, batch_size=128):
        self.lr = lr
        self.iters = iters
        self.w = None
        self.batch_size = batch_size
        
    def fit(self, X, y):
        samples, features = X.shape
        ones = np.ones((X.shape[0], 1))
        X_b = np.hstack([ones, X])
        self.w = np.zeros(features+1)
        rng = np.random.default_rng(seed=42)
        for _ in range(self.iters):
            indicies = rng.choice(np.arange(X_b.shape[0]), size=self.batch_size, replace=False)
            X_batch = X_b[indicies]
            y_batch = y[indicies]
            preds = np.dot(X_batch, self.w)
            
            dw = (1 / self.batch_size) * np.dot(X_batch.T, (preds - y_batch))
            
            self.w = self.w - self.lr * dw
    def predict(self, X_test):
        ones = np.ones((X_test.shape[0], 1))
        X_test_b = np.hstack([ones, X_test])
        preds = np.dot(X_test_b, self.w)
        return preds