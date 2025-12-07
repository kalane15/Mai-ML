import numpy as np


class MyCrossValidation:
    @staticmethod
    def k_fold_cross_validation(model, x, y, k=5):
        samples = len(x)

        indices = np.arange(samples)
        np.random.shuffle(indices)

        fold_size = samples // k
        mse_scores = []

        for i in range(k):
            test_indices = indices[i * fold_size: (i + 1) * fold_size]
            train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])

            x_train, x_test = x[train_indices], x[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]

            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)

            mse = np.mean((y_test - y_pred) ** 2)
            mse_scores.append(mse)

        return np.array(mse_scores)

    @staticmethod
    def leave_one_out_cross_validation(model, x, y):
        return MyCrossValidation.k_fold_cross_validation(model, x, y, len(x))
