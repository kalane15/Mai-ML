from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier

from customRealization.CustomBagging import CustomBagging

# Загружаем данные
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Базовый алгоритм
base = DecisionTreeClassifier()
my_bag = CustomBagging(base_model=base, n_estimators=10)
my_bag.fit(X_train, y_train)
y_pred_my = my_bag.predict(X_test)
acc_my = accuracy_score(y_test, y_pred_my)
print("Accuracy MyBagging:", acc_my)
sk_bag = BaggingClassifier(
    estimator=base,
    n_estimators=10,
    bootstrap=True,
    random_state=42
)
sk_bag.fit(X_train, y_train)
y_pred_sk = sk_bag.predict(X_test)
acc_sk = accuracy_score(y_test, y_pred_sk)
print("Accuracy sklearn Bagging:", acc_sk)
