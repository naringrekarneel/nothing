import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


iris = load_iris()
X = iris.data
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Dataset loaded successfully!")


log_model = LogisticRegression(max_iter=200)
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)

log_acc = accuracy_score(y_test, log_pred)

print("\n=== Logistic Regression ===")
print("Accuracy:", log_acc)
print(classification_report(y_test, log_pred))


tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)
tree_pred = tree_model.predict(X_test)

tree_acc = accuracy_score(y_test, tree_pred)

print("\n=== Decision Tree ===")
print("Accuracy:", tree_acc)
print(classification_report(y_test, tree_pred))


models = ['Logistic Regression', 'Decision Tree']
accuracies = [log_acc, tree_acc]

plt.bar(models, accuracies)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()


best_model = models[np.argmax(accuracies)]
print("\nBest Performing Model:", best_model)