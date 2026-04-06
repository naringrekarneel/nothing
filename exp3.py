import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

gini_model = DecisionTreeClassifier(criterion='gini', random_state=42)
gini_model.fit(X_train, y_train)

gini_pred = gini_model.predict(X_test)
gini_acc = accuracy_score(y_test, gini_pred)

cm_gini = confusion_matrix(y_test, gini_pred)

print("=== Gini Decision Tree ===")
print("Accuracy:", gini_acc)
print("Confusion Matrix:\n", cm_gini)

_, counts = np.unique(y, return_counts=True)
prob = counts / len(y)
gini_impurity = 1 - np.sum(prob**2)

print("Gini Impurity (Root):", gini_impurity)

sns.heatmap(cm_gini, annot=True, fmt='d', cmap='Blues')
plt.title("Gini Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

entropy_model = DecisionTreeClassifier(criterion='entropy', random_state=42)
entropy_model.fit(X_train, y_train)

entropy_pred = entropy_model.predict(X_test)
entropy_acc = accuracy_score(y_test, entropy_pred)

entropy_value = -np.sum(prob * np.log2(prob))

print("\n=== Entropy (ID3) Decision Tree ===")
print("Accuracy:", entropy_acc)
print("Entropy (Root):", entropy_value)

cm_entropy = confusion_matrix(y_test, entropy_pred)

sns.heatmap(cm_entropy, annot=True, fmt='d', cmap='Greens')
plt.title("Entropy Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()