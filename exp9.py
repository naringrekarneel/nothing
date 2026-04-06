import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

ab = AdaBoostClassifier(n_estimators=100, random_state=42)
ab.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
ab_pred = ab.predict(X_test)

rf_acc = accuracy_score(y_test, rf_pred)
ab_acc = accuracy_score(y_test, ab_pred)

print("Random Forest Accuracy:", rf_acc)
print("AdaBoost Accuracy:", ab_acc)

print("\nRandom Forest Report:\n", classification_report(y_test, rf_pred))
print("\nAdaBoost Report:\n", classification_report(y_test, ab_pred))

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt='d')
plt.title("Random Forest")

plt.subplot(1,2,2)
sns.heatmap(confusion_matrix(y_test, ab_pred), annot=True, fmt='d')
plt.title("AdaBoost")

plt.tight_layout()
plt.show()

plt.bar(['Random Forest', 'AdaBoost'], [rf_acc, ab_acc])
plt.title("Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()