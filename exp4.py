import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score

digits = load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

k_values = [1, 3, 5, 7, 9]
precision_list = []
recall_list = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    p = precision_score(y_test, y_pred, average='macro')
    r = recall_score(y_test, y_pred, average='macro')
    
    precision_list.append(p)
    recall_list.append(r)
    
    print("K =", k, " Precision:", p, " Recall:", r)

best_k = k_values[np.argmax(precision_list)]
print("\nBest K:", best_k)

plt.plot(k_values, precision_list, marker='o')
plt.plot(k_values, recall_list, marker='s')
plt.xlabel("K value")
plt.ylabel("Score")
plt.title("KNN Performance")
plt.show()