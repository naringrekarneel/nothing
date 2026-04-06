import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

digits = load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

linear = SVC(kernel='linear')
poly = SVC(kernel='poly')
rbf = SVC(kernel='rbf')

linear.fit(X_train, y_train)
poly.fit(X_train, y_train)
rbf.fit(X_train, y_train)

y1 = linear.predict(X_test)
y2 = poly.predict(X_test)
y3 = rbf.predict(X_test)

a1 = accuracy_score(y_test, y1)
a2 = accuracy_score(y_test, y2)
a3 = accuracy_score(y_test, y3)

print("Linear:", a1)
print("Poly:", a2)
print("RBF:", a3)

plt.bar(['Linear','Poly','RBF'], [a1, a2, a3])
plt.title("SVM Accuracy")
plt.show()

cm = confusion_matrix(y_test, y3)
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.show()

for i in range(5):
    plt.imshow(X_test[i].reshape(8,8))
    plt.title("Pred: " + str(y3[i]))
    plt.show()