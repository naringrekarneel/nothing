import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler


iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target


sns.scatterplot(x=df['sepal length (cm)'], 
                y=df['sepal width (cm)'], 
                hue=df['target'])
plt.title("Sepal Length vs Width")
plt.show()


scaler = MinMaxScaler()
df_scaled = df.copy()
df_scaled.iloc[:, :-1] = scaler.fit_transform(df_scaled.iloc[:, :-1])


df_scaled.iloc[:, :-1].hist(figsize=(8,6))
plt.suptitle("Histogram of Features")
plt.show()


sns.boxplot(data=df_scaled.iloc[:, :-1])
plt.title("Boxplot")
plt.show()