# Task-3
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn import tree
df = pd.read_csv(
r"C:\intern\bank+-marketing\bank\bank.csv",
sep=";"
)
print(df.head())
df.info()
print(df.isnull().sum())
for column in df.select_dtypes(include="object").columns:
le = LabelEncoder()
df[column] = le.fit_transform(df[column])
X = df.drop("y", axis=1)
y = df["y"]
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)
model = DecisionTreeClassifier(
criterion="entropy", max_depth=5, random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
plt.figure(figsize=(20,10))
tree.plot_tree(model, feature_names=X.columns,
class_names=["No","Yes"], filled=True)
plt.title("Decision Tree Classifier - Bank Marketing")
plt.show()
