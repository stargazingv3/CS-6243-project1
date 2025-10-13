import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("data/nat_data.csv")
X = df[["feature_1","feature_2"]].values
y = (df[["class"]].values).astype(int)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
clf = SVC(kernel="rbf", C=1.0, gamma=5.0).fit(Xtr, ytr)
print("Accuracy:", clf.score(Xte, yte))