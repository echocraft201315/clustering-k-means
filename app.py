import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans

print(plt.style.available)
plt.style.use("seaborn-v0_8-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)

df = pd.read_csv("./housing.csv")
X = df.loc[:, ["MedInc", "Latitude", "Longitude"]]

#create cluster feature
kmeans=KMeans(n_clusters=6)
X["Cluster"] = kmeans.fit_predict(X)
X["Cluster"] =X["Cluster"].astype("category")

sns.relplot(
    x="Longitude", y="Latitude", hue="Cluster", data=X, height=6,
)
print(X.head())

X["MedHouseVal"] = df["MedHouseVal"]
sns.catplot(x="MedHouseVal", y="Cluster", data=X, kind="boxen", height=6,hue="Cluster")
plt.show()