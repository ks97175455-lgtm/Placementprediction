import os,pandas as pd,matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df=pd.read_csv(os.path.join(ROOT,"datasets","placement_predict_50k Dataset.csv"))
X=df[["CGPA","AptitudeTestScore"]].apply(pd.to_numeric,errors="coerce").dropna()
Z=StandardScaler().fit_transform(X)
wcss=[]
for k in range(2,9):
    m=KMeans(n_clusters=k,n_init=10,random_state=42).fit(Z)
    wcss.append(m.inertia_)
print("WCSS:",wcss)
plt.plot(range(2,9),wcss,marker="o"); plt.xlabel("K"); plt.ylabel("WCSS"); plt.title("Elbow Method"); plt.show()
