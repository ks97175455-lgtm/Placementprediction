import os,pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df=pd.read_csv(os.path.join(ROOT,"datasets","placement_predict_50k Dataset.csv"))
X=df[["CGPA","AptitudeTestScore"]].apply(pd.to_numeric,errors="coerce").dropna()
Z=StandardScaler().fit_transform(X)
m=KMeans(n_clusters=3,n_init=10,random_state=42).fit(Z)
df.loc[X.index,"Cluster"]=m.labels_
d=m.transform(Z).min(axis=1)
threshold=pd.Series(d).quantile(.95)
out=X.index[d>=threshold]
print("Potential cluster-distance outliers:",len(out))
print(df.loc[out,["StudentID","CGPA","AptitudeTestScore","PlacementStatus"]].head(20))
