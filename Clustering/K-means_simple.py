import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA=os.path.join(ROOT,"datasets","placement_predict_50k Dataset.csv")
FEATURES=["CGPA","AptitudeTestScore"]

def load_data():
    df=pd.read_csv(DATA)
    X=df[FEATURES].apply(pd.to_numeric,errors="coerce").dropna()
    return df.loc[X.index].copy(), X

def run_kmeans(k=3):
    data,X=load_data()
    scaler=StandardScaler(); Z=scaler.fit_transform(X)
    model=KMeans(n_clusters=k,n_init=10,random_state=42)
    labels=model.fit_predict(Z)
    centers=scaler.inverse_transform(model.cluster_centers_)
    data["Cluster"]=labels
    print("K =",k)
    print("Centroids:\n",pd.DataFrame(centers,columns=FEATURES))
    print("\nCluster counts:\n",data["Cluster"].value_counts().sort_index())
    return data,centers

if __name__=="__main__":
    data,centers=run_kmeans(3)
