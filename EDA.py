import os,pandas as pd,matplotlib.pyplot as plt
ROOT=os.path.dirname(os.path.abspath(__file__))
df=pd.read_csv(os.path.join(ROOT,"datasets","placement_predict_50k Dataset.csv"))
print(df.describe(include="all").T)
fig,ax=plt.subplots(); df["PlacementStatus"].value_counts().sort_index().plot(kind="bar",ax=ax); ax.set_title("Placement Status"); plt.tight_layout(); plt.show()
fig,ax=plt.subplots(); ax.hist(df["CGPA"].dropna(),bins=20); ax.set_title("CGPA Distribution"); plt.tight_layout(); plt.show()
fig,ax=plt.subplots(); ax.scatter(df["CGPA"],df["Salary Package"],s=10,alpha=.25); ax.set_xlabel("CGPA"); ax.set_ylabel("Salary Package"); plt.tight_layout(); plt.show()
