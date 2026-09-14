import os,pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df=pd.read_csv(os.path.join(ROOT,"datasets","placement_predict_50k Dataset.csv"))
cols=["CGPA","AttendancePercent","AptitudeTestScore","CodingTestScore"]
X=df[cols].apply(pd.to_numeric,errors="coerce").fillna(df[cols].median())
scaled=pd.DataFrame(MinMaxScaler().fit_transform(X),columns=cols)
print("Min-Max Normalized Data:\n",scaled.head())
standard=pd.DataFrame(StandardScaler().fit_transform(X),columns=cols)
print("\nStandardized Data:\n",standard.head())
