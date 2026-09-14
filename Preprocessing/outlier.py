import os,pandas as pd
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df=pd.read_csv(os.path.join(ROOT,"datasets","placement_predict_50k Dataset.csv"))
for col in ["CGPA","AttendancePercent","AptitudeTestScore","CodingTestScore"]:
    s=pd.to_numeric(df[col],errors="coerce").dropna()
    q1,q3=s.quantile(.25),s.quantile(.75); iqr=q3-q1
    lo,hi=q1-1.5*iqr,q3+1.5*iqr
    count=((s<lo)|(s>hi)).sum()
    print(f"{col}: lower={lo:.3f}, upper={hi:.3f}, outliers={count}")
