"""Gradient Boosting placement classifier."""
import os,pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df=pd.read_csv(os.path.join(ROOT,"datasets","placement_predict_50k Dataset.csv"))
y=df["PlacementStatus"].astype(int); X=df.drop(columns=["StudentID","PlacementStatus","Salary Package","IsAnomaly"])
num=X.select_dtypes("number").columns; cat=X.select_dtypes(exclude="number").columns
prep=ColumnTransformer([("num",SimpleImputer(strategy="median"),num),("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore"))]),cat)])
model=Pipeline([("prep",prep),("gb",GradientBoostingClassifier(n_estimators=100,learning_rate=.08,max_depth=3,random_state=42))])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
model.fit(Xtr,ytr); print("Accuracy:",accuracy_score(yte,model.predict(Xte)))
