"""Linear Regression: predict Salary Package from placement dataset."""
import os, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA=os.path.join(ROOT,"datasets","placement_predict_50k Dataset.csv")

df=pd.read_csv(DATA)
y=df["Salary Package"]
X=df.drop(columns=["StudentID","Salary Package","PlacementStatus","IsAnomaly"])
num=X.select_dtypes("number").columns
cat=X.select_dtypes(exclude="number").columns
prep=ColumnTransformer([
 ("num",SimpleImputer(strategy="median"),num),
 ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore"))]),cat)
])
model=Pipeline([("prep",prep),("reg",LinearRegression())])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
model.fit(Xtr,ytr); pred=model.predict(Xte)
print("RMSE:",mean_squared_error(yte,pred)**.5)
print("R2:",r2_score(yte,pred))
