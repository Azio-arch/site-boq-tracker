import pandas as pd, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
df=pd.read_csv("data/historical_risk_data.csv")
X=df.drop(columns="high_risk"); y=df["high_risk"]
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
model=RandomForestClassifier(n_estimators=250,max_depth=6,random_state=42,class_weight="balanced")
model.fit(Xtr,ytr)
print("ROC-AUC:",round(roc_auc_score(yte,model.predict_proba(Xte)[:,1]),3))
joblib.dump(model,"construction_risk_model.joblib")
