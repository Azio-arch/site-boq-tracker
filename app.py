import streamlit as st, pandas as pd, numpy as np, joblib
st.set_page_config(page_title="AI Small-Scale Construction Coordination",layout="wide")
st.title("AI-Based Small-Scale Construction Coordination System")
st.caption("Monitor agreed time, progress and cost — and use AI to flag coordination risks.")

tasks=pd.read_csv("data/project_tasks.csv")
tasks["cost_variance_pct"]=((tasks.actual_cost_inr-tasks.agreed_cost_inr)/tasks.agreed_cost_inr*100).round(1)

c1,c2,c3,c4=st.columns(4)
c1.metric("Activities",len(tasks))
c2.metric("Delayed",int((tasks.days_delay>0).sum()))
c3.metric("Avg. Progress Gap",f"{tasks.progress_gap_pct.mean():.1f}%")
c4.metric("Avg. Cost Variance",f"{tasks.cost_variance_pct.mean():.1f}%")
st.subheader("Project Control Dashboard")
st.dataframe(tasks,use_container_width=True)

st.subheader("AI Risk Monitor")
try:
    model=joblib.load("construction_risk_model.joblib")
    tasks["open_actions"]=np.maximum((tasks.progress_gap_pct/10).round().astype(int),0)
    tasks["update_age_days"]=np.where(tasks.status=="Delayed",3,1)
    X=tasks[["progress_gap_pct","days_delay","cost_variance_pct","open_actions","update_age_days"]].copy()
    X["progress_gap_pct"]=X.progress_gap_pct.clip(lower=0)
    X["cost_variance_pct"]=X.cost_variance_pct.clip(lower=0)
    tasks["AI risk %"]=(model.predict_proba(X)[:,1]*100).round(1)
    tasks["AI risk"]=tasks["AI risk %"].apply(lambda x:"HIGH" if x>=70 else ("MEDIUM" if x>=40 else "LOW"))
    st.dataframe(tasks[["package","activity","vendor","progress_gap_pct","days_delay","cost_variance_pct","AI risk %","AI risk"]],use_container_width=True)
except FileNotFoundError:
    st.info("Run `python train_model.py` first.")

st.subheader("AI-Assisted Vendor/Site Update")
message=st.text_area("Paste an update from an already-selected vendor",
"Foundation work is 75% complete. We are 2 days behind because reinforcement delivery was late. Expected completion is Friday. No additional cost expected.")
if st.button("Analyse Update"):
    text=message.lower(); signals=[]
    if any(x in text for x in ["behind","delay","late"]): signals.append("Potential schedule delay")
    if "%" in text: signals.append("Progress update")
    if "cost" in text: signals.append("Cost-impact statement")
    if any(x in text for x in ["expected completion","complete","by "]): signals.append("Completion commitment/date")
    st.write("**Extracted signals:**",", ".join(signals) if signals else "No major signals detected.")
    if any(x in text for x in ["delay","behind","late"]):
        st.warning("Recommended action: create a follow-up, confirm the revised completion date, and increase monitoring frequency.")

st.subheader("Workflow")
st.write("Agreed baseline → Vendor/site update → AI extracts signals → Compare planned vs actual → Flag risk → Follow-up → Dashboard update")
