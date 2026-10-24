import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="FairTox AI", page_icon="🛡️", layout="wide")
BASE = Path(__file__).resolve().parent
LABELS = ["toxic", "obscene", "threat", "insult", "identity_hate", "sexual_explicit"]
NAMES = {"toxic":"Toxicity","obscene":"Obscenity","threat":"Threat","insult":"Insult","identity_hate":"Identity Hate","sexual_explicit":"Sexual Explicit"}
DEFAULT_THRESHOLDS = {"toxic":.720,"obscene":.900,"threat":.863,"insult":.789,"identity_hate":.821,"sexual_explicit":.920}

st.markdown("""
<style>
.stApp{background:#081326;color:#edf3ff}.stApp h1,.stApp h2,.stApp h3{color:#edf3ff}
[data-testid="stSidebar"]{background:#0a1428;border-right:1px solid #21385f}
div[data-testid="stMetric"]{background:#101f3a;border:1px solid #21385f;padding:12px;border-radius:10px}
.stTextArea textarea{background:#101f3a;color:#fff;border:1px solid #36517f}
.stButton>button{background:#4f5fff;color:white;border:0;border-radius:8px;font-weight:600}
.block-container{padding-top:1.3rem}
</style>
""", unsafe_allow_html=True)

def first(*paths):
    return next((Path(p) for p in paths if Path(p).exists()), None)

def read_json(path, default):
    try:
        return json.loads(path.read_text()) if path else default
    except Exception:
        return default

def read_table(*paths):
    for p in paths:
        p=Path(p)
        if p.exists():
            try:
                return (pd.read_parquet(p) if p.suffix==".parquet" else pd.read_csv(p)), p
            except Exception as e:
                st.warning(f"Could not read {p.name}: {e}")
    return None, None

thr_data=read_json(first(BASE/"thresholds_calibrated.json",BASE/"thresholds_calibrated.json",BASE/"thresholds.json"),DEFAULT_THRESHOLDS)
if "thresholds" in thr_data and isinstance(thr_data["thresholds"],dict): thr_data=thr_data["thresholds"]
THRESHOLDS={k:float(thr_data.get(k,v)) for k,v in DEFAULT_THRESHOLDS.items()}
calibration=read_json(first(BASE/"calibration_config.json",BASE/"calibration_config.json"),{})
abstention=read_json(first(BASE/"abstention_config.json",BASE/"abstention_config.json"),{})
fairness, fairness_path=read_table(BASE/"abstention_by_group.csv",BASE/"fairness_by_group.csv",BASE/"abstention_by_group.csv",BASE/"fairness_by_group.csv")
predictions, predictions_path=read_table(BASE/"test_predictions.parquet",BASE/"test_predictions.csv",BASE/"test_predictions.parquet")
metrics, metrics_path=read_table(BASE/"baseline_metrics.csv",BASE/"model_metrics.csv",BASE/"calibration_before_after.csv",BASE/"calibration_before_after.csv")

@st.cache_resource(show_spinner=False)
def load_model():
    model_dir=first(BASE/"final_model",BASE/"model/final_model")
    tok_dir=first(BASE/"tokenizer",BASE/"model/tokenizer")
    prep=BASE/"toxic_preprocessing.py"
    if not model_dir or not tok_dir: return None,None,None,"Missing final_model/ or tokenizer/ folder beside app.py."
    if not prep.exists(): return None,None,None,"Missing toxic_preprocessing.py beside app.py."
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        sys.path.insert(0,str(BASE))
        from toxic_preprocessing import clean_text
        device="cuda" if torch.cuda.is_available() else "cpu"
        tokenizer=AutoTokenizer.from_pretrained(str(tok_dir))
        model=AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device)
        model.eval()
        return model,tokenizer,(clean_text,device),None
    except Exception as e:
        return None,None,None,f"Model loading error: {e}"

def predict(text, model, tokenizer, clean_text, device):
    import torch
    with torch.no_grad():
        batch=[clean_text(text)]
        tokens=tokenizer(batch,truncation=True,max_length=240,padding=True,return_tensors="pt")
        tokens={k:v.to(device) for k,v in tokens.items()}
        return torch.sigmoid(model(**tokens).logits.float()).cpu().numpy()[0]

def calibrate(raw):
    temps=calibration.get("temperatures",{}) if isinstance(calibration,dict) else {}
    out=[]
    for i,l in enumerate(LABELS):
        p=np.clip(raw[i],1e-6,1-1e-6)
        t=max(float(temps.get(l,1.0)),1e-4)
        z=np.log(p/(1-p))/t
        out.append(1/(1+np.exp(-np.clip(z,-35,35))))
    return np.array(out)

def decision(p):
    thresholds=np.array([THRESHOLDS[l] for l in LABELS])
    delta=float(abstention.get("delta",0.0))
    p=np.clip(p,1e-6,1-1e-6); th=np.clip(thresholds,1e-6,1-1e-6)
    margin=np.abs(np.log(p/(1-p))-np.log(th/(1-th)))
    review=[LABELS[i] for i,m in enumerate(margin) if m<delta]
    flags=[LABELS[i] for i,v in enumerate(p) if v>=thresholds[i]]
    status="HUMAN_REVIEW" if review else ("AUTO_FLAG" if flags else "AUTO_ALLOW")
    return status,flags,review

def find_text_col(df):
    return next((c for c in ["text_raw","text","comment_text","comment"] if c in df.columns),None)

with st.sidebar:
    st.markdown("# 🛡️ FairTox AI")
    st.caption("Bias-aware, explainable and uncertainty-aware toxic comment classification")
    page=st.radio("Navigation",["Analyze Comment","Model Performance","Fairness Audit","Explanation & Insights","About"])
    st.divider()
    st.caption("Built with Streamlit")

if page=="Analyze Comment":
    st.title("Analyze a Comment")
    st.caption("Enter a comment to see predicted labels, explanations and uncertainty.")
    comment=st.text_area("Enter a comment", "You are such a lazy person. No wonder you can’t get anything done.",height=95)
    if st.button("🔎 Analyze"):
        if not comment.strip(): st.warning("Enter a comment first.")
        else:
            model,tokenizer,prep,error=load_model()
            if error: st.error(error)
            else:
                try:
                    raw=predict(comment,model,tokenizer,*prep)
                    p=calibrate(raw); status,flags,review=decision(p)
                    st.session_state["last"]={"text":comment,"p":p.tolist(),"status":status,"flags":flags,"review":review}
                except Exception as e: st.error(f"Prediction failed: {e}")
    result=st.session_state.get("last")
    left,right=st.columns([1.55,1])
    with left:
        st.subheader("Prediction Results")
        if result:
            status=result["status"]; p=np.array(result["p"])
            if status=="HUMAN_REVIEW": st.warning("Human review recommended")
            elif status=="AUTO_FLAG": st.error("Potentially toxic comment detected")
            else: st.success("No label crossed its threshold")
            a,b=st.columns(2); a.metric("Highest label probability",f"{p.max():.0%}"); b.metric("Flagged labels",len(result["flags"]))
            rows=pd.DataFrame([{"Label":NAMES[l],"Probability":p[i],"Threshold":THRESHOLDS[l],"Decision":"Flag" if l in result["flags"] else "Not flagged"} for i,l in enumerate(LABELS)])
            fig=px.bar(rows,x="Probability",y="Label",orientation="h",color="Decision",color_discrete_map={"Flag":"#ff4567","Not flagged":"#526a99"},range_x=[0,1])
            fig.update_layout(template="plotly_dark",paper_bgcolor="#081326",plot_bgcolor="#081326",height=320)
            st.plotly_chart(fig,use_container_width=True)
            st.dataframe(rows.style.format({"Probability":"{:.3f}","Threshold":"{:.3f}"}),use_container_width=True,hide_index=True)
            if result["flags"]: st.write("**Flagged:** "+", ".join(NAMES[x] for x in result["flags"]))
            if result["review"]: st.write("**Needs review:** "+", ".join(NAMES[x] for x in result["review"]))
            st.info("Model scores are not proof of intent. Human review is important for high-impact moderation decisions.")
        else: st.info("Run Analyze to see model predictions.")
        st.subheader("Explanation (SHAP)")
        shap_files=sorted((BASE/"member4_outputs").glob("shap_tokens_*.csv")) if (BASE/"member4_outputs").exists() else []
        if shap_files:
            chosen=st.selectbox("Saved SHAP label table",shap_files,format_func=lambda x:x.stem)
            sdf=pd.read_csv(chosen); st.dataframe(sdf.head(15),use_container_width=True,hide_index=True)
        else: st.caption("Saved SHAP token tables from Member 4 will appear here when copied into member4_outputs/.")
    with right:
        st.subheader("Fairness & Bias Audit")
        if fairness is not None:
            st.caption(f"Loaded {fairness_path.name}")
            st.dataframe(fairness.head(8),use_container_width=True,hide_index=True)
            gc=next((c for c in fairness.columns if c.lower() in ["group","identity_group","identity"]),None)
            fc=next((c for c in fairness.columns if "fpr" in c.lower()),None)
            if gc and fc:
                fig=px.bar(fairness,x=gc,y=fc,title="False-positive rate by group")
                fig.update_layout(template="plotly_dark",paper_bgcolor="#081326",plot_bgcolor="#081326")
                st.plotly_chart(fig,use_container_width=True)
        else: st.info("Add Member 3's audit CSV to member3_outputs/ to show real fairness results.")
        st.subheader("Uncertain Predictions")
        st.caption("The app uses Member 4's abstention_config.json when available.")
        st.metric("Review rule", "Threshold margin")
        st.subheader("Model Comparison")
        if metrics is not None: st.dataframe(metrics.head(8),use_container_width=True,hide_index=True)
        else: st.caption("Add baseline/transformer metric exports to show measured comparison.")

elif page=="Model Performance":
    st.title("Model Performance")
    if predictions is None:
        st.warning("No test_predictions.parquet or test_predictions.csv found beside app.py.")
    else:
        st.caption(f"Loaded {predictions_path.name}")
        rows=[]
        for l in LABELS:
            yc=next((c for c in [f"true_{l}",f"{l}_true"] if c in predictions.columns),None)
            pc=next((c for c in [f"prob_{l}",f"{l}_prob"] if c in predictions.columns),None)
            if not yc or not pc: continue
            y=pd.to_numeric(predictions[yc],errors="coerce"); p=pd.to_numeric(predictions[pc],errors="coerce")
            ok=y.notna()&p.notna()
            if not ok.any(): continue
            y=y[ok].astype(int).to_numpy(); pred=p[ok].to_numpy()>=THRESHOLDS[l]
            tp=((pred==1)&(y==1)).sum(); fp=((pred==1)&(y==0)).sum(); fn=((pred==0)&(y==1)).sum(); tn=((pred==0)&(y==0)).sum()
            precision=tp/(tp+fp) if tp+fp else 0; recall=tp/(tp+fn) if tp+fn else 0
            rows.append({"Label":NAMES[l],"Precision":precision,"Recall":recall,"F1":2*precision*recall/(precision+recall) if precision+recall else 0,"FPR":fp/(fp+tn) if fp+tn else np.nan,"FNR":fn/(fn+tp) if fn+tp else np.nan})
        if rows:
            df=pd.DataFrame(rows); st.dataframe(df.style.format("{:.3f}",subset=["Precision","Recall","F1","FPR","FNR"]),use_container_width=True)
            fig=px.bar(df.melt(id_vars="Label",value_vars=["Precision","Recall","F1"]),x="Label",y="value",color="variable",barmode="group",title="Per-label performance")
            fig.update_layout(template="plotly_dark",paper_bgcolor="#081326",plot_bgcolor="#081326",yaxis_range=[0,1]); st.plotly_chart(fig,use_container_width=True)
        else: st.warning("Prediction file found, but expected prob_<label> and true_<label> columns are missing.")
    if metrics is not None:
        st.subheader("Saved evaluation / calibration results"); st.dataframe(metrics,use_container_width=True,hide_index=True)
    st.info("Use validation data for tuning. Use the held-out test set for final reporting.")

elif page=="Fairness Audit":
    st.title("Fairness Audit")
    st.caption("Group-wise results generated by Member 3.")
    if fairness is None:
        st.warning("No fairness CSV found. Copy Member 3's audit output into member3_outputs/.")
        st.code("abstention_by_group.csv")
    else:
        st.success(f"Loaded {fairness_path}")
        st.dataframe(fairness,use_container_width=True,hide_index=True)
        gc=next((c for c in fairness.columns if c.lower() in ["group","identity_group","identity"]),None)
        for metric in [c for c in fairness.columns if "fpr" in c.lower() or "fnr" in c.lower()]:
            if gc:
                fig=px.bar(fairness,x=gc,y=metric,title=f"{metric} by identity group")
                fig.update_layout(template="plotly_dark",paper_bgcolor="#081326",plot_bgcolor="#081326")
                st.plotly_chart(fig,use_container_width=True)
        st.warning("A disparity is a signal to investigate, not by itself proof of unfairness. Check group sizes and annotation coverage.")

elif page=="Explanation & Insights":
    st.title("Explanation & Insights")
    files=sorted((BASE/"member4_outputs").glob("shap_tokens_*.csv")) if (BASE/"member4_outputs").exists() else []
    if files:
        f=st.selectbox("Choose SHAP table",files,format_func=lambda x:x.stem)
        df=pd.read_csv(f); st.dataframe(df,use_container_width=True,hide_index=True)
        token=next((c for c in df.columns if "token" in c.lower()),None)
        value=next((c for c in df.columns if "mean_abs" in c.lower() or "importance" in c.lower() or "shap" in c.lower()),None)
        if token and value:
            fig=px.bar(df.head(20),x=value,y=token,orientation="h",title="Token contributions")
            fig.update_layout(template="plotly_dark",paper_bgcolor="#081326",plot_bgcolor="#081326")
            st.plotly_chart(fig,use_container_width=True)
    else: st.info("Member 4's SHAP CSV outputs will appear here after you copy them into member4_outputs/.")
    st.subheader("Uncertain cases")
    if predictions is not None:
        tc=find_text_col(predictions)
        if tc: st.dataframe(predictions[[tc]].head(50),use_container_width=True,hide_index=True)
        else: st.caption("No text column was detected in the prediction file.")
    else: st.caption("Add a prediction dataset or a dedicated uncertain-cases CSV to browse examples.")

else:
    st.title("About FairTox AI")
    st.markdown("""
**FairTox AI** integrates the five project contributions:

- **Member 1:** Text preprocessing, TF-IDF and Logistic Regression baseline.
- **Member 2:** Fine-tuned DistilBERT multi-label classifier.
- **Member 3:** Fairness auditing, group-wise error analysis and counterfactual tests.
- **Member 4:** SHAP explanations, probability calibration and uncertainty-aware review policy.
- **Member 5:** Streamlit integration, dashboards, testing and deployment.

The six labels are toxicity, obscenity, threat, insult, identity hate and sexually explicit content.
""")
    st.subheader("Expected folder layout")
    st.code("""fairtox_app/
├── app.py
├── requirements.txt
├── final_model/ (config.json, model.safetensors)
├── tokenizer/ (tokenizer.json, tokenizer_config.json)
├── toxic_preprocessing.py
├── thresholds.json or thresholds_calibrated.json
├── calibration_config.json
├── abstention_config.json
├── test_predictions.parquet
├── member3_outputs/ (fairness audit CSVs)
└── member4_outputs/ (SHAP and calibration CSVs)
""")
    st.caption("The dashboard uses real files when present; it does not invent fairness or performance numbers.")
