#!/usr/bin/env python3
"""
Health Symptom Checker (Educational)
"""

import os
import re
from typing import List
import streamlit as st
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import streamlit.components.v1 as components

st.set_page_config(page_title="Health Symptom Checker (Educational)", layout="wide")

BASE_DIR = os.path.dirname(__file__)
DISEASE_SYMPTTOMS_CSV = os.path.join(BASE_DIR, "DiseaseAndSymptoms.csv")
PRECAUTION_CSV = os.path.join(BASE_DIR, "Disease precaution.csv")

DISCLAIMER = """
**MEDICAL DISCLAIMER**  
This tool is for **EDUCATIONAL PURPOSES ONLY**.  
It does **NOT** provide medical advice or diagnosis.  
Always consult qualified healthcare professionals.
"""

COMMON_MISSPELLINGS = {"vomitting":"vomiting","diahrrhea":"diarrhea","diarrhoea":"diarrhea"}

def clean_text(s: str) -> str:
    if not isinstance(s,str): s=str(s)
    s=s.strip().lower()
    for bad,good in COMMON_MISSPELLINGS.items():
        s = re.sub(r"\b"+re.escape(bad)+r"\b", good, s)
    return s.replace(";",",")

def split_symptom_list(s: str) -> List[str]:
    parts=re.split(r"[,/]| and ",s)
    return list(dict.fromkeys([clean_text(p) for p in parts if p.strip()]))

def load_ds(path:str)->pd.DataFrame:
    df=pd.read_csv(path)
    df=df.rename(columns={df.columns[0]:"disease",df.columns[1]:"symptoms"})[["disease","symptoms"]]
    df["disease"]=df["disease"].astype(str).map(clean_text)
    df["symptoms"]=df["symptoms"].astype(str).map(clean_text)
    df["symptom_list"]=df["symptoms"].apply(split_symptom_list)
    df["embed_text"]=df["disease"]+" : "+df["symptoms"]
    return df

def load_prec(path:str)->pd.DataFrame:
    df=pd.read_csv(path)
    df=df.rename(columns={df.columns[0]:"disease",df.columns[1]:"precautions"})[["disease","precautions"]]
    df["disease"]=df["disease"].astype(str).map(clean_text)
    df["precautions"]=df["precautions"].astype(str).map(clean_text)
    return df

def cosine(a,b):
    a_norm=a/(a.norm(dim=1,keepdim=True)+1e-8)
    b_norm=b/(b.norm(dim=1,keepdim=True)+1e-8)
    return torch.mm(a_norm,b_norm.t())

def score(emb_u,emb):
    sims=cosine(emb_u,emb).squeeze(0).cpu().numpy()
    sims=(sims-sims.min())/(sims.max()-sims.min()+1e-8)
    return [s*100 for s in sims]

def matches(text,model,emb,df,top=8):
    with torch.no_grad(): emb_u=model.encode([text],convert_to_tensor=True)
    sc=score(emb_u,emb)
    idxs=sorted(range(len(sc)),key=lambda i:sc[i],reverse=True)
    result=[];seen=set()
    for i in idxs:
        d=df.iloc[i]["disease"]
        if d not in seen:
            seen.add(d)
            result.append({"disease":d,"score":sc[i],"symptoms":df.iloc[i]["symptoms"]})
        if len(result)>=top: break
    return result

ds_df=load_ds(DISEASE_SYMPTTOMS_CSV)
prec_df=load_prec(PRECAUTION_CSV)
prec_map={r["disease"]:r["precautions"] for _,r in prec_df.iterrows()}
def get_prec(d): return prec_map.get(d,"No precautions available.")

@st.cache_resource(show_spinner=True)
def load_model():
    model=SentenceTransformer("all-MiniLM-L6-v2")
    with torch.no_grad():
        emb=model.encode(ds_df["embed_text"].tolist(),convert_to_tensor=True,show_progress_bar=True,batch_size=64)
    return model,emb

model,embeddings=load_model()

#css.part
st.markdown("""
<style>
.stApp {background: radial-gradient(circle at 20% 0%,rgba(56,189,248,0.25),rgba(15,23,42,0.97));background-attachment:fixed;}
.block-container {background:rgba(15,23,42,.85);padding:2rem;border-radius:18px;
border:1.5px solid rgba(56,189,248,.5);box-shadow:0 0 24px rgba(56,189,248,.3);}

.tip-box {
background:rgba(10,20,35,.85);padding:15px 18px;border-radius:14px;
border:1.5px solid rgba(56,189,248,.45);box-shadow:0 0 18px rgba(56,189,248,.45);font-size:17px;color:#e0faff;
}

.tile-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:1.3rem;margin-top:1.2rem;}

.result-tile{background:rgba(15,23,42,.96);padding:1rem 1.2rem;border-radius:1.1rem;
border:1.5px solid rgba(56,189,248,.4);box-shadow:0 0 18px rgba(15,23,42,.8);
opacity:0;transform:translateY(40px);transition:opacity .6s ease, transform .45s ease, box-shadow .45s ease;}
.result-tile.visible{opacity:1;transform:translateY(0);}
.result-tile:hover{box-shadow:0 0 32px rgba(56,189,248,1);transform:translateY(-6px) scale(1.03);}

.selected-bar{margin-top:10px;padding:12px;border-radius:14px;background:rgba(10,20,35,.85);
border:1.5px solid rgba(56,189,248,.45);box-shadow:0 0 18px rgba(56,189,248,.45);}
.selected-pill{padding:6px 12px;margin:4px;border-radius:999px;background:rgba(56,189,248,.25);
border:1.4px solid rgba(56,189,248,.9);color:#e0faff;font-size:14px;box-shadow:0 0 14px rgba(56,189,248,.85);}

.stButton>button{background:rgba(15,23,42,.92);border:2px solid rgba(56,189,248,.9);color:#e0f2fe;
font-size:20px;padding:12px 22px;font-weight:600;border-radius:12px;box-shadow:0 0 24px rgba(56,189,248,.6);transition:.25s;}
.stButton>button:hover{background:rgba(56,189,248,.24);transform:translateY(-4px) scale(1.05);
box-shadow:0 0 38px rgba(56,189,248,.95);}
</style>
""", unsafe_allow_html=True)

#scroll.ani.replay
components.html("""
<script>
document.addEventListener("DOMContentLoaded",function(){
const doc=window.parent.document;
const obs=new IntersectionObserver((entries)=>{
entries.forEach(e=>{ e.isIntersecting ? e.target.classList.add("visible") : e.target.classList.remove("visible");});
},{threshold:.25});
function run(){doc.querySelectorAll(".result-tile").forEach(t=>obs.observe(t));}
run();
new MutationObserver(run).observe(doc.body,{childList:true,subtree:true});
});
</script>
""", height=0)

#UI

st.title("Health Symptom Checker (Educational)")
st.markdown(DISCLAIMER)

#Tips
st.markdown("### Helpful Tips Before You Begin")
st.markdown("""
<div class='tip-box'>
• Select all symptoms you are experiencing for best matching.<br>
• If unsure of medical terms, search similar words like fever → high temperature.<br>
• Result confidence depends on similarity and symptom overlap.<br>
• This assists learning — final decisions must come from a real doctor.
</div>
""", unsafe_allow_html=True)

#Sympt.select
st.subheader("Select Your Symptoms")
all_symptoms = sorted(set(s for row in ds_df["symptom_list"] for s in row))
selected = st.multiselect("Choose symptoms", all_symptoms, placeholder="Type fever, cough, pain, fatigue etc.")

st.markdown("### Selected Symptoms")
if selected:
    st.markdown("<div class='selected-bar'>" +
        "".join(f"<span class='selected-pill'>{s}</span>" for s in selected) +
        "</div>", unsafe_allow_html=True)
else:
    st.markdown("<div class='selected-bar'>No symptoms selected</div>", unsafe_allow_html=True)

run = st.button("Start Check")

if run:
    if not selected:
        st.warning("Please select at least one symptom.")
    else:
        txt=", ".join(selected)
        results = matches(clean_text(txt), model, embeddings, ds_df)

        st.markdown("## Possible Conditions")
        st.markdown("<div class='tile-grid'>", unsafe_allow_html=True)

        for i,r in enumerate(results,start=1):
            d=r["disease"];sc=r["score"]
            sev="SEVERE" if any(x in d for x in ["pneumonia","meningitis","tuberculosis"]) else \
                "MODERATE" if any(x in d for x in ["influenza","bronchitis","uti","dengue"]) else "MILD"
            st.markdown(f"""
            <div class='result-tile'>
                <h3>🔹 {i}. {d.title()}</h3>
                <p><b>Match Score:</b> {sc:.1f}%</p>
                <p><b>Severity:</b> {sev}</p>
                <p><b>Symptoms:</b> {r['symptoms']}</p>
                <p><b>Precautions:</b> {get_prec(d)}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        best=results[0]
        st.markdown("### Final Recommendation")

        if best["score"]>=90:
            st.success(f"Very Strong Match: {best['disease'].title()} ({best['score']:.1f}%)")
        elif best["score"]>=70:
            st.info(f"Strong Match: {best['disease'].title()} ({best['score']:.1f}%)")
        elif best["score"]>=55:
            st.warning(f"Moderate Match: {best['disease'].title()} ({best['score']:.1f}%)")
        else:
            st.error("Low confidence — insufficient matching data.")

        st.info("This is NOT a medical diagnosis. Always consult a healthcare professional.")
