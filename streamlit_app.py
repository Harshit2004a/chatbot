"""
Health Symptom Checker (Educational)
- User selects symptoms from dataset (multiselect)
- Selected symptoms shown as glowing pills in a bar
- Animated background + neon button + floating result tiles
"""

import os
import re
from typing import List
import streamlit as st
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import streamlit.components.v1 as components

#config
st.set_page_config(page_title="Health Symptom Checker (Educational)", layout="wide")

BASE_DIR = os.path.dirname(__file__)
DISEASE_SYMPTOMS_CSV = os.path.join(BASE_DIR, "DiseaseAndSymptoms.csv")
PRECAUTION_CSV       = os.path.join(BASE_DIR, "Disease precaution.csv")

DISCLAIMER = """
**MEDICAL DISCLAIMER**

This tool is for **EDUCATIONAL PURPOSES ONLY**.
It is **NOT** a substitute for medical diagnosis or treatment.
Always consult qualified healthcare professionals.
"""

COMMON_MISSPELLINGS = {
    "vomitting": "vomiting",
    "diahrrhea": "diarrhea",
    "diarrhoea": "diarrhea",
}

#helpers
def clean_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.strip().lower()
    for bad, good in COMMON_MISSPELLINGS.items():
        s = re.sub(r"\b" + re.escape(bad) + r"\b", good, s)
    s = re.sub(r"\s+", " ", s)
    s = s.replace(";", ",")
    return s

def split_symptom_list(s: str) -> List[str]:
    if not isinstance(s, str):
        s = str(s)
    parts = re.split(r"[,/]| and ", s)
    parts = [clean_text(p) for p in parts if p and clean_text(p) != ""]
    return list(dict.fromkeys(parts))

#dataset load
def load_disease_symptoms(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={df.columns[0]: "disease", df.columns[1]: "symptoms"})[["disease", "symptoms"]]
    df["disease"] = df["disease"].astype(str).map(clean_text)
    df["symptoms"] = df["symptoms"].astype(str).map(clean_text)
    df["symptom_list"] = df["symptoms"].apply(split_symptom_list)
    df["embed_text"] = (df["disease"] + " : " + df["symptoms"]).map(lambda x: x.strip())
    return df

def load_precautions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={df.columns[0]: "disease", df.columns[1]: "precautions"})[["disease", "precautions"]]
    df["disease"] = df["disease"].astype(str).map(clean_text)
    df["precautions"] = df["precautions"].astype(str).map(clean_text)
    return df

#model
def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_norm = a / (a.norm(dim=1, keepdim=True) + 1e-8)
    b_norm = b / (b.norm(dim=1, keepdim=True) + 1e-8)
    return torch.mm(a_norm, b_norm.t())

def compute_scores(user_emb, embeddings, ds_df_local: pd.DataFrame):
    sims = cosine_sim(user_emb, embeddings).squeeze(0).cpu().numpy()
    sims = (sims - sims.min()) / (sims.max() - sims.min() + 1e-8)
    return [s * 100.0 for s in sims]

def get_top_matches(user_text, model, embeddings, ds_df_local, top_k=8):
    with torch.no_grad():
        user_emb = model.encode([user_text], convert_to_tensor=True)
    scores = compute_scores(user_emb, embeddings, ds_df_local)

    idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    seen, results = set(), []
    for i in idxs:
        d = ds_df_local.iloc[i]["disease"]
        if d not in seen:
            seen.add(d)
            results.append({
                "disease": d,
                "score": scores[i],
                "symptoms_example": ds_df_local.iloc[i]["symptoms"],
            })
        if len(results) >= top_k:
            break
    return results

#L.D
ds_df = load_disease_symptoms(DISEASE_SYMPTOMS_CSV)
prec_df = load_precautions(PRECAUTION_CSV)
prec_map = {row["disease"]: row["precautions"] for _, row in prec_df.iterrows()}

def get_precautions_for(disease: str) -> str:
    return prec_map.get(disease, "No specific precautions found.")

#M.Cache
@st.cache_resource(show_spinner=True)
def load_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    with torch.no_grad():
        emb = model.encode(
            ds_df["embed_text"].tolist(),
            convert_to_tensor=True,
            batch_size=64,
            show_progress_bar=True,
        )
    return model, emb

with st.spinner("Loading model..."):
    model, embeddings = load_model()

#Ani
st.markdown(
    """
<style>
/* App background */
.stApp {
    background: radial-gradient(circle at 20% 0%,
                rgba(56, 189, 248, 0.25),
                rgba(15, 23, 42, 0.98)),
                linear-gradient(135deg, #020617, #020617);
    background-attachment: fixed;
}

/* Main container */
.block-container {
    background: rgba(15, 23, 42, 0.85);
    padding: 2rem 2.5rem;
    border-radius: 1.25rem;
    border: 1.5px solid rgba(56,189,248,0.45);
    box-shadow: 0 0 22px rgba(15,23,42,0.9);
}

/* Title glow */
h1 {
    color: #e0f2fe;
    text-shadow: 0 0 10px #38bdf8, 0 0 24px #0ea5e9;
}

/* Multiselect styling */
.stMultiSelect div[data-baseweb="select"] {
    background: rgba(15,23,42,0.9);
    border-radius: 12px;
    border: 1.5px solid rgba(56,189,248,0.6);
    box-shadow: 0 0 14px rgba(15,23,42,0.8);
}

/* Selected bar */
.selected-bar {
    background: rgba(10, 20, 35, 0.8);
    padding: 10px 14px;
    border-radius: 14px;
    margin-top: 10px;
    border: 1.5px solid rgba(56,189,248,0.5);
    box-shadow: 0 0 22px rgba(56,189,248,0.45);
}
.selected-pill {
    display: inline-block;
    padding: 6px 12px;
    margin: 4px;
    border-radius: 999px;
    background: rgba(56,189,248,0.25);
    border: 1.5px solid rgba(56,189,248,0.8);
    color: #e0faff;
    font-size: 13px;
    box-shadow: 0 0 14px rgba(56,189,248,0.85);
}

/* Start Check button neon glow */
.stButton>button {
    background: rgba(15, 23, 42, 0.95);
    border: 2px solid rgba(56,189,248,0.9);
    color: #e0f2fe;
    padding: 10px 18px;
    border-radius: 10px;
    font-size: 18px;
    font-weight: 600;
    box-shadow: 0 0 22px rgba(56,189,248,0.5);
    transition: 0.25s ease;
}
.stButton>button:hover {
    background: rgba(56,189,248,0.25);
    transform: translateY(-3px) scale(1.05);
    box-shadow: 0 0 35px rgba(56,189,248,0.95);
}

/* Floating result tiles */
.tile-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 1.2rem;
    margin-top: 1.3rem;
}
.result-tile {
    position: relative;
    background: rgba(15,23,42,0.95);
    border-radius: 1.1rem;
    padding: 1rem 1.2rem;
    border: 1.5px solid rgba(56,189,248,0.4);
    box-shadow: 0 0 18px rgba(15,23,42,0.9);
    opacity: 0;
    animation: tileFadeUp 0.7s ease forwards;
}
.result-tile:hover {
    transform: translateY(-6px) scale(1.02);
    box-shadow: 0 0 26px rgba(56,189,248,0.85);
}

/* Tile animation */
@keyframes tileFadeUp {
    0% { opacity: 0; transform: translateY(28px); }
    100% { opacity: 1; transform: translateY(0); }
}
</style>
""",
    unsafe_allow_html=True,
)

#partile bg
components.html(
    """
<canvas id="bg-canvas" style="position:fixed;top:0;left:0;width:100%;height:100%;z-index:-1;"></canvas>
<script>
const c = document.getElementById('bg-canvas');
const x = c.getContext('2d');
let p = [];
function resize(){c.width=innerWidth;c.height=innerHeight;}
window.addEventListener('resize',resize);resize();
function init(){p=[];
  for(let i=0;i<45;i++){
    p.push({x:Math.random()*c.width,
            y:Math.random()*c.height,
            r:Math.random()*2+1,
            dx:(Math.random()-.5)*.4,
            dy:(Math.random()-.5)*.4});
  }
}
init();
function loop(){
  x.clearRect(0,0,c.width,c.height);
  p.forEach(pt=>{
    pt.x+=pt.dx; pt.y+=pt.dy;
    if(pt.x<0||pt.x>c.width) pt.dx*=-1;
    if(pt.y<0||pt.y>c.height) pt.dy*=-1;
    x.beginPath();
    x.arc(pt.x,pt.y,pt.r,0,Math.PI*2);
    x.fillStyle="rgba(56,189,248,0.55)";
    x.fill();
  });
  requestAnimationFrame(loop);
}
loop();
</script>
""",
    height=0,
    width=0,
)

#M.UI
st.title("Health Symptom Checker (Educational)")
st.markdown(DISCLAIMER)

# Build full symptom list
all_symptoms = sorted(set(s for row in ds_df["symptom_list"] for s in row))

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Select your symptoms")
    selected_symptoms = st.multiselect(
        "Search and select symptoms from the dataset:",
        all_symptoms,
        placeholder="Type to search symptoms like 'fever', 'headache', 'cough'...",
    )

with col2:
    st.subheader("Tips")
    st.write("• Select all symptoms you are currently experiencing.")
    st.write("• Use the search box to quickly find symptoms.")
    st.write("• This is **not** a diagnosis — only educational guidance.")

st.markdown("### Selected Symptoms")
selected_bar = st.empty()

if selected_symptoms:
    pills_html = "".join(
        f"<span class='selected-pill'>{sym}</span>" for sym in selected_symptoms
    )
    selected_bar.markdown(
        f"<div class='selected-bar'>{pills_html}</div>",
        unsafe_allow_html=True,
    )
else:
    selected_bar.markdown(
        "<div class='selected-bar'>No symptoms selected yet.</div>",
        unsafe_allow_html=True,
    )

run_btn = st.button("Start Check")

#R.predict
if run_btn:
    if not selected_symptoms:
        st.warning("Please select at least one symptom before running the check.")
    else:
        user_text = ", ".join(selected_symptoms)
        cleaned = clean_text(user_text)
        results = get_top_matches(cleaned, model, embeddings, ds_df)

        st.markdown("## Possible Conditions")
        st.markdown("<div class='tile-grid'>", unsafe_allow_html=True)

        for i, r in enumerate(results, start=1):
            dis = r["disease"]
            score = r["score"]
            severity = (
                "SEVERE"
                if any(x in dis for x in ["pneumonia", "meningitis", "tuberculosis"])
                else (
                    "MODERATE"
                    if any(x in dis for x in ["influenza", "bronchitis", "uti", "dengue"])
                    else "MILD"
                )
            )
            precautions = get_precautions_for(dis)

            tile_html = f"""
            <div class='result-tile'>
                <h3>🔹 {i}. {dis.title()}</h3>
                <p><strong>Match Score:</strong> {score:.1f}%</p>
                <p><strong>Severity:</strong> {severity}</p>
                <p><strong>Typical Symptoms:</strong> {r['symptoms_example']}</p>
                <p><strong>Precautions:</strong> {precautions}</p>
            </div>
            """
            st.markdown(tile_html, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### Final Recommendation")
        best = results[0]
        if best["score"] >= 90:
            st.success(f"Very Strong Match: {best['disease'].title()} ({best['score']:.1f}%)")
        elif best["score"] >= 70:
            st.info(f"Strong Match: {best['disease'].title()} ({best['score']:.1f}%)")
        elif best["score"] >= 55:
            st.warning(f"Moderate Match: {best['disease'].title()} ({best['score']:.1f}%)")
        else:
            st.error("Low confidence — insufficient match.")

        st.info("**This is NOT a diagnosis. Always consult a qualified healthcare provider.**")
else:
    st.write("Select your symptoms and press **Start Check** to see possible conditions.")
