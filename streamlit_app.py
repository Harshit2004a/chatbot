"""
Health Symptom Checker (Educational)
- Animated interactive UI:
  • Particles + reactive background
  • Typing glow
  • Ripple clicks
  • Floating tiles for each result (staggered + halo glow)
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

This tool is for EDUCATIONAL PURPOSES ONLY.
It is NOT a substitute for professional medical advice, diagnosis, or treatment.
Always consult a qualified doctor. Use at your own risk.
"""

COMMON_MISSPELLINGS = {
    "vomitting": "vomiting",
    "diahrrhea": "diarrhea",
    "diarrhoea": "diarrhea",
}

#Help
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

#Loaders
def load_disease_symptoms(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    possible_sym_cols = ["symptoms", "symptom", "symptom_list", "symptoms_list", "Signs", "Signs & Symptoms"]
    possible_dis_cols = ["disease", "diseases", "Disease", "illness", "Condition", "Diagnosis"]

    sym_col = next((c for c in df.columns if c.lower() in [s.lower() for s in possible_sym_cols]), df.columns[1])
    dis_col = next((c for c in df.columns if c.lower() in [d.lower() for d in possible_dis_cols]), df.columns[0])

    df = df.rename(columns={dis_col: "disease", sym_col: "symptoms"})[["disease", "symptoms"]]
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

#Match
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
                "symptom_list": ds_df_local.iloc[i]["symptom_list"],
            })
        if len(results) >= top_k:
            break
    return results

#UID
st.markdown(
    """
<style>
/* App background */
.stApp {
    background: radial-gradient(circle at var(--mx, 50%) var(--my, 20%),
                rgba(56, 189, 248, 0.18),
                rgba(8, 47, 73, 0.95)),
                linear-gradient(135deg, #020617, #0f172a);
    background-attachment: fixed;
}

/* Main container with neon pulse */
.block-container {
    background: rgba(15, 23, 42, 0.80);
    padding: 2rem 2.5rem;
    border-radius: 1.25rem;
    backdrop-filter: blur(18px);
    position: relative;
    border: 1.5px solid rgba(56,189,248,0.45);
    box-shadow: 0 0 18px rgba(56,189,248,0.4), inset 0 0 12px rgba(56,189,248,0.25);
    animation: containerPulse 6s ease-in-out infinite alternate;
}

/* Title breathing glow */
h1 {
    animation: titleGlow 3.5s ease-in-out infinite alternate;
}

/* Textarea styling + glow */
textarea {
    border: 2px solid rgba(56, 189, 248, 0.5) !important;
    border-radius: 1.2rem !important;
    animation: glowOff 1s ease-out forwards;
    transition: 0.3s ease;
}
textarea:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 18px rgba(56,189,248,0.75) !important;
}

/* Floating Tiles + Staggered Animation + Halo Glow */
.tile-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 1.2rem;
    margin-top: 1.5rem;
}

.result-tile {
    position: relative;
    background: rgba(26, 33, 57, 0.78);
    border-radius: 1.2rem;
    padding: 1rem 1.2rem;
    border: 1.5px solid rgba(56,189,248,0.25);
    box-shadow: 0 0 16px rgba(56,189,248,0.35);
    transition: transform 0.45s ease, box-shadow 0.45s ease;
    opacity: 0;
    animation: tileFadeUp 0.8s ease forwards, tileFloat 5s ease-in-out infinite;
}

/* Spinning Glow Halo under tile */
.result-tile::before {
    content: "";
    position: absolute;
    top: 50%; left: 50%;
    width: 160px; height: 160px;
    transform: translate(-50%, -50%);
    border-radius: 50%;
    background: radial-gradient(rgba(56,189,248,0.55), rgba(56,189,248,0.0));
    filter: blur(45px);
    z-index: -1;
    animation: glowSpin 12s linear infinite;
}

/* Hover behavior */
.result-tile:hover {
    transform: translateY(-12px) scale(1.05);
    box-shadow: 0 0 40px rgba(56,189,248,0.95);
    border-color: rgba(56,189,248,1);
}

/* Hover glow for subheadings */
h2:hover, h3:hover {
    text-shadow: 0 0 12px #38bdf8;
}

/* Keyframes */
@keyframes glowOn {
    0% { box-shadow:0 0 0px rgba(56,189,248,0.0); }
    50% { box-shadow:0 0 18px rgba(56,189,248,0.9); }
    100% { box-shadow:0 0 0px rgba(56,189,248,0.0); }
}
@keyframes glowOff {
    from { box-shadow: 0 0 15px rgba(56,189,248,0.65); }
    to { box-shadow: 0 0 0px rgba(56,189,248,0.0); }
}
@keyframes titleGlow {
    0% { text-shadow: 0 0 8px #38bdf8, 0 0 16px #0ea5e9; }
    100% { text-shadow: 0 0 22px #38bdf8, 0 0 35px #0ea5e9; }
}
@keyframes containerPulse {
    0%   { box-shadow: 0 0 8px rgba(56,189,248,0.45); }
    100% { box-shadow: 0 0 26px rgba(56,189,248,0.85); }
}
@keyframes tileFadeUp {
    0%   { opacity: 0; transform: translateY(35px) scale(0.98); }
    100% { opacity: 1; transform: translateY(0) scale(1); }
}
@keyframes tileFloat {
    0%, 100% { transform: translateY(0); }
    50%      { transform: translateY(-6px); }
}
@keyframes glowSpin {
    from { transform: translate(-50%, -50%) rotate(0deg); }
    to   { transform: translate(-50%, -50%) rotate(360deg); }
}
</style>
""",
    unsafe_allow_html=True,
)

#typ glow
components.html(
    """
<script>
document.addEventListener("DOMContentLoaded", function() {
    const ta = window.parent.document.querySelector("textarea");
    if (!ta) return;
    ta.addEventListener("input", function() {
        ta.style.animation = "none";
        void ta.offsetWidth;
        ta.style.animation = "glowOn 0.6s ease-out";
    });
});
</script>
""",
    height=0,
    width=0,
)

#bg
components.html(
    """
<canvas id="particle-canvas" style="position:fixed;top:0;left:0;width:100%;height:100%;z-index:-1;"></canvas>
<script>
const canvas=document.getElementById('particle-canvas');
const ctx=canvas.getContext('2d');
let particles=[]; let mouse={x:0,y:0};

function resize(){
  canvas.width=window.innerWidth; canvas.height=window.innerHeight;
}
window.addEventListener('resize', resize); resize();

function initParticles(num=60){
  particles=[];
  for(let i=0;i<num;i++){
    particles.push({x:Math.random()*canvas.width,y:Math.random()*canvas.height,r:Math.random()*2+1,
      dx:(Math.random()-.5)*.6,dy:(Math.random()-.5)*.6});
  }
}
initParticles();

function animate(){
  ctx.clearRect(0,0,canvas.width,canvas.height);
  particles.forEach(p=>{
    p.x+=p.dx; p.y+=p.dy;
    if(p.x<0||p.x>canvas.width)p.dx*=-1;
    if(p.y<0||p.y>canvas.height)p.dy*=-1;
    ctx.beginPath(); ctx.arc(p.x,p.y,p.r,0,Math.PI*2);
    ctx.fillStyle="rgba(56,189,248,.8)"; ctx.fill();
    let dist=Math.hypot(mouse.x-p.x,mouse.y-p.y);
    if(dist<120){
      ctx.beginPath(); ctx.moveTo(p.x,p.y);
      ctx.lineTo(mouse.x,mouse.y);
      ctx.strokeStyle="rgba(56,189,248,"+(1-dist/120)+")"; ctx.stroke();
    }
  });
  requestAnimationFrame(animate);
}
animate();
window.addEventListener("mousemove",e=>{mouse.x=e.clientX;mouse.y=e.clientY;});
</script>
""",
    height=0,
    width=0,
)

#Click eff
components.html(
    """
<style>
#ripple-container {
  position: fixed;
  top: 0; left: 0;
  width: 100vw; height: 100vh;
  overflow: hidden;
  pointer-events: none;
  z-index: 5;
}
.ripple {
  position: absolute;
  border-radius: 50%;
  background: rgba(56, 189, 248, 0.4);
  transform: scale(0);
  animation: rippleEffect 0.8s ease-out forwards;
}
@keyframes rippleEffect {
  to {
    transform: scale(20);
    opacity: 0;
  }
}
</style>

<div id="ripple-container"></div>

<script>
document.addEventListener("click", function(e) {
  const container = document.getElementById("ripple-container");
  const ripple = document.createElement("div");
  ripple.classList.add("ripple");
  ripple.style.left = e.clientX + "px";
  ripple.style.top = e.clientY + "px";
  ripple.style.width = ripple.style.height = Math.max(window.innerWidth, window.innerHeight) + "px";
  container.appendChild(ripple);
  setTimeout(() => ripple.remove(), 900);
});
</script>
""",
    height=0,
    width=0,
)

#UI Area
st.title("Health Symptom Checker (Educational)")
st.markdown(DISCLAIMER)

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Enter Symptoms")
    user_symptoms = st.text_area(
        "Describe your symptoms separated by commas:",
        placeholder="fever, cold, headache, dry cough",
        height=140,
    )
    run_btn = st.button("Start Check")

with col2:
    st.subheader("Tips")
    st.write("- Example: `fever, fatigue, vomiting`")
    st.write("- Use specific symptoms for better matching")
    st.write("- This is for learning and awareness, not a diagnosis")

#Loading
ds_df = load_disease_symptoms(DISEASE_SYMPTOMS_CSV)
prec_df = load_precautions(PRECAUTION_CSV)

prec_map = {clean_text(r["disease"]): r["precautions"] for _, r in prec_df.iterrows()}

def get_precautions_for(d):
    return prec_map.get(clean_text(d), "No specific precautions found.")

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

with st.spinner("Loading model…"):
    model, embeddings = load_model()

#R.C
if run_btn:
    if not user_symptoms.strip():
        st.warning("Please enter symptoms.")
    else:
        cleaned = clean_text(user_symptoms)
        results = get_top_matches(cleaned, model, embeddings, ds_df)

        st.markdown("### Possible Conditions")
        st.markdown("<div class='tile-grid'>", unsafe_allow_html=True)

        for i, r in enumerate(results, start=1):
            dis, score = r["disease"], r["score"]
            severity = (
                "SEVERE"
                if any(x in dis for x in ["pneumonia", "meningitis", "tuberculosis"])
                else (
                    "MODERATE"
                    if any(x in dis for x in ["influenza", "bronchitis", "uti", "dengue"])
                    else "MILD"
                )
            )

            delay = i * 0.15  #delay
            tile_html = f"""
            <div class='result-tile' style="animation-delay:{delay}s;">
                <h3>🔹 {i}. {dis.title()}</h3>
                <p><strong>Match Score:</strong> {score:.1f}%</p>
                <p><strong>Severity:</strong> {severity}</p>
                <p><strong>Symptoms:</strong> {r['symptoms_example']}</p>
                <p><strong>Precautions:</strong> {get_precautions_for(dis)}</p>
            </div>
            """
            st.markdown(tile_html, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # F.Recom
        best = results[0]
        st.markdown("### Final Recommendation")
        if best["score"] >= 90:
            st.success(f"Very Strong Match: {best['disease'].title()} ({best['score']:.1f}%)")
        elif best["score"] >= 70:
            st.info(f"Strong Match: {best['disease'].title()} ({best['score']:.1f}%)")
        elif best["score"] >= 55:
            st.warning(f"Moderate Match: {best['disease'].title()} ({best['score']:.1f}%)")
        else:
            st.write("Weak match — insufficient confidence.")

        st.info("**This is NOT a diagnosis. Always consult a qualified healthcare provider.**")
else:
    st.write("Enter symptoms and press **Start Check** to view possible conditions.")
