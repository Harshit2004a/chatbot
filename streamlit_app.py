"""
Health Symptom Checker (Educational) - Streamlit single-file app
- No adaptive Q/A shown in UI
- Title: "Health Symptom Checker (Educational)"
- Starts check and immediately shows final results (top matches)
"""

import os
import re
from typing import List, Tuple
import streamlit as st
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer



DISEASE_SYMPTOMS_CSV = "/workspaces/chatbot/DiseaseAndSymptoms.csv"
PRECAUTION_CSV       = "/workspaces/chatbot/Disease precaution.csv"


SCREENSHOT_PATH = "/mnt/data/Screenshot 2025-11-25 at 3.13.10 AM.png"

DISCLAIMER = """
**MEDICAL DISCLAIMER**

This tool is for EDUCATIONAL PURPOSES ONLY.
It is NOT a substitute for professional medical advice, diagnosis, or treatment.
Always consult a qualified doctor. Use at your own risk.
"""

COMMON_MISSPELLINGS = {
    'vomitting': 'vomiting',
    'diahrrhea': 'diarrhea',
    'diarrhoea': 'diarrhea'
}

#H
def clean_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.strip().lower()
    for bad, good in COMMON_MISSPELLINGS.items():
        s = re.sub(r'\b' + re.escape(bad) + r'\b', good, s)
    s = re.sub(r'\s+', ' ', s)
    s = s.replace(';', ',')
    return s

def split_symptom_list(s: str) -> List[str]:
    if not isinstance(s, str):
        s = str(s)
    parts = re.split(r'[,/]| and ', s)
    parts = [clean_text(p) for p in parts if p and clean_text(p) != '']
    return list(dict.fromkeys(parts))

def load_disease_symptoms(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Symptoms CSV not found: {path}")
    df = pd.read_csv(path)
    possible_sym_cols = ['symptoms', 'symptom', 'symptom_list', 'symptoms_list', 'Signs', 'Signs & Symptoms']
    possible_dis_cols = ['disease', 'diseases', 'Disease', 'illness', 'Condition', 'Diagnosis']

    sym_col = next((c for c in df.columns if c.lower() in [s.lower() for s in possible_sym_cols]), None)
    dis_col = next((c for c in df.columns if c.lower() in [d.lower() for d in possible_dis_cols]), None)

    if not sym_col or not dis_col:
        if len(df.columns) >= 2:
            dis_col, sym_col = df.columns[0], df.columns[1]
        else:
            raise ValueError("Could not detect disease/symptoms columns in CSV.")

    df = df.rename(columns={dis_col: 'disease', sym_col: 'symptoms'})[['disease', 'symptoms']]
    df['disease'] = df['disease'].astype(str).map(clean_text)
    df['symptoms'] = df['symptoms'].astype(str).map(clean_text)
    df = df.drop_duplicates(subset=['disease', 'symptoms']).reset_index(drop=True)
    df['symptom_list'] = df['symptoms'].apply(split_symptom_list)
    df['embed_text'] = (df['disease'] + " : " + df['symptoms']).map(lambda x: x.strip())
    return df

def load_precautions(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Precaution CSV not found: {path}")
    df = pd.read_csv(path)
    possible_dis_cols = ['disease', 'Disease', 'Condition', 'illness']
    possible_prec_cols = ['precautions', 'precaution', 'advice', 'Precautions', 'Advice']

    dis_col = next((c for c in df.columns if c.lower() in [d.lower() for d in possible_dis_cols]), None)
    prec_col = next((c for c in df.columns if c.lower() in [p.lower() for p in possible_prec_cols]), None)

    if not dis_col or not prec_col:
        if len(df.columns) >= 2:
            dis_col, prec_col = df.columns[0], df.columns[1]
        else:
            raise ValueError("Could not detect disease/precautions columns in CSV.")

    df = df.rename(columns={dis_col: 'disease', prec_col: 'precautions'})[['disease', 'precautions']]
    df['disease'] = df['disease'].astype(str).map(clean_text)
    df['precautions'] = df['precautions'].astype(str).map(clean_text)
    return df

#M
@st.cache_resource(show_spinner=False)
def load_model_and_embeddings():
    # model and embeddings will be computed using the global ds_df (must be loaded before calling)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    with torch.no_grad():
        texts = ds_df['embed_text'].tolist()
        embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=True, batch_size=64)
    return model, embeddings

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_norm = a / (a.norm(dim=1, keepdim=True) + 1e-8)
    b_norm = b / (b.norm(dim=1, keepdim=True) + 1e-8)
    return torch.mm(a_norm, b_norm.t())

def compute_adjusted_scores(user_emb: torch.Tensor, confirmed: List[str], denied: List[str],
                            embeddings: torch.Tensor, ds_df_local: pd.DataFrame) -> List[float]:
    sims = cosine_sim(user_emb, embeddings).squeeze(0).cpu().numpy()
    sims = (sims - sims.min()) / (sims.max() - sims.min() + 1e-8)
    adj = sims.copy()
    for i, row in ds_df_local.iterrows():
        s_list = row['symptom_list']
        for d in denied:
            if d and any(d in s for s in s_list):
                adj[i] = max(0.0, adj[i] - 0.25)
        for c in confirmed:
            if c and any(c in s for s in s_list):
                adj[i] = min(1.0, adj[i] + 0.15)
    return adj.tolist()

def get_top_matches(user_text: str, confirmed: List[str], denied: List[str], model, embeddings, ds_df_local: pd.DataFrame, top_k: int = 5):
    with torch.no_grad():
        user_emb = model.encode([user_text], convert_to_tensor=True)
    adj_scores = compute_adjusted_scores(user_emb, confirmed, denied, embeddings, ds_df_local)
    scores_percent = [s * 100.0 for s in adj_scores]
    idxs = sorted(range(len(scores_percent)), key=lambda i: scores_percent[i], reverse=True)
    seen = set()
    results = []
    for i in idxs:
        d = ds_df_local.iloc[i]['disease']
        if d in seen: continue
        seen.add(d)
        results.append({
            'disease': ds_df_local.iloc[i]['disease'],
            'score': scores_percent[i],
            'symptoms_example': ds_df_local.iloc[i]['symptoms'],
            'symptom_list': ds_df_local.iloc[i]['symptom_list']
        })
        if len(results) >= top_k:
            break
    return results

#UI
st.set_page_config(page_title="Health Symptom Checker (Educational)", layout="wide")
st.title("Health Symptom Checker (Educational)")
st.markdown(DISCLAIMER)

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Enter symptoms")
    user_symptoms = st.text_area(
        "Type symptoms separated by commas (e.g. fever, dry cough, tired):",
        placeholder="fever, headache, cold, tiredness, fatigue"
    )
    start_btn = st.button("Start Check")

with col_right:
    st.subheader("Tips")
    st.write("- Be specific (e.g., 'dry cough', 'vomiting').")
    st.write("- This is educational only; answer honestly for better results.")
    if os.path.exists(SCREENSHOT_PATH):
        st.image(SCREENSHOT_PATH, caption="App screenshot", use_column_width=True)

#LD
try:
    ds_df = load_disease_symptoms(DISEASE_SYMPTOMS_CSV)
    prec_df = load_precautions(PRECAUTION_CSV)
except Exception as e:
    st.error(f"Error loading CSVs: {e}")
    st.stop()

#Prec
prec_map = {}
for _, r in prec_df.iterrows():
    d = r['disease'].strip()
    if d == '':
        continue
    if d in prec_map:
        prec_map[d] = prec_map[d] + "; " + r['precautions']
    else:
        prec_map[d] = r['precautions']

def get_precautions_for(disease: str) -> str:
    return prec_map.get(clean_text(disease), "No specific precautions found in database.")

#Cac
with st.spinner("Loading model and embeddings (first run may take a minute)..."):
    model, embeddings = load_model_and_embeddings()

#Q
if start_btn:
    if not user_symptoms or not user_symptoms.strip():
        st.warning("Please enter at least one symptom to run the check.")
    else:
        # Use cleaned user text and no confirmed/denied follow-ups (since we hide adaptive questioning)
        user_text = clean_text(user_symptoms)
        final_results = get_top_matches(user_text, [], [], model, embeddings, ds_df, top_k=8)

        st.markdown("### Possible Conditions")
        for rank, r in enumerate(final_results, start=1):
            disease = r['disease']
            score = r['score']
            se = "SEVERE" if any(x in disease for x in ['pneumonia','meningitis','tuberculosis']) else ("MODERATE" if any(x in disease for x in ['influenza','bronchitis','uti','mononucleosis','dengue']) else "MILD")
            precautions = get_precautions_for(disease)

            st.write(f"**{rank}. {disease.title()}** — *{score:.1f}%*")
            st.write(f"• Severity: {se}")
            st.write(f"• Typical symptoms: {r['symptoms_example']}")
            st.write(f"• Precautions: {precautions}")
            st.markdown("---")

        #F-R
        best = final_results[0] if final_results else None
        st.markdown("### Final Recommendation")
        if best and best['score'] >= 90:
            st.success(f"VERY STRONG MATCH: **{best['disease'].title()}** ({best['score']:.1f}%)")
            st.write("Recommendation: Seek medical evaluation within 24 hours (or sooner for severe symptoms).")
        elif best and best['score'] >= 70:
            st.info(f"STRONG MATCH: **{best['disease'].title()}** ({best['score']:.1f}%)")
            st.write("Recommendation: Arrange to see a doctor in 48–72 hours or earlier if symptoms worsen.")
        elif best and best['score'] >= 55:
            st.warning(f"MODERATE MATCH: **{best['disease'].title()}** ({best['score']:.1f}%)")
            st.write("Recommendation: Monitor symptoms; consult if worsening or persistent.")
        else:
            st.write("WEAK MATCH or insufficient data to reach high confidence.")
            st.write("Recommendation: Monitor symptoms and consult if persistent/worse.")

        st.info("**This is NOT a diagnosis. Always consult a qualified healthcare provider.**")

else:
    st.write("Enter symptoms and press **Start Check** to see possible conditions.")
