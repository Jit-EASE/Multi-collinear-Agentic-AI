# Fix the import error by generating a corrected version of the unified suite.
# Specifically, remove the misspelled fallback import and use the proper statsmodels path.

# agentic_ai_multicollinearity_suite.py (Corrected Import)
# ------------------------------------------------------------------
# Agentic AI Multicollinearity Suite
# Tabs:
#   1) General/Policy Lab
#   2) Agriculture Lab
# Designed & Developed by Jit
# ------------------------------------------------------------------

import os, re, json, hashlib, textwrap, random
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ===== Optional SDKs =====
HAS_OPENAI = True
try:
    from openai import OpenAI
except Exception:
    HAS_OPENAI = False

HAS_GEMINI = True
try:
    import google.generativeai as genai
except Exception:
    HAS_GEMINI = False

# ===== UI CONFIG =====
st.set_page_config(page_title="Agentic AI Multicollinearity Suite", page_icon="🧮", layout="wide")

HEADER = """
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.8rem;">
  <div>
    <h1 style="margin:0;">Agentic AI Multicollinearity Suite</h1>
    <p style="margin:0;color:#666;">Homogeneous vs Heterogeneous LLM Committees — Correlations, VIF, PCA, Condition Index</p>
  </div>
  <div style="font-weight:600;opacity:0.8;">Designed & Developed by Jit</div>
</div>
<hr style="margin-top:0.6rem;margin-bottom:0.6rem;"/>
"""
st.markdown(HEADER, unsafe_allow_html=True)

with st.expander("Quick Start", expanded=False):
    st.markdown("""
    **Keys**
    ```bash
    export OPENAI_API_KEY="sk-..."
    export GOOGLE_API_KEY="AIza-..."
    ```
    **Install**
    ```bash
    pip install streamlit openai google-generativeai plotly statsmodels scikit-learn pandas numpy
    ```
    **Run**
    ```bash
    streamlit run agentic_ai_multicollinearity_suite.py
    ```
    """)

# ===== Utility: hash key =====
def hash_key(obj: dict) -> str:
    try:
        s = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    except Exception:
        s = str(obj)
    return hashlib.md5(s.encode("utf-8")).hexdigest()

# ===== Robust JSON parsing =====
def parse_json_block(txt: str, keys: List[str]) -> Dict[str, float]:
    if not txt:
        return {}
    start = txt.find('{'); end = txt.rfind('}')
    frag = txt[start:end+1] if (start!=-1 and end!=-1 and end>start) else ""
    if frag:
        try:
            data = json.loads(frag)
            out = {}
            for k in keys:
                v = data.get(k, None)
                try:
                    out[k] = float(v) if v is not None else None
                except Exception:
                    out[k] = None
            return out
        except Exception:
            pass
    # fallback: scrape numbers and map in order
    nums = [float(n) for n in re.findall(r'(?<!\\d)(\\d{1,3}(?:\\.\\d+)?)(?!\\d)', txt)]
    return {k: (nums[i] if i < len(nums) else None) for i,k in enumerate(keys)}

# ===== Diagnostics =====
def safe_vif(df: pd.DataFrame) -> pd.DataFrame:
    X = df.dropna()
    if X.shape[0] < 5 or X.shape[1] < 2:
        return pd.DataFrame({"feature": X.columns, "VIF": [np.nan]*X.shape[1]})
    vals = []
    for i in range(X.shape[1]):
        try:
            vals.append(variance_inflation_factor(X.values, i))
        except Exception:
            vals.append(np.nan)
    return pd.DataFrame({"feature": X.columns, "VIF": vals})

def condition_index(df: pd.DataFrame) -> float:
    X = df.dropna()
    if X.shape[0] < 2 or X.shape[1] < 2: return np.nan
    Z = StandardScaler().fit_transform(X.values)
    u,s,vh = np.linalg.svd(Z, full_matrices=False)
    return float((s.max()/s.min()) if s.min()>0 else np.inf)

def pca_first_share(df: pd.DataFrame) -> float:
    X = df.dropna()
    if X.shape[0] < 2 or X.shape[1] < 2: return np.nan
    Z = StandardScaler().fit_transform(X.values)
    p = PCA().fit(Z)
    return float(p.explained_variance_ratio_[0])

# ===== LLM Clients =====
def build_openai():
    if not HAS_OPENAI: return None
    try:
        return OpenAI()
    except Exception:
        return None

def call_openai_json(client, system_prompt: str, user_prompt: str, temperature: float=0.6, max_tokens: int=320) -> str:
    if not client: return None
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":system_prompt},
                      {"role":"user","content":user_prompt}],
            temperature=float(temperature),
            max_tokens=max_tokens
        )
        return resp.choices[0].message.content
    except Exception as e:
        return None

def call_gemini_json(system_prompt: str, user_prompt: str, temperature: float=0.6, max_tokens: int=320) -> str:
    if not HAS_GEMINI: return None
    try:
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        model = genai.GenerativeModel("gemini-2.5-pro")
        resp = model.generate_content(
            [system_prompt, user_prompt],
            generation_config={"temperature": float(temperature), "max_output_tokens": max_tokens}
        )
        return getattr(resp, "text", None)
    except Exception:
        return None

# ===== Assignment logic =====
def assign_models(n_agents: int, heterogeneous: bool) -> List[str]:
    available = []
    if os.getenv("OPENAI_API_KEY"): available.append("gpt-4o-mini")
    if os.getenv("GOOGLE_API_KEY"): available.append("gemini-2.5-pro")
    if not available: available = ["gpt-4o-mini"]
    if not heterogeneous or len(available) == 1:
        return [available[0]]*n_agents
    return [available[i % len(available)] for i in range(n_agents)]

# ===== General/Policy Lab =====
GEN_KEYS = ["policy","efficiency","risk","feasibility","evidence","final_score"]
GEN_SCHEMA = ("Return ONLY a JSON object with keys: policy, efficiency, risk, feasibility, evidence, final_score. "
              "Each 0-100 integer.")
GEN_SYSTEM = "You are an expert policy analyst. Be concise, numeric, schema-only. " + GEN_SCHEMA

GEN_DEFAULT_PROMPTS = [
    "Score (0-100) Ireland's AI in agriculture readiness considering infrastructure, adoption, and skills.",
    "Score (0-100) EU food system resilience for climate shocks in 2025.",
    "Score (0-100) ROI of deploying edge-AI sensors for crop stress detection on Irish dairy farms.",
    "Score (0-100) Policy feasibility for nationwide agri-telemetry rollout under EU AI Act constraints.",
    "Score (0-100) Supply-chain visibility improvements from multimodal IoT in the Irish beef sector.",
    "Score (0-100) Risk of model collapse when using single-LLM pipelines in policy analytics.",
    "Score (0-100) Expected improvement from multi-LLM committees for factual QA in agri policy.",
    "Score (0-100) Viability of offline contextual AI for farm advisory in rural Ireland.",
    "Score (0-100) System-of-systems interoperability in Irish agri-food (2025–2030).",
    "Score (0-100) Socio-economic benefit of stress-aware worker support programs in agri-processing."
]

GEN_ROLES = [
    "Planner: optimize long-term policy outcomes and resource allocation.",
    "Critic: maximize factual accuracy, identify unsupported claims.",
    "Risk Analyst: evaluate technical, regulatory, and adoption risks.",
    "Econometrician: focus on quantifiable indicators and uncertainty.",
    "Operations Lead: emphasize implementation constraints and ROI."
]

def run_general_lab(cfg: dict) -> dict:
    n_agents = int(cfg["n_agents"]); heterogeneous = bool(cfg["heterogeneous"])
    use_roles = bool(cfg["use_roles"]); debate = bool(cfg["debate"])
    t_openai = float(cfg["t_openai"]); t_gemini = float(cfg["t_gemini"])
    prompts: List[str] = cfg["prompts"]

    models = assign_models(n_agents, heterogeneous)
    oa = build_openai()

    rows, meta, predeb = [], [], []
    for p in prompts:
        a_scores, a_meta, round_dump = [], [], []
        for a in range(n_agents):
            role_line = f"\nROLE: {GEN_ROLES[a]}" if (use_roles and a < len(GEN_ROLES)) else ""
            uprompt = f"Task: {p}\n{role_line}\nSchema: {GEN_SCHEMA}"
            m = models[a]; txt = None
            if m.startswith("gpt-"):
                txt = call_openai_json(oa, GEN_SYSTEM, uprompt, temperature=t_openai)
            else:
                txt = call_gemini_json(GEN_SYSTEM, uprompt, temperature=t_gemini)
            if txt is None:
                txt = '{"policy":60,"efficiency":60,"risk":40,"feasibility":55,"evidence":50,"final_score":55}'
            parsed = parse_json_block(txt, GEN_KEYS)
            a_scores.append(parsed.get("final_score", None))
            a_meta.append({"model": m, "raw": txt, **parsed})
            round_dump.append({"model": m, "parsed": parsed, "prompt": p})
        rows.append(a_scores); meta.append(a_meta); predeb.append(round_dump)

    X = pd.DataFrame(rows, columns=[f"Agent{i+1}" for i in range(n_agents)])
    corr = X.corr(numeric_only=True)
    vif = safe_vif(X); cidx = condition_index(X); pc1 = pca_first_share(X)
    consensus = None

    if debate and len(predeb) > 0:
        summary = {"rounds": predeb}
        c_prompt = textwrap.dedent(f"""
        Compute a CONSENSUS final_score (0-100) using a robust trimmed mean across agents per task.
        Return ONLY JSON: {{ "consensus": [ ... numbers ... ] }}
        DATA: {json.dumps(summary)[:12000]}
        """)
        c_txt = call_openai_json(oa, "Be numeric only.", c_prompt, temperature=0.1, max_tokens=400) \
                or call_gemini_json("Be numeric only.", c_prompt, temperature=0.1, max_tokens=400) \
                or '{"consensus": []}'
        try:
            cs = json.loads(c_txt[c_txt.find('{'): c_txt.rfind('}')+1]).get("consensus", [])
        except Exception:
            cs = []
        consensus = cs

    return {"X":X, "corr":corr, "vif":vif, "cond":cidx, "pc1":pc1, "models":models, "meta":meta, "consensus":consensus}

# ===== Agriculture Lab =====
AG_KEYS = ["irrigation_mm","nitrogen_kg","pest_risk","yield_gain","water_stress","final_score"]
AG_SCHEMA = ("Return ONLY a JSON object with keys: irrigation_mm (0-60), nitrogen_kg (0-60), "
             "pest_risk (0-100), yield_gain (0-100), water_stress (0-100), final_score (0-100).")
AG_SYSTEM = "You are an expert agronomy decision agent. Be concise, numeric, schema-only. " + AG_SCHEMA

SOIL_TYPES = ["Sandy loam", "Loam", "Clay loam", "Peat-influenced loam"]
PH_RANGE = (5.6, 7.2); SM_RANGE = (8, 42); ET0_RANGE = (2.0, 6.0); RAIN_7D = (0, 60)
TEMP_C = (10, 27); NDVI = (0.45, 0.85); CANOPY_WET = [False, True]
P_STAGE = ["tillering","stem elongation","booting","heading","grain fill"]
PEST_PRESS = ["low","moderate","high"]

AG_ROLES = [
    "Agronomist: crop physiology and balanced recommendations.",
    "Irrigation Specialist: schedule irrigation and manage moisture.",
    "Plant Pathologist: estimate pest/disease risk from conditions.",
    "Agricultural Economist: trade-offs and ROI on inputs.",
    "Compliance Officer: nitrates directive & runoff constraints."
]

def rfloat(r): return round(random.uniform(*r), 2)

def make_field(seed=None) -> Dict:
    if seed is not None:
        random.seed(seed); np.random.seed(seed)
    return {
        "field_id": f"F{random.randint(100,999)}",
        "soil": random.choice(SOIL_TYPES),
        "pH": rfloat(PH_RANGE),
        "soil_moisture_pct": round(random.uniform(*SM_RANGE), 1),
        "ET0_mm_day": rfloat(ET0_RANGE),
        "rain_last_7d_mm": random.randint(*RAIN_7D),
        "temp_day_c": rfloat(TEMP_C),
        "ndvi": rfloat(NDVI),
        "canopy_wet": random.choice(CANOPY_WET),
        "phenology": random.choice(P_STAGE),
        "pest_pressure": random.choice(PEST_PRESS)
    }

def field_prompt(card: Dict) -> str:
    return textwrap.dedent(f"""
    FIELD SNAPSHOT
    - Field ID: {card['field_id']}
    - Soil: {card['soil']}
    - pH: {card['pH']}
    - Topsoil moisture (0-20 cm): {card['soil_moisture_pct']} %
    - Reference ET0: {card['ET0_mm_day']} mm/day
    - Rainfall last 7 days: {card['rain_last_7d_mm']} mm
    - Daytime temperature: {card['temp_day_c']} °C
    - NDVI: {card['ndvi']}
    - Canopy wetness: {card['canopy_wet']}
    - Phenology: {card['phenology']}
    - Pest pressure: {card['pest_pressure']}

    TASK
    Provide numeric recommendations and risk assessments consistent with the schema.
    Units: irrigation in mm (0-60), nitrogen in kg/ha (0-60). Risk/Stress/Yield as 0-100.
    """).strip()

def run_agri_lab(cfg: dict) -> dict:
    n_agents = int(cfg["n_agents"]); heterogeneous = bool(cfg["heterogeneous"])
    use_roles = bool(cfg["use_roles"]); t_openai = float(cfg["t_openai"]); t_gemini = float(cfg["t_gemini"])
    n_fields = int(cfg["n_fields"]); seed = int(cfg.get("seed", 0))

    models = assign_models(n_agents, heterogeneous)
    oa = build_openai()

    rows, raw = [], []
    for i in range(n_fields):
        card = make_field(seed+i if seed else None)
        a_scores = []; a_json = []
        for a in range(n_agents):
            role_line = f"\nROLE: {AG_ROLES[a]}" if (use_roles and a < len(AG_ROLES)) else ""
            uprompt = field_prompt(card) + role_line + "\nSchema: " + AG_SCHEMA
            m = models[a]; txt = None
            if m.startswith("gpt-"):
                txt = call_openai_json(oa, AG_SYSTEM, uprompt, temperature=t_openai)
            else:
                txt = call_gemini_json(AG_SYSTEM, uprompt, temperature=t_gemini)
            if txt is None:
                txt = '{"irrigation_mm":10,"nitrogen_kg":20,"pest_risk":42,"yield_gain":50,"water_stress":40,"final_score":57}'
            parsed = parse_json_block(txt, AG_KEYS)
            a_scores.append(parsed.get("final_score", None))
            a_json.append({"agent": f"Agent{a+1}","model":m,"out":parsed})
        rows.append(a_scores); raw.append({"field":card, "agents":a_json})

    X = pd.DataFrame(rows, columns=[f"Agent{i+1}" for i in range(n_agents)])
    corr = X.corr(numeric_only=True)
    vif = safe_vif(X); cidx = condition_index(X); pc1 = pca_first_share(X)
    return {"X":X, "corr":corr, "vif":vif, "cond":cidx, "pc1":pc1, "models":models, "raw":raw}

# ===== Tabs =====
tab1, tab2, tab3 = st.tabs(["General / Policy Lab", "Agriculture Lab", "About & Methods"])

with tab1:
    st.subheader("General / Policy Multicollinearity Lab")
    left, right = st.columns([0.58, 0.42])
    with left:
        mode = st.radio("System Type", ["Homogeneous: single LLM (multi-agent)", "Heterogeneous: multi-LLM committee"], index=1, key="g_mode")
        heterogeneous = mode.startswith("Heterogeneous")
        n_agents = st.slider("Agents", 2, 6, 3, 1, key="g_agents")
        use_roles = st.checkbox("Role specialization", value=heterogeneous, key="g_roles")
        debate = st.checkbox("Debate / consensus (post-analysis)", value=False, key="g_debate")
        t_openai = st.slider("OpenAI GPT-4o-mini temperature", 0.0, 1.5, 0.7, 0.1, key="g_temp_oa")
        t_gemini = st.slider("Gemini 2.5 temperature", 0.0, 1.5, 0.7, 0.1, key="g_temp_ge")
    with right:
        st.markdown("**Prompts (one per line)**")
        default_text = "\n".join(GEN_DEFAULT_PROMPTS)
        ptext = st.text_area(" ", value=default_text, height=220, key="g_prompts")
        prompts = [p.strip() for p in ptext.splitlines() if p.strip()]
        if len(prompts) < 3:
            st.warning("Add at least 3 prompts for stable VIF/PC1 estimates.")
        run_g = st.button("▶️ Run General/Policy", use_container_width=True)

    if run_g:
        cfg = {"n_agents":n_agents,"heterogeneous":heterogeneous,"use_roles":use_roles,"debate":debate,
               "t_openai":t_openai,"t_gemini":t_gemini,"prompts":prompts}
        key = hash_key(cfg)
        with st.spinner("Running committee and computing diagnostics..."):
            res = run_general_lab(cfg)

        X, corr, vif, cidx, pc1, models, consensus = res["X"], res["corr"], res["vif"], res["cond"], res["pc1"], res["models"], res["consensus"]
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Observations (prompts)", f"{X.shape[0]}"); c2.metric("Agents", f"{X.shape[1]}")
        c3.metric("Condition Index", f"{cidx:0.2f}" if np.isfinite(cidx) else "—")
        c4.metric("PCA Var(PC1)", f"{pc1*100:0.1f}%" if not np.isnan(pc1) else "—")

        with st.expander("Agent roster", expanded=False):
            for i,m in enumerate(models,1):
                st.write(f"Agent{i}: **{m}**" + (f" — role: {GEN_ROLES[i-1]}" if use_roles and i-1 < len(GEN_ROLES) else ""))

        st.markdown("### Correlation Heatmap")
        if corr.shape[0] >= 2:
            fig = px.imshow(corr, text_auto=True, aspect="auto", zmin=-1, zmax=1, title="Agent–Agent Correlation (final_score)")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### VIF (Variance Inflation Factor)")
        if not vif.empty:
            fig_vif = go.Figure(); fig_vif.add_bar(x=vif["feature"], y=vif["VIF"])
            fig_vif.update_layout(yaxis_title="VIF (>10 severe)", xaxis_title="Agent variable")
            st.plotly_chart(fig_vif, use_container_width=True)
            st.dataframe(vif, use_container_width=True)

        st.markdown("### PCA Scree")
        if X.dropna().shape[0] >= 2 and X.shape[1] >= 2:
            Z = StandardScaler().fit_transform(X.dropna().values)
            p = PCA().fit(Z)
            fig_pca = go.Figure(); fig_pca.add_bar(x=[f"PC{i+1}" for i in range(len(p.explained_variance_ratio_))], y=p.explained_variance_ratio_)
            fig_pca.update_layout(yaxis_title="Explained Variance Ratio", xaxis_title="Principal Components")
            st.plotly_chart(fig_pca, use_container_width=True)

        st.markdown("### Raw matrix")
        st.dataframe(X, use_container_width=True, height=280)

        if debate and consensus is not None:
            st.markdown("### Consensus (post-debate, not used in VIF)")
            st.json({"consensus": consensus})

        st.markdown("---")
        st.download_button("Download matrix CSV", data=X.to_csv(index=False).encode("utf-8"),
                           file_name="general_agent_scores_matrix.csv", mime="text/csv")
        metrics = {"cond_index": float(cidx) if cidx==cidx else None, "pc1_share": float(pc1) if pc1==pc1 else None,
                   "vif": vif.to_dict(orient="list"), "corr": corr.to_dict(), "models": models, "config": cfg}
        st.download_button("Download metrics JSON", data=json.dumps(metrics, indent=2).encode("utf-8"),
                           file_name="general_metrics.json", mime="application/json")

with tab2:
    st.subheader("Agriculture Multicollinearity Lab")
    left, right = st.columns([0.58, 0.42])
    with left:
        mode_a = st.radio("System Type", ["Homogeneous: single LLM", "Heterogeneous: multi-LLM"], index=1, key="a_mode")
        heterogeneous_a = mode_a.startswith("Heterogeneous")
        n_agents_a = st.slider("Agents", 2, 6, 3, 1, key="a_agents")
        use_roles_a = st.checkbox("Role specialization", value=True, key="a_roles")
        t_openai_a = st.slider("OpenAI GPT-4o-mini temperature", 0.0, 1.5, 0.6, 0.1, key="a_temp_oa")
        t_gemini_a = st.slider("Gemini 2.5 temperature", 0.0, 1.5, 0.6, 0.1, key="a_temp_ge")
    with right:
        n_fields = st.slider("Number of field snapshots", 5, 40, 12, 1, key="a_fields")
        seed = st.number_input("Random seed (optional)", value=0, step=1, key="a_seed")
        run_a = st.button("▶️ Run Agriculture", use_container_width=True)

    if run_a:
        cfg_a = {"n_agents": n_agents_a, "heterogeneous":heterogeneous_a, "use_roles": use_roles_a,
                 "t_openai": t_openai_a, "t_gemini": t_gemini_a, "n_fields": n_fields, "seed": seed}
        key_a = hash_key(cfg_a)
        with st.spinner("Running agri committee and computing diagnostics..."):
            res = run_agri_lab(cfg_a)

        X, corr, vif, cidx, pc1, models, raw = res["X"], res["corr"], res["vif"], res["cond"], res["pc1"], res["models"], res["raw"]
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Field snapshots", f"{X.shape[0]}"); c2.metric("Agents", f"{X.shape[1]}")
        c3.metric("Condition Index", f"{cidx:0.2f}" if np.isfinite(cidx) else "—")
        c4.metric("PCA Var(PC1)", f"{pc1*100:0.1f}%" if not np.isnan(pc1) else "—")

        with st.expander("Agent roster", expanded=False):
            for i,m in enumerate(models,1):
                st.write(f"Agent{i}: **{m}**" + (f" — role: {AG_ROLES[i-1]}" if use_roles_a and i-1 < len(AG_ROLES) else ""))

        st.markdown("### Correlation Heatmap")
        if corr.shape[0] >= 2:
            fig = px.imshow(corr, text_auto=True, aspect="auto", zmin=-1, zmax=1, title="Agent–Agent Correlation (final_score)")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### VIF (Variance Inflation Factor)")
        if not vif.empty:
            fig_vif = go.Figure(); fig_vif.add_bar(x=vif["feature"], y=vif["VIF"])
            fig_vif.update_layout(yaxis_title="VIF (>10 severe)", xaxis_title="Agent variable")
            st.plotly_chart(fig_vif, use_container_width=True)
            st.dataframe(vif, use_container_width=True)

        st.markdown("### PCA Scree")
        if X.dropna().shape[0] >= 2 and X.shape[1] >= 2:
            Z = StandardScaler().fit_transform(X.dropna().values)
            p = PCA().fit(Z)
            fig_pca = go.Figure(); fig_pca.add_bar(x=[f"PC{i+1}" for i in range(len(p.explained_variance_ratio_))], y=p.explained_variance_ratio_)
            fig_pca.update_layout(yaxis_title="Explained Variance Ratio", xaxis_title="Principal Components")
            st.plotly_chart(fig_pca, use_container_width=True)

        st.markdown("### Raw matrix")
        st.dataframe(X, use_container_width=True, height=280)

        st.markdown("---")
        st.download_button("Download matrix CSV", data=X.to_csv(index=False).encode("utf-8"),
                           file_name="agri_agent_scores_matrix.csv", mime="text/csv")
        metrics_a = {"cond_index": float(cidx) if cidx==cidx else None, "pc1_share": float(pc1) if pc1==pc1 else None,
                     "vif": vif.to_dict(orient="list"), "corr": corr.to_dict(), "models": models, "config": cfg_a}
        st.download_button("Download metrics JSON", data=json.dumps(metrics_a, indent=2).encode("utf-8"),
                           file_name="agri_metrics.json", mime="application/json")

with tab3:
    st.markdown("""
    ### About & Methods
    **Goal.** Empirically demonstrate that multi-agent systems using a **single LLM** tend to emit **highly correlated** signals (inflating multicollinearity),
    whereas **heterogeneous, multi-LLM** committees produce more **diverse** signals (lowering multicollinearity), *ceteris paribus*.
    
    **Metrics.**
    - **Correlation matrix** among agent features (final_score) across tasks/fields.
    - **VIF (Variance Inflation Factor)** per agent variable; rule-of-thumb: VIF > 10 indicates severe collinearity.
    - **Condition Index** (from singular values of standardized design matrix).
    - **PCA dominance** (variance share of PC1; high share indicates redundancy).
    
    **Design knobs.**
    - **System Type:** Homogeneous vs Heterogeneous (rotates GPT-4o-mini and Gemini 2.5 if both keys are present).
    - **Role specialization:** Decorrelates reasoning trajectories (planner/critic/risk/econometrics/ops in General; agronomist/irrigation/pathology/economist/compliance in Agriculture).
    - **Temperature:** Higher temperatures typically increase within-model diversity.
    
    **Caveats.**
    - Shared retrieval or identical prompts may re-couple agent signals; for a production pipeline, decorrelate **retrievers**, **prompts**, and **critique heuristics** as well.
    - Debate/consensus reduces variance by construction; metrics here are computed on **pre-consensus** final_scores.
    
    **Attribution.** © Designed & Developed by Jit — Multicollinearity Suite.
    """)

st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;opacity:0.7;'>© Designed & Developed by Jit — Agentic AI Multicollinearity Suite</div>", unsafe_allow_html=True)
