# Patch the suite to handle statsmodels returning either ndarray or DataFrame for conf_int().
# We'll coerce with np.asarray(...) and remove .values usage.

# agentic_ai_multicollinearity_suite_v2_1.py
# ------------------------------------------------------------------
# Hotfix: robust conf_int() handling across statsmodels versions (ndarray vs DataFrame).
# Also adds extra guards for small-sample OLS to avoid crashes.
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
import statsmodels.api as sm

# ===== Optional SDKs =====
HAS_OPENAI = True
try:
    from openai import OpenAI
except Exception:
    HAS_OPENAI = False

HAS_GEMINI = True
try:
    import GEMINI.generativeai as genai
except Exception:
    HAS_GEMINI = False

st.set_page_config(page_title="Agentic AI Multicollinearity Suite v2.1", page_icon="🧮", layout="wide")

HEADER = """
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.8rem;">
  <div>
    <h1 style="margin:0;">Agentic AI Multicollinearity Suite — v2.1</h1>
    <p style="margin:0;color:#666;">Homogeneous vs Heterogeneous LLM Committees — Regression, VIF, Correlation, PCA</p>
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
    export GEMINI_API_KEY="AIza-..."
    ```
    **Install**
    ```bash
    pip install streamlit openai GEMINI-generativeai plotly statsmodels scikit-learn pandas numpy
    ```
    **Run**
    ```bash
    streamlit run agentic_ai_multicollinearity_suite_v2_1.py
    ```
    """)

def hash_key(obj: dict) -> str:
    try: s = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    except Exception: s = str(obj)
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def parse_json_block(txt: str, keys: List[str]) -> Dict[str, float]:
    if not txt: return {}
    start = txt.find('{'); end = txt.rfind('}')
    frag = txt[start:end+1] if (start!=-1 and end!=-1 and end>start) else ""
    if frag:
        try:
            data = json.loads(frag)
            out = {}
            for k in keys:
                v = data.get(k, None)
                try: out[k] = float(v) if v is not None else None
                except Exception: out[k] = None
            return out
        except Exception:
            pass
    nums = [float(n) for n in re.findall(r'(?<!\\d)(\\d{1,3}(?:\\.\\d+)?)(?!\\d)', txt)]
    return {k: (nums[i] if i < len(nums) else None) for i,k in enumerate(keys)}

def safe_vif(df: pd.DataFrame) -> pd.DataFrame:
    X = df.dropna()
    if X.shape[0] < 5 or X.shape[1] < 2:
        return pd.DataFrame({"feature": X.columns, "VIF": [np.nan]*X.shape[1]})
    vals = []
    for i in range(X.shape[1]):
        try: vals.append(variance_inflation_factor(X.values, i))
        except Exception: vals.append(np.nan)
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

def build_openai():
    if not HAS_OPENAI: return None
    try: return OpenAI()
    except Exception: return None

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
    except Exception:
        return None

def call_gemini_json(system_prompt: str, user_prompt: str, temperature: float=0.6, max_tokens: int=320) -> str:
    if not HAS_GEMINI: return None
    try:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel("gemini-2.5-pro")
        resp = model.generate_content(
            [system_prompt, user_prompt],
            generation_config={"temperature": float(temperature), "max_output_tokens": max_tokens}
        )
        return getattr(resp, "text", None)
    except Exception:
        return None

def assign_models(n_agents: int, heterogeneous: bool, force_provider: str=None) -> List[str]:
    providers = []
    if os.getenv("OPENAI_API_KEY"): providers.append("openai")
    if os.getenv("GEMINI_API_KEY"): providers.append("gemini")
    if force_provider in {"openai","gemini"}:
        providers = [force_provider] if force_provider in providers else providers
    if not providers: providers = ["openai"]
    if not heterogeneous or len(providers) == 1:
        return [providers[0]] * n_agents
    return [providers[i % len(providers)] for i in range(n_agents)]

def provider_label(provider: str) -> str:
    return "GPT-4o-mini (OpenAI)" if provider=="openai" else "Gemini 2.5 (GEMINI)"

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
    use_roles = bool(cfg["use_roles"])
    t_openai = float(cfg["t_openai"]); t_gemini = float(cfg["t_gemini"])
    prompts: List[str] = cfg["prompts"]
    force_provider = cfg.get("force_provider", None)

    providers = assign_models(n_agents, heterogeneous, force_provider=force_provider)
    oa = build_openai()

    rows, meta = [], []
    for p in prompts:
        a_scores, a_meta = [], []
        for a in range(n_agents):
            prov = providers[a]
            role_line = f"\nROLE: {GEN_ROLES[a]}" if (use_roles and a < len(GEN_ROLES)) else ""
            uprompt = f"Task: {p}\n{role_line}\nSchema: {GEN_SCHEMA}"
            txt = call_openai_json(oa, GEN_SYSTEM, uprompt, temperature=t_openai) if prov=="openai" \
                  else call_gemini_json(GEN_SYSTEM, uprompt, temperature=t_gemini)
            if txt is None:
                txt = '{"policy":60,"efficiency":60,"risk":40,"feasibility":55,"evidence":50,"final_score":55}'
            parsed = parse_json_block(txt, GEN_KEYS)
            a_scores.append(parsed.get("final_score", None))
            a_meta.append({"provider": prov, "raw": txt, **parsed})
        rows.append(a_scores); meta.append(a_meta)

    X = pd.DataFrame(rows, columns=[f"Agent{i+1}" for i in range(n_agents)])
    corr = X.corr(numeric_only=True)
    vif = safe_vif(X); cidx = condition_index(X); pc1 = pca_first_share(X)

    # Regression target (synthetic) with guards
    if X.shape[0] >= (X.shape[1] + 2):
        y = X.mean(axis=1).values + np.random.normal(0, 1.0, size=X.shape[0])
        X_ols = sm.add_constant(X.values)
        model = sm.OLS(y, X_ols, hasconst=True).fit()
        params = model.params[1:]
        conf = model.conf_int()
        conf_arr = np.asarray(conf)
        if conf_arr.ndim == 1:
            conf_arr = conf_arr.reshape(-1, 2)
        conf_arr = conf_arr[1:]
        se = (params - conf_arr[:,0]) / 1.96
        reg_df = pd.DataFrame({"Agent": [f"Agent{i+1}" for i in range(n_agents)], "coef": params, "se": se})
        reg_summary = model.summary().as_text()
    else:
        reg_df = pd.DataFrame({"Agent": [f"Agent{i+1}" for i in range(n_agents)], "coef": [np.nan]*n_agents, "se": [np.nan]*n_agents})
        reg_summary = "Not enough observations for OLS with all agents. Increase prompts."

    return {"X":X, "corr":corr, "vif":vif, "cond":cidx, "pc1":pc1, "providers":providers,
            "meta":meta, "reg_df": reg_df, "reg_summary": reg_summary}

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
    force_provider = cfg.get("force_provider", None)

    providers = assign_models(n_agents, heterogeneous, force_provider=force_provider)
    oa = build_openai()

    rows, raw = [], []
    for i in range(n_fields):
        card = make_field(seed+i if seed else None)
        a_scores = []; a_json = []
        for a in range(n_agents):
            prov = providers[a]
            role_line = f"\nROLE: {AG_ROLES[a]}" if (use_roles and a < len(AG_ROLES)) else ""
            uprompt = field_prompt(card) + role_line + "\nSchema: " + AG_SCHEMA
            txt = call_openai_json(oa, AG_SYSTEM, uprompt, temperature=t_openai) if prov=="openai" \
                  else call_gemini_json(AG_SYSTEM, uprompt, temperature=t_gemini)
            if txt is None:
                txt = '{"irrigation_mm":10,"nitrogen_kg":20,"pest_risk":42,"yield_gain":50,"water_stress":40,"final_score":57}'
            parsed = parse_json_block(txt, AG_KEYS)
            a_scores.append(parsed.get("final_score", None))
            a_json.append({"agent": f"Agent{a+1}","provider":prov,"out":parsed})
        rows.append(a_scores); raw.append({"field":card, "agents":a_json})

    X = pd.DataFrame(rows, columns=[f"Agent{i+1}" for i in range(n_agents)])
    corr = X.corr(numeric_only=True)
    vif = safe_vif(X); cidx = condition_index(X); pc1 = pca_first_share(X)

    # Regression on yield proxy with guards
    if X.shape[0] >= (X.shape[1] + 2):
        yield_proxy = []
        for rec in raw:
            f = rec["field"]
            stress = max(0.0, (f["ET0_mm_day"]*10 - f["soil_moisture_pct"]) - 0.2*f["rain_last_7d_mm"])
            yv = 20 + (f["ndvi"]*100)*0.8 - 0.3*stress + np.random.normal(0, 2.0)
            yield_proxy.append(yv)
        y = np.array(yield_proxy)

        X_ols = sm.add_constant(X.values)
        model = sm.OLS(y, X_ols, hasconst=True).fit()
        params = model.params[1:]
        conf = model.conf_int()
        conf_arr = np.asarray(conf)
        if conf_arr.ndim == 1:
            conf_arr = conf_arr.reshape(-1, 2)
        conf_arr = conf_arr[1:]
        se = (params - conf_arr[:,0]) / 1.96
        reg_df = pd.DataFrame({"Agent": [f"Agent{i+1}" for i in range(n_agents)], "coef": params, "se": se})
        reg_summary = model.summary().as_text()
    else:
        reg_df = pd.DataFrame({"Agent": [f"Agent{i+1}" for i in range(n_agents)], "coef": [np.nan]*n_agents, "se": [np.nan]*n_agents})
        reg_summary = "Not enough field snapshots for OLS with all agents. Increase 'Number of field snapshots'."

    return {"X":X, "corr":corr, "vif":vif, "cond":cidx, "pc1":pc1,
            "providers":providers, "raw":raw, "reg_df": reg_df, "reg_summary": reg_summary}

# ===== Tabs =====
tab1, tab2, tab3 = st.tabs(["General / Policy Lab", "Agriculture Lab", "About & Methods"])

# Provider status & test
st.sidebar.subheader("Provider Status")
st.sidebar.write(f"OpenAI key detected: {'✅' if os.getenv('OPENAI_API_KEY') else '❌'}")
st.sidebar.write(f"Gemini key detected: {'✅' if os.getenv('GEMINI_API_KEY') else '❌'}")

if HAS_GEMINI and os.getenv("GEMINI_API_KEY"):
    if st.sidebar.button("Test Gemini call"):
        try:
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            model = genai.GenerativeModel("gemini-2.5-pro")
            resp = model.generate_content("Reply with JSON: {\"ok\": true}")
            st.sidebar.success(f"Gemini responded: {getattr(resp, 'text', '')[:60]}")
        except Exception as e:
            st.sidebar.error(f"Gemini error: {e}")

with tab1:
    st.subheader("General / Policy Multicollinearity Lab")
    left, right = st.columns([0.6, 0.4])
    with left:
        mode = st.radio("System Type", ["Homogeneous: single LLM (multi-agent)", "Heterogeneous: multi-LLM committee"], index=1, key="g_mode")
        heterogeneous = mode.startswith("Heterogeneous")
        n_agents = st.slider("Agents", 2, 6, 3, 1, key="g_agents")
        use_roles = st.checkbox("Role specialization", value=heterogeneous, key="g_roles")
        t_openai = st.slider("OpenAI GPT-4o-mini temperature", 0.0, 1.5, 0.7, 0.1, key="g_temp_oa")
        t_gemini = st.slider("Gemini 2.5 temperature", 0.0, 1.5, 0.7, 0.1, key="g_temp_ge")
        force_provider = st.selectbox("Force provider (debug)", ["auto (alternate when hetero)", "openai only", "gemini only"], index=0, key="g_force")
        fp = None if force_provider.startswith("auto") else ("openai" if "openai" in force_provider else "gemini")
    with right:
        st.markdown("**Prompts (one per line)**")
        default_text = "\n".join(GEN_DEFAULT_PROMPTS)
        ptext = st.text_area(" ", value=default_text, height=220, key="g_prompts")
        prompts = [p.strip() for p in ptext.splitlines() if p.strip()]
        if len(prompts) < 3:
            st.warning("Add at least 3 prompts for stable VIF/PC1 estimates.")
        run_g = st.button("▶️ Run General/Policy", use_container_width=True)

    if run_g:
        cfg = {"n_agents":n_agents,"heterogeneous":heterogeneous,"use_roles":use_roles,
               "t_openai":t_openai,"t_gemini":t_gemini,"prompts":prompts, "force_provider": fp}
        with st.spinner("Running committee and computing diagnostics..."):
            res = run_general_lab(cfg)

        X, corr, vif, cidx, pc1 = res["X"], res["corr"], res["vif"], res["cond"], res["pc1"]
        providers, reg_df, reg_summary = res["providers"], res["reg_df"], res["reg_summary"]

        with st.expander("Agent roster (provider per agent)", expanded=True):
            badges = [f"Agent{i+1}: **{provider_label(p)}**" for i,p in enumerate(providers)]
            st.write(", ".join(badges))

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Observations (prompts)", f"{X.shape[0]}"); c2.metric("Agents", f"{X.shape[1]}")
        c3.metric("Condition Index", f"{cidx:0.2f}" if np.isfinite(cidx) else "—")
        c4.metric("PCA Var(PC1)", f"{pc1*100:0.1f}%" if not np.isnan(pc1) else "—")

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

        st.markdown("### OLS Regression on Synthetic Target (coef ±95% CI)")
        fig_coef = go.Figure()
        fig_coef.add_trace(go.Bar(
            x=reg_df["Agent"], y=reg_df["coef"],
            error_y=dict(type="data", array=reg_df["se"]*1.96, visible=True)
        ))
        fig_coef.update_layout(yaxis_title="Coefficient", xaxis_title="Agent variable",
                               title="Coefficient instability increases under multicollinearity")
        st.plotly_chart(fig_coef, use_container_width=True)
        with st.expander("Regression summary (diagnostic)", expanded=False):
            st.text(reg_summary)

        st.markdown("### Raw matrix")
        st.dataframe(X, use_container_width=True, height=280)

with tab2:
    st.subheader("Agriculture Multicollinearity Lab")
    left, right = st.columns([0.6, 0.4])
    with left:
        mode_a = st.radio("System Type", ["Homogeneous: single LLM", "Heterogeneous: multi-LLM"], index=1, key="a_mode")
        heterogeneous_a = mode_a.startswith("Heterogeneous")
        n_agents_a = st.slider("Agents", 2, 6, 3, 1, key="a_agents")
        use_roles_a = st.checkbox("Role specialization", value=True, key="a_roles")
        t_openai_a = st.slider("OpenAI GPT-4o-mini temperature", 0.0, 1.5, 0.6, 0.1, key="a_temp_oa")
        t_gemini_a = st.slider("Gemini 2.5 temperature", 0.0, 1.5, 0.6, 0.1, key="a_temp_ge")
        force_provider_a = st.selectbox("Force provider (debug)", ["auto (alternate when hetero)", "openai only", "gemini only"], index=0, key="a_force")
        fp_a = None if force_provider_a.startswith("auto") else ("openai" if "openai" in force_provider_a else "gemini")
    with right:
        n_fields = st.slider("Number of field snapshots", 5, 40, 12, 1, key="a_fields")
        seed = st.number_input("Random seed (optional)", value=0, step=1, key="a_seed")
        run_a = st.button("▶️ Run Agriculture", use_container_width=True)

    if run_a:
        cfg_a = {"n_agents": n_agents_a, "heterogeneous":heterogeneous_a, "use_roles": use_roles_a,
                 "t_openai": t_openai_a, "t_gemini": t_gemini_a, "n_fields": n_fields, "seed": seed,
                 "force_provider": fp_a}
        with st.spinner("Running agri committee and computing diagnostics..."):
            res = run_agri_lab(cfg_a)

        X, corr, vif, cidx, pc1 = res["X"], res["corr"], res["vif"], res["cond"], res["pc1"]
        providers, reg_df, reg_summary = res["providers"], res["reg_df"], res["reg_summary"]

        with st.expander("Agent roster (provider per agent)", expanded=True):
            badges = [f"Agent{i+1}: **{provider_label(p)}**" for i,p in enumerate(providers)]
            st.write(", ".join(badges))

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Field snapshots", f"{X.shape[0]}"); c2.metric("Agents", f"{X.shape[1]}")
        c3.metric("Condition Index", f"{cidx:0.2f}" if np.isfinite(cidx) else "—")
        c4.metric("PCA Var(PC1)", f"{pc1*100:0.1f}%" if not np.isnan(pc1) else "—")

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

        st.markdown("### OLS Regression on Yield Proxy (coef ±95% CI)")
        fig_coef = go.Figure()
        fig_coef.add_trace(go.Bar(
            x=reg_df["Agent"], y=reg_df["coef"],
            error_y=dict(type="data", array=reg_df["se"]*1.96, visible=True)
        ))
        fig_coef.update_layout(yaxis_title="Coefficient", xaxis_title="Agent variable",
                               title="Coefficient instability increases under multicollinearity")
        st.plotly_chart(fig_coef, use_container_width=True)
        with st.expander("Regression summary (diagnostic)", expanded=False):
            st.text(reg_summary)

        st.markdown("### Raw matrix")
        st.dataframe(X, use_container_width=True, height=280)

with tab3:
    st.markdown("""
    ### About & Methods
    **Hotfix v2.1**: robust handling of `conf_int()` across statsmodels versions (ndarray vs DataFrame) and extra guards for small-sample regression.
    The rest of the analysis is unchanged: correlation heatmap, VIF bar chart, PCA PC1 share, and regression CIs to visualize multicollinearity effects.
    """)

st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;opacity:0.7;'>© Designed & Developed by Jit — Agentic AI Multicollinearity Suite v2.1</div>", unsafe_allow_html=True)

