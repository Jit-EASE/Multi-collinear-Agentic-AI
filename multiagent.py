# agentic_ai_multicollinearity_suite_v2_5.py
# ------------------------------------------------------------------
# v2.5 — Realistic targets, VIF pruning, PCA fallback, RidgeCV regularization,
#        optional Late-Fusion (per-agent base learners + reliability-weighted ensemble),
#        corrected Gemini import, unique Streamlit keys.
#        Designed & Developed for Jit
# ------------------------------------------------------------------

import os, re, json, hashlib, textwrap, random
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

import streamlit as st
import plotly.graph_objects as go

from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.metrics import r2_score

# ---------------- LLM Providers ----------------
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

# ---------------- Streamlit Setup ----------------
st.set_page_config(page_title="Agentic AI Multicollinearity Suite v2.5", page_icon="🧮", layout="wide")
st.markdown("### Agentic AI Multicollinearity Suite — v2.5")

# ---------------- Utilities ----------------
def hash_key(obj: dict) -> str:
    try:
        s = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    except Exception:
        s = str(obj)
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def parse_json_block(txt: str, keys: List[str]) -> Dict[str, float]:
    """
    Extract a JSON object with known keys; if JSON fails, fallback to numeric scraping.
    """
    if not txt:
        return {}
    start = txt.find('{'); end = txt.rfind('}')
    frag = txt[start:end+1] if (start != -1 and end != -1 and end > start) else ""
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
    # Fallback: scrape numbers in order
    nums = [float(n) for n in re.findall(r'(?<!\d)(\d{1,3}(?:\.\d+)?)(?!\d)', txt)]
    return {k: (nums[i] if i < len(nums) else None) for i, k in enumerate(keys)}

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
    if X.shape[0] < 2 or X.shape[1] < 2:
        return np.nan
    Z = StandardScaler().fit_transform(X.values)
    u, s, vh = np.linalg.svd(Z, full_matrices=False)
    return float((s.max()/s.min()) if s.min() > 0 else np.inf)

def pca_first_share(df: pd.DataFrame) -> float:
    X = df.dropna()
    if X.shape[0] < 2 or X.shape[1] < 2:
        return np.nan
    Z = StandardScaler().fit_transform(X.values)
    p = PCA().fit(Z)
    return float(p.explained_variance_ratio_[0])

def build_openai():
    if not HAS_OPENAI:
        return None
    try:
        return OpenAI()
    except Exception:
        return None

def call_openai_json(client, system_prompt: str, user_prompt: str, temperature: float=0.6, max_tokens: int=320) -> str:
    if not client:
        return None
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_prompt}],
            temperature=float(temperature),
            max_tokens=int(max_tokens)
        )
        return resp.choices[0].message.content
    except Exception:
        return None

def call_gemini_json(system_prompt: str, user_prompt: str, temperature: float=0.6, max_tokens: int=320) -> str:
    if not HAS_GEMINI:
        return None
    try:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel("gemini-2.5-pro")
        resp = model.generate_content(
            [system_prompt, user_prompt],
            generation_config={"temperature": float(temperature), "max_output_tokens": int(max_tokens)}
        )
        return getattr(resp, "text", None)
    except Exception:
        return None

def assign_models(n_agents: int, heterogeneous: bool, force_provider: str=None) -> List[str]:
    providers = []
    if os.getenv("OPENAI_API_KEY"):
        providers.append("openai")
    if os.getenv("GEMINI_API_KEY"):
        providers.append("gemini")
    if force_provider in {"openai", "gemini"}:
        providers = [force_provider] if force_provider in providers else providers
    if not providers:
        providers = ["openai"]  # safe default
    if not heterogeneous or len(providers) == 1:
        return [providers[0]] * n_agents
    return [providers[i % len(providers)] for i in range(n_agents)]

def provider_label(provider: str) -> str:
    return "GPT-4o-mini (OpenAI)" if provider == "openai" else "Gemini 2.5 (Google)"

# ---------------- Narrative Explanations ----------------
def explain_corr(corr: pd.DataFrame) -> str:
    try:
        M = corr.copy()
        if M.shape[0] < 2:
            return "Not enough agents to compute correlations."
        mask = np.triu(np.ones(M.shape, dtype=bool), k=1)
        vals = M.where(mask).stack()
        if vals.empty:
            return "Correlation matrix is empty."
        max_pair = vals.abs().idxmax()
        max_val = float(vals.loc[max_pair])
        mean_abs = float(vals.abs().mean())
        txt = []
        txt.append(f"Average |correlation| across agent pairs: **{mean_abs:.2f}**. ")
        txt.append(f"Strongest pair: **{max_pair[0]}–{max_pair[1]} = {max_val:.2f}**. ")
        if mean_abs >= 0.6 or abs(max_val) >= 0.9:
            txt.append("This indicates **high collinearity**; agents are delivering very similar signals.")
        elif mean_abs >= 0.4:
            txt.append("This suggests **moderate collinearity**; there is meaningful redundancy.")
        else:
            txt.append("This looks **diverse**; agent signals are relatively independent.")
        return "".join(txt)
    except Exception:
        return "Correlation explanation unavailable."

def explain_vif(vif_df: pd.DataFrame) -> str:
    try:
        df = vif_df.dropna()
        if df.empty:
            return "VIF not available (insufficient rows/agents)."
        severe = df[df["VIF"] >= 10].shape[0]
        moderate = df[(df["VIF"] >= 5) & (df["VIF"] < 10)].shape[0]
        top = df.iloc[df["VIF"].idxmax()]
        txt = [f"Max VIF is **{top['VIF']:.1f}** for **{top['feature']}**. "]
        if severe > 0:
            txt.append(f"**{severe}** variables exceed **10** (severe multicollinearity). ")
        elif moderate > 0:
            txt.append(f"**{moderate}** variables are in **5–10** (moderate multicollinearity). ")
        else:
            txt.append("No VIFs exceed **5** — collinearity appears limited. ")
        txt.append("High VIF inflates standard errors and destabilizes coefficients.")
        return "".join(txt)
    except Exception:
        return "VIF explanation unavailable."

def explain_overall(cidx: float, pc1: float, providers: list) -> str:
    prov_kind = "heterogeneous (multi-LLM)" if len(set(providers)) > 1 else "homogeneous (single LLM)"
    pieces = [f"System detected as **{prov_kind}**. "]
    if np.isfinite(cidx):
        if cidx > 30:
            pieces.append(f"Condition Index **{cidx:.1f}** → **serious multicollinearity**. ")
        elif cidx > 10:
            pieces.append(f"Condition Index **{cidx:.1f}** → **moderate multicollinearity**. ")
        else:
            pieces.append(f"Condition Index **{cidx:.1f}** → **low multicollinearity**. ")
    if not np.isnan(pc1):
        if pc1 >= 0.6:
            pieces.append(f"PC1 explains **{pc1*100:.1f}%** of variance → **redundant signals** dominate. ")
        elif pc1 >= 0.4:
            pieces.append(f"PC1 explains **{pc1*100:.1f}%** → **some redundancy**. ")
        else:
            pieces.append(f"PC1 explains **{pc1*100:.1f}%** → **diverse signals**. ")
    pieces.append("Mitigations: VIF pruning, PCA fallback, regularized regression, or late-fusion ensembles.")
    return "".join(pieces)

# ---------------- Collinearity Controls & Models ----------------
def drop_high_vif(df: pd.DataFrame, thresh: float = 10.0) -> Tuple[pd.DataFrame, List[str]]:
    dropped = []
    if df.shape[1] < 2:
        return df, dropped
    while True:
        vif_df = safe_vif(df)
        if vif_df["VIF"].isna().all():
            break
        max_vif = vif_df["VIF"].max()
        if max_vif > thresh:
            drop_feat = vif_df.loc[vif_df["VIF"].idxmax(), "feature"]
            df = df.drop(columns=[drop_feat])
            dropped.append(drop_feat)
            if df.shape[1] < 2:
                break
        else:
            break
    return df, dropped

def apply_pca_if_needed(df: pd.DataFrame, var_threshold: float = 0.90) -> Tuple[pd.DataFrame, int]:
    if df.shape[1] < 2:
        return df.copy(), df.shape[1]
    Z = StandardScaler().fit_transform(df.values)
    p = PCA().fit(Z)
    cumvar = np.cumsum(p.explained_variance_ratio_)
    ncomp = int(np.searchsorted(cumvar, var_threshold) + 1)
    Z_pca = p.transform(Z)[:, :ncomp]
    cols = [f"PC{i+1}" for i in range(ncomp)]
    return pd.DataFrame(Z_pca, columns=cols), ncomp

def run_regression_pipeline(X: pd.DataFrame, y: np.ndarray) -> Tuple[float, pd.DataFrame, str, pd.DataFrame, float, float]:
    """
    Early-fusion pipeline:
    - VIF pruning
    - PCA fallback if Condition Index > 30
    - RidgeCV
    Returns: r2, reg_df(coefs), method_desc, vif_df(before drop), cond_idx(before), pc1(before)
    """
    if X.shape[0] < (X.shape[1] + 2):
        return np.nan, pd.DataFrame(), "Insufficient observations for regression.", pd.DataFrame(), np.nan, np.nan

    # Diagnostics BEFORE pruning
    vif_before = safe_vif(X)
    cond_before = condition_index(X)
    pc1_before = pca_first_share(X)

    # Step 1: VIF prune
    X_vif, dropped = drop_high_vif(X, thresh=10.0)

    # Step 2: PCA fallback if still highly collinear
    method = ""
    if condition_index(X_vif) > 30 and X_vif.shape[1] >= 2:
        X_final, ncomp = apply_pca_if_needed(X_vif, var_threshold=0.90)
        method = f"PCA applied ({ncomp} components) after VIF drop={dropped}"
    else:
        X_final = X_vif
        method = f"VIF pruned: dropped={dropped if dropped else 'None'}"

    # Step 3: RidgeCV (regularized)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 20), cv=5))
    ])
    model.fit(X_final, y)
    r2 = model.score(X_final, y)

    # Coefs (note: for PCA features, coefficients are on PCs)
    coefs = model.named_steps["ridge"].coef_
    labels = list(X_final.columns)
    reg_df = pd.DataFrame({"Feature": labels, "coef": coefs})

    return r2, reg_df, method, vif_before, cond_before, pc1_before

def reliability_weight(r2cv: float) -> float:
    # Only reward positive skill; square to accentuate separation
    return max(r2cv, 0.0) ** 2

def late_fusion_ensemble(agent_feature_blocks: List[pd.DataFrame], y: np.ndarray) -> Tuple[float, pd.DataFrame]:
    """
    Per-agent models -> cross-validated predictions -> reliability-weighted ensemble.
    agent_feature_blocks: list of DataFrames, one per agent (columns = that agent's schema keys)
    Returns: ensemble_r2, summary_df with agent r2cv and weights
    """
    if len(agent_feature_blocks) == 0:
        return np.nan, pd.DataFrame()

    kf = KFold(n_splits=min(5, len(y)), shuffle=True, random_state=42)
    agent_results = []
    preds_matrix = []

    for i, F in enumerate(agent_feature_blocks):
        # Guard for degenerate design
        if F.shape[1] == 0 or F.shape[0] < (F.shape[1] + 2):
            preds_matrix.append(np.full_like(y, fill_value=np.nan, dtype=float))
            agent_results.append({"Agent": f"Agent{i+1}", "r2cv": np.nan, "weight": 0.0})
            continue

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 20), cv=5))
        ])

        try:
            r2cv_scores = cross_val_score(pipe, F.values, y, scoring="r2", cv=kf)
            r2cv = float(np.nanmean(r2cv_scores))
            yhat = cross_val_predict(pipe, F.values, y, cv=kf)
        except Exception:
            r2cv = np.nan
            yhat = np.full_like(y, fill_value=np.nan, dtype=float)

        preds_matrix.append(yhat)
        agent_results.append({"Agent": f"Agent{i+1}", "r2cv": r2cv, "weight": reliability_weight(r2cv)})

    preds_mat = np.vstack(preds_matrix)  # shape: n_agents x n_samples
    weights = np.array([a["weight"] for a in agent_results], dtype=float)

    if np.all(np.isnan(preds_mat)) or weights.sum() == 0:
        return np.nan, pd.DataFrame(agent_results)

    # Weighted ensemble (ignore NaNs per-sample)
    yhat_ens = np.zeros_like(y, dtype=float)
    for t in range(len(y)):
        col = preds_mat[:, t]
        mask = np.isfinite(col)
        if not mask.any():
            yhat_ens[t] = np.nan
        else:
            w = weights[mask]
            if w.sum() == 0:
                yhat_ens[t] = np.nan
            else:
                yhat_ens[t] = np.dot(w, col[mask]) / w.sum()

    mask_valid = np.isfinite(yhat_ens)
    if mask_valid.sum() < 3:
        ensemble_r2 = np.nan
    else:
        ensemble_r2 = r2_score(y[mask_valid], yhat_ens[mask_valid])

    return ensemble_r2, pd.DataFrame(agent_results)

# ---------------- Labs: Schemas & Prompts ----------------
# General / Policy Lab
GEN_KEYS = ["policy","efficiency","risk","feasibility","evidence","final_score"]
GEN_SCHEMA = ("Return ONLY a JSON object with keys: policy, efficiency, risk, "
              "feasibility, evidence, final_score. Each 0-100 integer.")
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
    "Score (0-100) System-of-systems interoperability in agri-food (2025–2030).",
    "Score (0-100) Socio-economic benefit of stress-aware worker support programs in agri-processing."
]
GEN_ROLES = [
    "Planner: optimize long-term policy outcomes and resource allocation.",
    "Critic: maximize factual accuracy, identify unsupported claims.",
    "Risk Analyst: evaluate technical, regulatory, and adoption risks.",
    "Econometrician: focus on quantifiable indicators and uncertainty.",
    "Operations Lead: emphasize implementation constraints and ROI."
]

# Agriculture Lab
AG_KEYS = ["irrigation_mm","nitrogen_kg","pest_risk","yield_gain","water_stress","final_score"]
AG_SCHEMA = ("Return ONLY a JSON object with keys: irrigation_mm (0-60), nitrogen_kg (0-60), "
             "pest_risk (0-100), yield_gain (0-100), water_stress (0-100), final_score (0-100).")
AG_SYSTEM = "You are an expert agronomy decision agent. Be concise, numeric, schema-only. " + AG_SCHEMA
AG_ROLES = [
    "Agronomist: crop physiology and balanced recommendations.",
    "Irrigation Specialist: schedule irrigation and manage moisture.",
    "Plant Pathologist: estimate pest/disease risk from conditions.",
    "Agricultural Economist: trade-offs and ROI on inputs.",
    "Compliance Officer: nitrates directive & runoff constraints."
]
SOIL_TYPES = ["Sandy loam", "Loam", "Clay loam", "Peat-influenced loam"]
PH_RANGE = (5.6, 7.2); SM_RANGE = (8, 42); ET0_RANGE = (2.0, 6.0); RAIN_7D = (0, 60)
TEMP_C = (10, 27); NDVI = (0.45, 0.85); CANOPY_WET = [False, True]
P_STAGE = ["tillering","stem elongation","booting","heading","grain fill"]
PEST_PRESS = ["low","moderate","high"]

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
    return f"""
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
""".strip()

# ---------------- Core Runners ----------------
def run_general_lab(cfg: dict) -> dict:
    n_agents = int(cfg["n_agents"]); heterogeneous = bool(cfg["heterogeneous"])
    use_roles = bool(cfg["use_roles"])
    t_openai = float(cfg["t_openai"]); t_gemini = float(cfg["t_gemini"])
    prompts: List[str] = cfg["prompts"]
    force_provider = cfg.get("force_provider", None)
    fusion_mode = cfg.get("fusion_mode", "Hybrid")
    use_all_feats = bool(cfg.get("use_all_feats", False))
    seed = int(cfg.get("seed", 0))

    if seed:
        random.seed(seed); np.random.seed(seed)

    providers = assign_models(n_agents, heterogeneous, force_provider=force_provider)
    oa = build_openai()

    rows_full = []   # all keys per agent (flattened later)
    agent_blocks = []  # per-agent DataFrames for late fusion
    # init blocks
    for _ in range(n_agents):
        agent_blocks.append([])

    # Collect outputs
    for p in prompts:
        agent_row_full = []
        per_agent_vals = []
        for a in range(n_agents):
            prov = providers[a]
            role_line = f"\nROLE: {GEN_ROLES[a]}" if (use_roles and a < len(GEN_ROLES)) else ""
            uprompt = f"Task: {p}\n{role_line}\nSchema: {GEN_SCHEMA}"
            txt = call_openai_json(oa, GEN_SYSTEM, uprompt, temperature=t_openai) if prov == "openai" \
                  else call_gemini_json(GEN_SYSTEM, uprompt, temperature=t_gemini)
            if txt is None:
                txt = '{"policy":60,"efficiency":60,"risk":40,"feasibility":55,"evidence":50,"final_score":55}'
            parsed = parse_json_block(txt, GEN_KEYS)
            vals = [parsed.get(k, None) for k in GEN_KEYS]
            per_agent_vals.append(vals)

        # append per agent values to row_full
        for vals in per_agent_vals:
            agent_row_full.extend(vals)
        rows_full.append(agent_row_full)

    # Build feature matrices
    colnames = []
    for i in range(n_agents):
        for k in GEN_KEYS:
            colnames.append(f"A{i+1}_{k}")
    X_full = pd.DataFrame(rows_full, columns=colnames)

    # Early-fusion features
    if use_all_feats:
        X_ef = X_full.copy()
    else:
        # only final_score per agent
        X_ef = X_full[[f"A{i+1}_final_score" for i in range(n_agents)]].copy()

    # Late-fusion blocks: per-agent DataFrames with their own schema features
    agent_feature_blocks = []
    for i in range(n_agents):
        block = X_full[[f"A{i+1}_{k}" for k in GEN_KEYS]].copy()
        agent_feature_blocks.append(block)

    # Synthetic, *non-circular* target (independent of agent scores)
    # Draw a latent "difficulty" and some noise to emulate target variability
    diff = np.random.normal(0, 1, size=X_full.shape[0])
    y = 50 + 10*diff + np.random.normal(0, 8, size=X_full.shape[0])

    results = {"providers": providers, "X_full": X_full, "X_ef": X_ef, "y": y}

    # Early fusion
    r2_early, reg_df, method_used, vif_before, cond_before, pc1_before = run_regression_pipeline(X_ef, y)
    results.update({
        "r2_early": r2_early,
        "reg_df": reg_df,
        "method_used": method_used,
        "vif_before": vif_before,
        "cond_before": cond_before,
        "pc1_before": pc1_before
    })

    # Late fusion
    r2_late, lf_df = late_fusion_ensemble(agent_feature_blocks, y)
    results.update({"r2_late": r2_late, "lf_df": lf_df})

    # Best
    if fusion_mode == "Early":
        results["r2_best"] = r2_early
        results["best_mode"] = "Early Fusion"
    elif fusion_mode == "Late":
        results["r2_best"] = r2_late
        results["best_mode"] = "Late Fusion"
    else:
        # Hybrid: pick the better R2
        if (np.nan_to_num(r2_early, nan=-1) >= np.nan_to_num(r2_late, nan=-1)):
            results["r2_best"] = r2_early
            results["best_mode"] = "Early Fusion"
        else:
            results["r2_best"] = r2_late
            results["best_mode"] = "Late Fusion"

    return results

def run_agri_lab(cfg: dict) -> dict:
    n_agents = int(cfg["n_agents"]); heterogeneous = bool(cfg["heterogeneous"])
    use_roles = bool(cfg["use_roles"])
    t_openai = float(cfg["t_openai"]); t_gemini = float(cfg["t_gemini"])
    n_fields = int(cfg["n_fields"]); seed = int(cfg.get("seed", 0))
    force_provider = cfg.get("force_provider", None)
    fusion_mode = cfg.get("fusion_mode", "Hybrid")
    use_all_feats = bool(cfg.get("use_all_feats", True))

    if seed:
        random.seed(seed); np.random.seed(seed)

    providers = assign_models(n_agents, heterogeneous, force_provider=force_provider)
    oa = build_openai()

    rows_full = []
    agent_feature_blocks = [ [] for _ in range(n_agents) ]
    raw_fields = []

    for i in range(n_fields):
        card = make_field(seed + i if seed else None)
        raw_fields.append(card)
        per_agent_vals = []
        for a in range(n_agents):
            prov = providers[a]
            role_line = f"\nROLE: {AG_ROLES[a]}" if (use_roles and a < len(AG_ROLES)) else ""
            uprompt = field_prompt(card) + role_line + "\nSchema: " + AG_SCHEMA
            txt = call_openai_json(oa, AG_SYSTEM, uprompt, temperature=t_openai) if prov == "openai" \
                  else call_gemini_json(AG_SYSTEM, uprompt, temperature=t_gemini)
            if txt is None:
                txt = '{"irrigation_mm":10,"nitrogen_kg":20,"pest_risk":42,"yield_gain":50,"water_stress":40,"final_score":57}'
            parsed = parse_json_block(txt, AG_KEYS)
            vals = [parsed.get(k, None) for k in AG_KEYS]
            per_agent_vals.append(vals)

        row_full = []
        for vals in per_agent_vals:
            row_full.extend(vals)
        rows_full.append(row_full)

    # Build feature matrices
    colnames = []
    for i in range(n_agents):
        for k in AG_KEYS:
            colnames.append(f"A{i+1}_{k}")
    X_full = pd.DataFrame(rows_full, columns=colnames)

    # Early-fusion features
    if use_all_feats:
        X_ef = X_full.copy()
    else:
        X_ef = X_full[[f"A{i+1}_final_score" for i in range(n_agents)]].copy()

    # Late-fusion blocks
    agent_feature_blocks = []
    for i in range(n_agents):
        block = X_full[[f"A{i+1}_{k}" for k in AG_KEYS]].copy()
        agent_feature_blocks.append(block)

    # Realistic agronomic target (proxy)
    # NDVI -> positive yield; water stress (ET0*10 - soil_moisture - 0.2*rain) -> negative yield
    y_proxy = []
    for f in raw_fields:
        stress = max(0.0, (f["ET0_mm_day"]*10 - f["soil_moisture_pct"]) - 0.2*f["rain_last_7d_mm"])
        yv = 20 + (f["ndvi"]*100)*0.8 - 0.3*stress + np.random.normal(0, 2.0)
        y_proxy.append(yv)
    y = np.array(y_proxy)

    results = {"providers": providers, "X_full": X_full, "X_ef": X_ef, "y": y}

    # Early fusion
    r2_early, reg_df, method_used, vif_before, cond_before, pc1_before = run_regression_pipeline(X_ef, y)
    results.update({
        "r2_early": r2_early,
        "reg_df": reg_df,
        "method_used": method_used,
        "vif_before": vif_before,
        "cond_before": cond_before,
        "pc1_before": pc1_before
    })

    # Late fusion
    r2_late, lf_df = late_fusion_ensemble(agent_feature_blocks, y)
    results.update({"r2_late": r2_late, "lf_df": lf_df})

    # Best
    if fusion_mode == "Early":
        results["r2_best"] = r2_early
        results["best_mode"] = "Early Fusion"
    elif fusion_mode == "Late":
        results["r2_best"] = r2_late
        results["best_mode"] = "Late Fusion"
    else:
        if (np.nan_to_num(r2_early, nan=-1) >= np.nan_to_num(r2_late, nan=-1)):
            results["r2_best"] = r2_early
            results["best_mode"] = "Early Fusion"
        else:
            results["r2_best"] = r2_late
            results["best_mode"] = "Late Fusion"

    return results

# ---------------- UI ----------------
st.sidebar.subheader("Provider Status")
st.sidebar.write(f"OpenAI key detected: {'✅' if os.getenv('OPENAI_API_KEY') else '❌'}")
st.sidebar.write(f"Gemini key detected: {'✅' if os.getenv('GEMINI_API_KEY') else '❌'}")
if HAS_GEMINI and os.getenv("GEMINI_API_KEY"):
    if st.sidebar.button("Test Gemini call", key="sidebar_test_gemini_v25"):
        try:
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            model = genai.GenerativeModel("gemini-2.5-pro")
            resp = model.generate_content("Reply with JSON: {\"ok\": true}")
            st.sidebar.success(f"Gemini responded: {getattr(resp, 'text', '')[:60]}")
        except Exception as e:
            st.sidebar.error(f"Gemini error: {e}")

tab1, tab2 = st.tabs(["General / Policy Lab", "Agriculture Lab"])

# -------- General Tab --------
with tab1:
    left, right = st.columns([0.60, 0.40])
    with left:
        mode = st.radio("System Type (General)", ["Homogeneous (single LLM)", "Heterogeneous (multi-LLM)"],
                        index=1, key="g_radio_mode_v25")
        heterogeneous = mode.startswith("Heterogeneous")

        n_agents = st.slider("Agents (General)", 2, 6, 3, 1, key="g_slider_agents_v25")
        use_roles = st.checkbox("Role specialization (General)", value=heterogeneous, key="g_roles_v25")

        fusion_mode = st.selectbox("Fusion Strategy",
                                   ["Hybrid (pick best)", "Early", "Late"],
                                   index=0, key="g_fusion_v25")
        use_all_feats = st.checkbox("Use all schema features (not just final_score)", value=False, key="g_allfeats_v25")

        t_openai = st.slider("OpenAI GPT-4o-mini temperature", 0.0, 1.5, 0.7, 0.1, key="g_temp_oa_v25")
        t_gemini = st.slider("Gemini 2.5 temperature", 0.0, 1.5, 0.7, 0.1, key="g_temp_ge_v25")

        force_provider = st.selectbox("Force provider", ["auto", "openai only", "gemini only"], index=0, key="g_force_v25")
        fp = None if force_provider=="auto" else ("openai" if "openai" in force_provider else "gemini")

        seed = st.number_input("Random seed (General)", value=0, step=1, key="g_seed_v25")

    with right:
        default_text = "\n".join(GEN_DEFAULT_PROMPTS)
        ptext = st.text_area("Prompts (General) — one per line", value=default_text, height=220, key="g_prompts_v25")
        prompts = [p.strip() for p in ptext.splitlines() if p.strip()]
        run_g = st.button("▶️ Run General/Policy", use_container_width=True, key="g_run_v25")

    if run_g:
        cfg = {
            "n_agents": n_agents, "heterogeneous": heterogeneous, "use_roles": use_roles,
            "t_openai": t_openai, "t_gemini": t_gemini, "prompts": prompts, "force_provider": fp,
            "fusion_mode": {"Hybrid (pick best)":"Hybrid","Early":"Early","Late":"Late"}[fusion_mode],
            "use_all_feats": use_all_feats, "seed": seed
        }
        res = run_general_lab(cfg)

        providers = res["providers"]; X_ef = res["X_ef"]; y = res["y"]
        r2_early = res["r2_early"]; r2_late = res["r2_late"]
        best_mode = res["best_mode"]; r2_best = res["r2_best"]

        st.write("**Providers:**", ", ".join([provider_label(p) for p in providers]))
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Observations", f"{X_ef.shape[0]}")
        c2.metric("Agent Features", f"{X_ef.shape[1]}")
        c3.metric("Early R²", f"{r2_early:.3f}" if np.isfinite(r2_early) else "—")
        c4.metric("Late R²", f"{r2_late:.3f}" if np.isfinite(r2_late) else "—")
        c5.metric("Best", f"{best_mode} (R²={r2_best:.3f})" if np.isfinite(r2_best) else best_mode)

        # Diagnostics only meaningful for Early-fusion feature matrix
        st.markdown("### Correlation Heatmap (Early-Fusion Features)")
        if X_ef.shape[1] >= 2:
            corr = X_ef.corr(numeric_only=True)
            heat = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index,
                                             zmin=-1, zmax=1, colorscale="RdBu", reversescale=True))
            heat.update_layout(height=380, margin=dict(l=10,r=10,t=20,b=10))
            st.plotly_chart(heat, use_container_width=True, key="g_corr_plot_v25")
            st.info(explain_corr(corr))

        st.markdown("### VIF (Early-Fusion Features)")
        vif_df = res["vif_before"]
        if not vif_df.empty:
            bar = go.Figure(data=[go.Bar(x=vif_df["feature"], y=vif_df["VIF"])])
            bar.update_layout(height=320, margin=dict(l=10,r=10,t=20,b=10))
            st.plotly_chart(bar, use_container_width=True, key="g_vif_plot_v25")
            st.dataframe(vif_df, use_container_width=True, key="g_vif_df_v25")
            st.info(explain_vif(vif_df))

        st.markdown("### Model (Early-Fusion, RidgeCV)")
        reg_df = res["reg_df"]
        if not reg_df.empty:
            fig = go.Figure()
            fig.add_bar(x=reg_df["Feature"], y=reg_df["coef"])
            fig.update_layout(height=320, margin=dict(l=10,r=10,t=20,b=10))
            st.plotly_chart(fig, use_container_width=True, key="g_coef_plot_v25")
            st.caption(res["method_used"])
        else:
            st.write("Insufficient data for early-fusion regression.")

        st.markdown("### Overall diagnosis (Early-Fusion feature space)")
        st.warning(explain_overall(res["cond_before"], res["pc1_before"], providers))

        if res["lf_df"] is not None and not res["lf_df"].empty:
            st.markdown("### Late-Fusion Agent Reliability")
            st.dataframe(res["lf_df"], use_container_width=True, key="g_lf_df_v25")

        st.markdown("### Raw feature matrix (Early-Fusion)")
        st.dataframe(X_ef, use_container_width=True, height=260, key="g_raw_df_v25")

# -------- Agriculture Tab --------
with tab2:
    left, right = st.columns([0.60, 0.40])
    with left:
        mode_a = st.radio("System Type (Agriculture)", ["Homogeneous (single LLM)", "Heterogeneous (multi-LLM)"],
                          index=1, key="a_radio_mode_v25")
        heterogeneous_a = mode_a.startswith("Heterogeneous")

        n_agents_a = st.slider("Agents (Agriculture)", 2, 6, 3, 1, key="a_slider_agents_v25")
        use_roles_a = st.checkbox("Role specialization (Agriculture)", value=True, key="a_roles_v25")

        fusion_mode_a = st.selectbox("Fusion Strategy (Agriculture)",
                                     ["Hybrid (pick best)", "Early", "Late"],
                                     index=0, key="a_fusion_v25")
        use_all_feats_a = st.checkbox("Use all schema features (not just final_score)", value=True, key="a_allfeats_v25")

        t_openai_a = st.slider("OpenAI GPT-4o-mini temperature", 0.0, 1.5, 0.6, 0.1, key="a_temp_oa_v25")
        t_gemini_a = st.slider("Gemini 2.5 temperature", 0.0, 1.5, 0.6, 0.1, key="a_temp_ge_v25")

        force_provider_a = st.selectbox("Force provider", ["auto", "openai only", "gemini only"], index=0, key="a_force_v25")
        fp_a = None if force_provider_a=="auto" else ("openai" if "openai" in force_provider_a else "gemini")

        n_fields = st.slider("Number of field snapshots", 5, 60, 18, 1, key="a_fields_v25")
        seed_a = st.number_input("Random seed (Agriculture)", value=0, step=1, key="a_seed_v25")

    with right:
        run_a = st.button("▶️ Run Agriculture", use_container_width=True, key="a_run_v25")

    if run_a:
        cfg_a = {
            "n_agents": n_agents_a, "heterogeneous": heterogeneous_a, "use_roles": use_roles_a,
            "t_openai": t_openai_a, "t_gemini": t_gemini_a, "n_fields": n_fields, "seed": seed_a,
            "force_provider": fp_a,
            "fusion_mode": {"Hybrid (pick best)":"Hybrid","Early":"Early","Late":"Late"}[fusion_mode_a],
            "use_all_feats": use_all_feats_a
        }
        res = run_agri_lab(cfg_a)

        providers = res["providers"]; X_ef = res["X_ef"]; y = res["y"]
        r2_early = res["r2_early"]; r2_late = res["r2_late"]
        best_mode = res["best_mode"]; r2_best = res["r2_best"]

        st.write("**Providers:**", ", ".join([provider_label(p) for p in providers]))
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Field snapshots", f"{X_ef.shape[0]}")
        c2.metric("Agent Features", f"{X_ef.shape[1]}")
        c3.metric("Early R²", f"{r2_early:.3f}" if np.isfinite(r2_early) else "—")
        c4.metric("Late R²", f"{r2_late:.3f}" if np.isfinite(r2_late) else "—")
        c5.metric("Best", f"{best_mode} (R²={r2_best:.3f})" if np.isfinite(r2_best) else best_mode)

        st.markdown("### Correlation Heatmap (Early-Fusion Features)")
        if X_ef.shape[1] >= 2:
            corr = X_ef.corr(numeric_only=True)
            heat = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index,
                                             zmin=-1, zmax=1, colorscale="RdBu", reversescale=True))
            heat.update_layout(height=380, margin=dict(l=10,r=10,t=20,b=10))
            st.plotly_chart(heat, use_container_width=True, key="a_corr_plot_v25")
            st.info(explain_corr(corr))

        st.markdown("### VIF (Early-Fusion Features)")
        vif_df = res["vif_before"]
        if not vif_df.empty:
            bar = go.Figure(data=[go.Bar(x=vif_df["feature"], y=vif_df["VIF"])])
            bar.update_layout(height=320, margin=dict(l=10,r=10,t=20,b=10))
            st.plotly_chart(bar, use_container_width=True, key="a_vif_plot_v25")
            st.dataframe(vif_df, use_container_width=True, key="a_vif_df_v25")
            st.info(explain_vif(vif_df))

        st.markdown("### Model (Early-Fusion, RidgeCV)")
        reg_df = res["reg_df"]
        if not reg_df.empty:
            fig = go.Figure()
            fig.add_bar(x=reg_df["Feature"], y=reg_df["coef"])
            fig.update_layout(height=320, margin=dict(l=10,r=10,t=20,b=10))
            st.plotly_chart(fig, use_container_width=True, key="a_coef_plot_v25")
            st.caption(res["method_used"])
        else:
            st.write("Insufficient data for early-fusion regression.")

        st.markdown("### Overall diagnosis (Early-Fusion feature space)")
        st.warning(explain_overall(res["cond_before"], res["pc1_before"], providers))

        if res["lf_df"] is not None and not res["lf_df"].empty:
            st.markdown("### Late-Fusion Agent Reliability")
            st.dataframe(res["lf_df"], use_container_width=True, key="a_lf_df_v25")

        st.markdown("### Raw feature matrix (Early-Fusion)")
        st.dataframe(X_ef, use_container_width=True, height=260, key="a_raw_df_v25")

# ---------------- Footer ----------------
st.caption("v2.5 • VIF→PCA→RidgeCV + optional Late-Fusion • Designed & Developed by Jit")
