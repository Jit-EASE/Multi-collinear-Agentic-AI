# agentic_ai_multicollinearity_suite_v2_6.py
# ------------------------------------------------------------------
# v2.6 — Roles made functional (distinct schemas), no final_score anywhere,
#         per-agent calibration (z-score), VIF→PCA→RidgeCV early fusion,
#         reliability-weighted late fusion, non-circular targets,
#         composite designer (read-only) for General lab.
#         Designed & Developed by Jit
# ------------------------------------------------------------------

import os, re, json, hashlib, random
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

import streamlit as st
import plotly.graph_objects as go

from statsmodels.stats.outliers_influence import variance_inflation_factor
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
st.set_page_config(page_title="Agentic AI Multicollinearity Suite v2.6", page_icon="🧮", layout="wide")
st.markdown("### Agentic AI Multicollinearity Suite — v2.6")

# ---------------- Utilities ----------------
def hash_key(obj: dict) -> str:
    try:
        s = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    except Exception:
        s = str(obj)
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def parse_json_block(txt: str, keys: List[str]) -> Dict[str, float]:
    if not txt or not keys:
        return {}
    # Try strict JSON extraction first
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
    # Fallback: scrape numeric tokens in order
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
    s = np.linalg.svd(Z, full_matrices=False, compute_uv=False)
    return float((s.max()/s.min()) if (s.min() > 0) else np.inf)

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
        providers = ["openai"]
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
            return "Not enough features to compute correlations."
        mask = np.triu(np.ones(M.shape, dtype=bool), k=1)
        vals = M.where(mask).stack()
        if vals.empty:
            return "Correlation matrix is empty."
        max_pair = vals.abs().idxmax(); max_val = float(vals.loc[max_pair])
        mean_abs = float(vals.abs().mean())
        txt = [f"Average |correlation|: **{mean_abs:.2f}**. ",
               f"Strongest pair: **{max_pair[0]}–{max_pair[1]} = {max_val:.2f}**. "]
        if mean_abs >= 0.6 or abs(max_val) >= 0.9:
            txt.append("Signals are highly redundant.")
        elif mean_abs >= 0.4:
            txt.append("Moderate redundancy present.")
        else:
            txt.append("Signals appear reasonably diverse.")
        return "".join(txt)
    except Exception:
        return "Correlation explanation unavailable."

def explain_vif(vif_df: pd.DataFrame) -> str:
    try:
        df = vif_df.dropna()
        if df.empty:
            return "VIF not available."
        severe = df[df["VIF"] >= 10].shape[0]
        moderate = df[(df["VIF"] >= 5) & (df["VIF"] < 10)].shape[0]
        top = df.iloc[df["VIF"].idxmax()]
        txt = [f"Max VIF **{top['VIF']:.1f}** at **{top['feature']}**. "]
        if severe > 0:
            txt.append(f"**{severe}** features exceed 10 (severe). ")
        elif moderate > 0:
            txt.append(f"**{moderate}** features in 5–10 (moderate). ")
        else:
            txt.append("No VIFs exceed 5. ")
        txt.append("High VIF inflates standard errors and destabilizes coefficients.")
        return "".join(txt)
    except Exception:
        return "VIF explanation unavailable."

def explain_overall(cidx: float, pc1: float, providers: list) -> str:
    prov_kind = "heterogeneous (multi-LLM)" if len(set(providers)) > 1 else "homogeneous (single LLM)"
    pieces = [f"System: **{prov_kind}**. "]
    if np.isfinite(cidx):
        if cidx > 30: pieces.append(f"Condition Index **{cidx:.1f}** → serious multicollinearity. ")
        elif cidx > 10: pieces.append(f"Condition Index **{cidx:.1f}** → moderate multicollinearity. ")
        else: pieces.append(f"Condition Index **{cidx:.1f}** → low multicollinearity. ")
    if not np.isnan(pc1):
        if pc1 >= 0.6: pieces.append(f"PC1 explains **{pc1*100:.1f}%** → redundant signals dominate. ")
        elif pc1 >= 0.4: pieces.append(f"PC1 explains **{pc1*100:.1f}%** → some redundancy. ")
        else: pieces.append(f"PC1 explains **{pc1*100:.1f}%** → diverse signals. ")
    pieces.append("Mitigations: per-agent calibration, VIF pruning, PCA fallback, and late-fusion by skill.")
    return "".join(pieces)

# ---------------- Role Schemas (distinct outputs) ----------------
# General / Policy roles
GEN_ROLES = [
    "Planner", "Critic", "Risk Analyst", "Econometrician", "Operations Lead"
]
ROLE_KEYS_GEN: Dict[str, List[str]] = {
    "Planner": ["policy_score","efficiency_score"],
    "Critic": ["evidence_strength","factual_risk"],
    "Risk Analyst": ["adoption_risk","regulatory_feasibility"],
    "Econometrician": ["uncertainty_index","data_quality"],
    "Operations Lead": ["implementation_effort","expected_roi"]
}
# Ranges guidance for prompts (0–100)
ROLE_RANGES_GEN: Dict[str, Dict[str, str]] = {
    "policy_score":"0-100", "efficiency_score":"0-100",
    "evidence_strength":"0-100", "factual_risk":"0-100",
    "adoption_risk":"0-100", "regulatory_feasibility":"0-100",
    "uncertainty_index":"0-100", "data_quality":"0-100",
    "implementation_effort":"0-100", "expected_roi":"0-100"
}

# Agriculture roles
AG_ROLES = [
    "Agronomist", "Irrigation Specialist", "Plant Pathologist", "Agricultural Economist", "Compliance Officer"
]
ROLE_KEYS_AG: Dict[str, List[str]] = {
    "Agronomist": ["yield_gain","crop_health"],                   # 0-100
    "Irrigation Specialist": ["irrigation_mm","water_stress"],    # 0-60, 0-100
    "Plant Pathologist": ["pest_risk","disease_pressure"],        # 0-100, 0-100
    "Agricultural Economist": ["input_roi","cost_pressure"],      # 0-100, 0-100
    "Compliance Officer": ["nitrate_risk","compliance_score"]     # 0-100, 0-100
}
ROLE_RANGES_AG: Dict[str, Dict[str, str]] = {
    "yield_gain":"0-100", "crop_health":"0-100",
    "irrigation_mm":"0-60", "water_stress":"0-100",
    "pest_risk":"0-100", "disease_pressure":"0-100",
    "input_roi":"0-100", "cost_pressure":"0-100",
    "nitrate_risk":"0-100", "compliance_score":"0-100"
}

# ---------------- Prompts ----------------
GEN_SYSTEM = "You are an expert policy analyst. Return ONLY the requested JSON keys with numeric values."
AG_SYSTEM  = "You are an expert agronomy decision agent. Return ONLY the requested JSON keys with numeric values."

GEN_DEFAULT_PROMPTS = [
    "Assess national AI-adoption capacity for agricultural analytics in 2025.",
    "Evaluate resilience of the EU food system to climate shocks in the next season.",
    "Estimate operational gains from edge-AI deployment in livestock monitoring.",
    "Rate feasibility of a nationwide farm telemetry rollout under current regulation.",
    "Estimate supply-chain visibility improvement from multimodal IoT in beef sector."
]

# ---------------- Field Generator (Agri) ----------------
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
""".strip()

# ---------------- Collinearity Controls & Models ----------------
def drop_high_vif(df: pd.DataFrame, thresh: float = 10.0) -> Tuple[pd.DataFrame, List[str]]:
    dropped = []
    if df.shape[1] < 2: return df, dropped
    while True:
        vif_df = safe_vif(df)
        if vif_df["VIF"].isna().all(): break
        max_vif = vif_df["VIF"].max()
        if max_vif > thresh:
            drop_feat = vif_df.loc[vif_df["VIF"].idxmax(), "feature"]
            df = df.drop(columns=[drop_feat]); dropped.append(drop_feat)
            if df.shape[1] < 2: break
        else: break
    return df, dropped

def apply_pca_if_needed(df: pd.DataFrame, var_threshold: float = 0.90) -> Tuple[pd.DataFrame, int]:
    if df.shape[1] < 2: return df.copy(), df.shape[1]
    Z = StandardScaler().fit_transform(df.values)
    p = PCA().fit(Z)
    cumvar = np.cumsum(p.explained_variance_ratio_)
    ncomp = int(np.searchsorted(cumvar, var_threshold) + 1)
    Z_pca = p.transform(Z)[:, :ncomp]
    cols = [f"PC{i+1}" for i in range(ncomp)]
    return pd.DataFrame(Z_pca, columns=cols), ncomp

def run_regression_pipeline(X: pd.DataFrame, y: np.ndarray) -> Tuple[float, pd.DataFrame, str, pd.DataFrame, float, float]:
    if X.shape[0] < (X.shape[1] + 2):
        return np.nan, pd.DataFrame(), "Insufficient observations for regression.", pd.DataFrame(), np.nan, np.nan
    vif_before = safe_vif(X)
    cond_before = condition_index(X)
    pc1_before = pca_first_share(X)
    X_vif, dropped = drop_high_vif(X, thresh=10.0)
    if condition_index(X_vif) > 30 and X_vif.shape[1] >= 2:
        X_final, ncomp = apply_pca_if_needed(X_vif, var_threshold=0.90)
        method = f"PCA({ncomp}) after VIF drop={dropped}"
    else:
        X_final, method = X_vif, f"VIF drop={dropped if dropped else 'None'}"
    model = Pipeline([("scaler", StandardScaler()),
                      ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 20), cv=5))])
    model.fit(X_final, y)
    r2 = model.score(X_final, y)
    reg_df = pd.DataFrame({"Feature": list(X_final.columns),
                           "coef": model.named_steps["ridge"].coef_})
    return r2, reg_df, method, vif_before, cond_before, pc1_before

def reliability_weight(r2cv: float) -> float:
    return max(r2cv, 0.0) ** 2

def late_fusion_ensemble(agent_blocks: List[pd.DataFrame], y: np.ndarray, calibrate: bool=True) -> Tuple[float, pd.DataFrame]:
    if len(agent_blocks) == 0: return np.nan, pd.DataFrame()
    kf = KFold(n_splits=max(3, min(5, len(y))), shuffle=True, random_state=42)
    rows = []; preds = []
    for i, F in enumerate(agent_blocks):
        F = F.copy()
        # Drop empty columns if any
        if F.shape[1] == 0 or F.dropna().shape[0] < (F.shape[1] + 2):
            preds.append(np.full_like(y, np.nan, dtype=float))
            rows.append({"Agent": f"Agent{i+1}", "r2cv": np.nan, "weight": 0.0}); continue
        # Optional per-agent calibration (z-score)
        if calibrate:
            F = (F - F.mean()) / (F.std(ddof=0) + 1e-6)
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 20), cv=5))])
        try:
            r2cv_scores = cross_val_score(pipe, F.values, y, scoring="r2", cv=kf)
            r2cv = float(np.nanmean(r2cv_scores))
            yhat = cross_val_predict(pipe, F.values, y, cv=kf)
        except Exception:
            r2cv = np.nan; yhat = np.full_like(y, np.nan, dtype=float)
        preds.append(yhat)
        rows.append({"Agent": f"Agent{i+1}", "r2cv": r2cv, "weight": reliability_weight(r2cv)})
    P = np.vstack(preds)
    weights = np.array([r["weight"] for r in rows], dtype=float)
    yhat_ens = np.zeros_like(y, dtype=float)
    for t in range(len(y)):
        col = P[:, t]; m = np.isfinite(col)
        if not m.any(): yhat_ens[t] = np.nan
        else:
            w = weights[m]; yhat_ens[t] = np.dot(w, col[m]) / (w.sum() if w.sum() != 0 else 1.0)
    m2 = np.isfinite(yhat_ens)
    ensemble_r2 = r2_score(y[m2], yhat_ens[m2]) if m2.sum() >= 3 else np.nan
    return ensemble_r2, pd.DataFrame(rows)

# ---------------- Lab Runners ----------------
def gen_role_keys(idx: int) -> List[str]:
    role = GEN_ROLES[idx % len(GEN_ROLES)]
    return ROLE_KEYS_GEN[role]

def ag_role_keys(idx: int) -> List[str]:
    role = AG_ROLES[idx % len(AG_ROLES)]
    return ROLE_KEYS_AG[role]

def gen_role_label(idx: int) -> str:
    return GEN_ROLES[idx % len(GEN_ROLES)]

def ag_role_label(idx: int) -> str:
    return AG_ROLES[idx % len(AG_ROLES)]

def build_schema_line(keys: List[str], ranges_map: Dict[str, str]) -> str:
    parts = []
    for k in keys:
        r = ranges_map.get(k, "0-100")
        parts.append(f"{k} ({r})")
    return "Return ONLY a JSON object with keys: " + ", ".join(parts) + "."

def run_general_lab(cfg: dict) -> dict:
    n_agents = int(cfg["n_agents"]); heterogeneous = bool(cfg["heterogeneous"])
    t_openai = float(cfg["t_openai"]); t_gemini = float(cfg["t_gemini"])
    use_roles = bool(cfg["use_roles"]); calibrate = bool(cfg["calibrate"])
    prompts: List[str] = cfg["prompts"]
    fusion_mode = cfg.get("fusion_mode", "Hybrid")
    force_provider = cfg.get("force_provider", None)
    seed = int(cfg.get("seed", 0))
    if seed: random.seed(seed); np.random.seed(seed)

    providers = assign_models(n_agents, heterogeneous, force_provider=force_provider)
    oa = build_openai()

    # Collect role-specific outputs per agent
    rows_full = []
    for p in prompts:
        row_vals = []
        for a in range(n_agents):
            prov = providers[a]
            role = gen_role_label(a) if use_roles else "Generalist"
            keys = gen_role_keys(a) if use_roles else ["policy_score","efficiency_score"]
            schema_line = build_schema_line(keys, ROLE_RANGES_GEN)
            uprompt = f"Task: {p}\nROLE: {role}\n{schema_line}"
            txt = call_openai_json(oa, GEN_SYSTEM, uprompt, temperature=t_openai) if prov=="openai" \
                  else call_gemini_json(GEN_SYSTEM, uprompt, temperature=t_gemini)
            if txt is None:
                # fallback defaults: center-ish numbers
                fallback = {k: 50.0 for k in keys}
                parsed = fallback
            else:
                parsed = parse_json_block(txt, keys)
                for k in keys:
                    if parsed.get(k) is None:
                        parsed[k] = 50.0
            # append in order
            row_vals.extend([parsed[k] for k in keys])
        rows_full.append(row_vals)

    # Build feature matrix (Early-fusion = union of all role keys across agents)
    colnames = []
    for a in range(n_agents):
        keys = gen_role_keys(a) if use_roles else ["policy_score","efficiency_score"]
        for k in keys:
            colnames.append(f"A{a+1}_{k}")
    X_full = pd.DataFrame(rows_full, columns=colnames)

    # Optional per-agent calibration
    if calibrate:
        for a in range(n_agents):
            cols = [c for c in X_full.columns if c.startswith(f"A{a+1}_")]
            block = X_full[cols]
            X_full.loc[:, cols] = (block - block.mean()) / (block.std(ddof=0) + 1e-6)

    # Early-fusion feature matrix:
    X_ef = X_full.copy()

    # Late-fusion blocks (per agent)
    agent_blocks = []
    for a in range(n_agents):
        cols = [c for c in X_full.columns if c.startswith(f"A{a+1}_")]
        agent_blocks.append(X_full[cols].copy())

    # Independent (non-circular) policy target: latent difficulty + noise
    # Emulate external drivers that models do NOT see
    latent = np.random.normal(0, 1, size=X_full.shape[0])
    exog1  = np.random.normal(50, 15, size=X_full.shape[0])
    exog2  = np.random.normal(30, 10, size=X_full.shape[0])
    y = 40 + 8*latent + 0.2*exog1 - 0.1*exog2 + np.random.normal(0, 6, size=X_full.shape[0])

    # Early fusion
    r2_early, reg_df, method_used, vif_before, cond_before, pc1_before = run_regression_pipeline(X_ef, y)
    # Late fusion
    r2_late, lf_df = late_fusion_ensemble(agent_blocks, y, calibrate=calibrate)

    # Best mode
    if fusion_mode == "Early":
        r2_best, best_mode = r2_early, "Early Fusion"
    elif fusion_mode == "Late":
        r2_best, best_mode = r2_late, "Late Fusion"
    else:
        r2_best, best_mode = (r2_early, "Early Fusion") if (np.nan_to_num(r2_early, nan=-1) >= np.nan_to_num(r2_late, nan=-1)) \
                             else (r2_late, "Late Fusion")

    return {
        "providers": providers, "X_ef": X_ef, "y": y,
        "r2_early": r2_early, "r2_late": r2_late, "r2_best": r2_best, "best_mode": best_mode,
        "reg_df": reg_df, "method_used": method_used,
        "vif_before": vif_before, "cond_before": cond_before, "pc1_before": pc1_before,
        "lf_df": lf_df
    }

def run_agri_lab(cfg: dict) -> dict:
    n_agents = int(cfg["n_agents"]); heterogeneous = bool(cfg["heterogeneous"])
    t_openai = float(cfg["t_openai"]); t_gemini = float(cfg["t_gemini"])
    use_roles = bool(cfg["use_roles"]); calibrate = bool(cfg["calibrate"])
    n_fields = int(cfg["n_fields"]); seed = int(cfg.get("seed", 0))
    fusion_mode = cfg.get("fusion_mode", "Hybrid")
    force_provider = cfg.get("force_provider", None)
    if seed: random.seed(seed); np.random.seed(seed)

    providers = assign_models(n_agents, heterogeneous, force_provider=force_provider)
    oa = build_openai()

    rows_full = []; raw_fields = []
    for i in range(n_fields):
        card = make_field(seed + i if seed else None); raw_fields.append(card)
        row_vals = []
        for a in range(n_agents):
            prov = providers[a]
            role = ag_role_label(a) if use_roles else "Generalist"
            keys = ag_role_keys(a) if use_roles else ["yield_gain","crop_health"]
            # Build role-specific schema (with ranges)
            parts = []
            for k in keys:
                parts.append(f"{k} ({ROLE_RANGES_AG.get(k,'0-100')})")
            schema_line = "Return ONLY a JSON object with keys: " + ", ".join(parts) + "."
            uprompt = field_prompt(card) + f"\nROLE: {role}\n" + schema_line
            txt = call_openai_json(oa, AG_SYSTEM, uprompt, temperature=t_openai) if prov=="openai" \
                  else call_gemini_json(AG_SYSTEM, uprompt, temperature=t_gemini)
            if txt is None:
                parsed = {}
                for k in keys:
                    # basic plausible defaults by key
                    if k == "irrigation_mm": parsed[k] = 15.0
                    else: parsed[k] = 50.0
            else:
                parsed = parse_json_block(txt, keys)
                for k in keys:
                    if parsed.get(k) is None:
                        parsed[k] = 50.0 if k != "irrigation_mm" else 15.0
            row_vals.extend([parsed[k] for k in keys])
        rows_full.append(row_vals)

    # Early-fusion features: union of role keys across agents
    colnames = []
    for a in range(n_agents):
        keys = ag_role_keys(a) if use_roles else ["yield_gain","crop_health"]
        for k in keys:
            colnames.append(f"A{a+1}_{k}")
    X_full = pd.DataFrame(rows_full, columns=colnames)

    # Optional per-agent calibration
    if calibrate:
        for a in range(n_agents):
            cols = [c for c in X_full.columns if c.startswith(f"A{a+1}_")]
            block = X_full[cols]
            X_full.loc[:, cols] = (block - block.mean()) / (block.std(ddof=0) + 1e-6)

    X_ef = X_full.copy()
    agent_blocks = []
    for a in range(n_agents):
        cols = [c for c in X_full.columns if c.startswith(f"A{a+1}_")]
        agent_blocks.append(X_full[cols].copy())

    # Realistic agronomic target (independent of agent outputs)
    y_proxy = []
    for f in raw_fields:
        stress = max(0.0, (f["ET0_mm_day"]*10 - f["soil_moisture_pct"]) - 0.2*f["rain_last_7d_mm"])
        yv = 20 + (f["ndvi"]*100)*0.8 - 0.3*stress + np.random.normal(0, 2.0)
        y_proxy.append(yv)
    y = np.array(y_proxy)

    # Early fusion
    r2_early, reg_df, method_used, vif_before, cond_before, pc1_before = run_regression_pipeline(X_ef, y)
    # Late fusion
    r2_late, lf_df = late_fusion_ensemble(agent_blocks, y, calibrate=calibrate)

    if fusion_mode == "Early":
        r2_best, best_mode = r2_early, "Early Fusion"
    elif fusion_mode == "Late":
        r2_best, best_mode = r2_late, "Late Fusion"
    else:
        r2_best, best_mode = (r2_early, "Early Fusion") if (np.nan_to_num(r2_early, nan=-1) >= np.nan_to_num(r2_late, nan=-1)) \
                             else (r2_late, "Late Fusion")

    return {
        "providers": providers, "X_ef": X_ef, "y": y,
        "r2_early": r2_early, "r2_late": r2_late, "r2_best": r2_best, "best_mode": best_mode,
        "reg_df": reg_df, "method_used": method_used,
        "vif_before": vif_before, "cond_before": cond_before, "pc1_before": pc1_before,
        "lf_df": lf_df
    }

# ---------------- UI ----------------
st.sidebar.subheader("Provider Status")
st.sidebar.write(f"OpenAI key detected: {'✅' if os.getenv('OPENAI_API_KEY') else '❌'}")
st.sidebar.write(f"Gemini key detected: {'✅' if os.getenv('GEMINI_API_KEY') else '❌'}")
if HAS_GEMINI and os.getenv("GEMINI_API_KEY"):
    if st.sidebar.button("Test Gemini call", key="sidebar_test_gemini_v26"):
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
                        index=1, key="g_radio_mode_v26")
        heterogeneous = mode.startswith("Heterogeneous")
        n_agents = st.slider("Agents (General)", 2, 6, 3, 1, key="g_slider_agents_v26")
        use_roles = st.checkbox("Role specialization (General)", value=heterogeneous, key="g_roles_v26")
        calibrate = st.checkbox("Per-agent calibration (z-score)", value=True, key="g_cal_v26")
        fusion_mode = st.selectbox("Fusion Strategy", ["Hybrid (pick best)", "Early", "Late"], index=0, key="g_fusion_v26")
        t_openai = st.slider("OpenAI GPT-4o-mini temperature", 0.0, 1.5, 0.7, 0.1, key="g_temp_oa_v26")
        t_gemini = st.slider("Gemini 2.5 temperature", 0.0, 1.5, 0.7, 0.1, key="g_temp_ge_v26")
        force_provider = st.selectbox("Force provider", ["auto", "openai only", "gemini only"], index=0, key="g_force_v26")
        fp = None if force_provider=="auto" else ("openai" if "openai" in force_provider else "gemini")
        seed = st.number_input("Random seed (General)", value=0, step=1, key="g_seed_v26")

        st.markdown("#### Composite Designer (read-only)")
        w_eff = st.slider("Weight: efficiency_score", 0.0, 1.0, 0.30, 0.05, key="g_w_eff")
        w_feas = st.slider("Weight: regulatory_feasibility", 0.0, 1.0, 0.25, 0.05, key="g_w_feas")
        w_roi = st.slider("Weight: expected_roi", 0.0, 1.0, 0.25, 0.05, key="g_w_roi")
        w_risk = st.slider("Weight (negative): adoption/factual risk", 0.0, 1.0, 0.20, 0.05, key="g_w_risk")

    with right:
        default_text = "\n".join(GEN_DEFAULT_PROMPTS)
        ptext = st.text_area("Prompts (General) — one per line", value=default_text, height=220, key="g_prompts_v26")
        prompts = [p.strip() for p in ptext.splitlines() if p.strip()]
        run_g = st.button("▶️ Run General/Policy", use_container_width=True, key="g_run_v26")

    if run_g:
        cfg = {"n_agents": n_agents, "heterogeneous": heterogeneous, "use_roles": use_roles, "calibrate": calibrate,
               "t_openai": t_openai, "t_gemini": t_gemini, "prompts": prompts, "force_provider": fp,
               "fusion_mode": {"Hybrid (pick best)":"Hybrid","Early":"Early","Late":"Late"}[fusion_mode], "seed": seed}
        res = run_general_lab(cfg)

        providers = res["providers"]; X_ef = res["X_ef"]
        r2_early, r2_late, r2_best, best_mode = res["r2_early"], res["r2_late"], res["r2_best"], res["best_mode"]

        st.write("**Providers:**", ", ".join([provider_label(p) for p in providers]))
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Observations", f"{X_ef.shape[0]}")
        c2.metric("Features (Early)", f"{X_ef.shape[1]}")
        c3.metric("Early R²", f"{r2_early:.3f}" if np.isfinite(r2_early) else "—")
        c4.metric("Late R²", f"{r2_late:.3f}" if np.isfinite(r2_late) else "—")
        c5.metric("Best", f"{best_mode} (R²={r2_best:.3f})" if np.isfinite(r2_best) else best_mode)

        st.markdown("### Correlation Heatmap (Early-Fusion Features)")
        if X_ef.shape[1] >= 2:
            corr = X_ef.corr(numeric_only=True)
            heat = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index,
                                             zmin=-1, zmax=1, colorscale="RdBu", reversescale=True))
            heat.update_layout(height=380, margin=dict(l=10,r=10,t=20,b=10))
            st.plotly_chart(heat, use_container_width=True, key="g_corr_plot_v26")
            st.info(explain_corr(corr))

        st.markdown("### VIF (Early-Fusion Features)")
        vif_df = res["vif_before"]
        if not vif_df.empty:
            bar = go.Figure(data=[go.Bar(x=vif_df["feature"], y=vif_df["VIF"])])
            bar.update_layout(height=320, margin=dict(l=10,r=10,t=20,b=10))
            st.plotly_chart(bar, use_container_width=True, key="g_vif_plot_v26")
            st.dataframe(vif_df, use_container_width=True, key="g_vif_df_v26")
            st.info(explain_vif(vif_df))

        st.markdown("### Model (Early-Fusion, RidgeCV)")
        reg_df = res["reg_df"]
        if not reg_df.empty:
            fig = go.Figure(); fig.add_bar(x=reg_df["Feature"], y=reg_df["coef"])
            fig.update_layout(height=320, margin=dict(l=10,r=10,t=20,b=10))
            st.plotly_chart(fig, use_container_width=True, key="g_coef_plot_v26")
            st.caption(res["method_used"])
        else:
            st.write("Insufficient data for early-fusion regression.")

        st.markdown("### Overall diagnosis (Early-Fusion feature space)")
        st.warning(explain_overall(res["cond_before"], res["pc1_before"], providers))

        if res["lf_df"] is not None and not res["lf_df"].empty:
            st.markdown("### Late-Fusion Agent Reliability")
            st.dataframe(res["lf_df"], use_container_width=True, key="g_lf_df_v26")

        st.markdown("### Raw feature matrix (Early-Fusion)")
        st.dataframe(X_ef, use_container_width=True, height=260, key="g_raw_df_v26")

# -------- Agriculture Tab --------
with tab2:
    left, right = st.columns([0.60, 0.40])
    with left:
        mode_a = st.radio("System Type (Agriculture)", ["Homogeneous (single LLM)", "Heterogeneous (multi-LLM)"],
                          index=1, key="a_radio_mode_v26")
        heterogeneous_a = mode_a.startswith("Heterogeneous")
        n_agents_a = st.slider("Agents (Agriculture)", 2, 6, 3, 1, key="a_slider_agents_v26")
        use_roles_a = st.checkbox("Role specialization (Agriculture)", value=True, key="a_roles_v26")
        calibrate_a = st.checkbox("Per-agent calibration (z-score)", value=True, key="a_cal_v26")
        fusion_mode_a = st.selectbox("Fusion Strategy (Agriculture)",
                                     ["Hybrid (pick best)", "Early", "Late"],
                                     index=0, key="a_fusion_v26")
        t_openai_a = st.slider("OpenAI GPT-4o-mini temperature", 0.0, 1.5, 0.6, 0.1, key="a_temp_oa_v26")
        t_gemini_a = st.slider("Gemini 2.5 temperature", 0.0, 1.5, 0.6, 0.1, key="a_temp_ge_v26")
        force_provider_a = st.selectbox("Force provider", ["auto", "openai only", "gemini only"], index=0, key="a_force_v26")
        fp_a = None if force_provider_a=="auto" else ("openai" if "openai" in force_provider_a else "gemini")
        n_fields = st.slider("Number of field snapshots", 5, 60, 18, 1, key="a_fields_v26")
        seed_a = st.number_input("Random seed (Agriculture)", value=0, step=1, key="a_seed_v26")

    with right:
        run_a = st.button("▶️ Run Agriculture", use_container_width=True, key="a_run_v26")

    if run_a:
        cfg_a = {"n_agents": n_agents_a, "heterogeneous": heterogeneous_a, "use_roles": use_roles_a, "calibrate": calibrate_a,
                 "t_openai": t_openai_a, "t_gemini": t_gemini_a, "n_fields": n_fields, "seed": seed_a,
                 "force_provider": fp_a, "fusion_mode": {"Hybrid (pick best)":"Hybrid","Early":"Early","Late":"Late"}[fusion_mode_a]}
        res = run_agri_lab(cfg_a)

        providers = res["providers"]; X_ef = res["X_ef"]
        r2_early, r2_late, r2_best, best_mode = res["r2_early"], res["r2_late"], res["r2_best"], res["best_mode"]

        st.write("**Providers:**", ", ".join([provider_label(p) for p in providers]))
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Field snapshots", f"{X_ef.shape[0]}")
        c2.metric("Features (Early)", f"{X_ef.shape[1]}")
        c3.metric("Early R²", f"{r2_early:.3f}" if np.isfinite(r2_early) else "—")
        c4.metric("Late R²", f"{r2_late:.3f}" if np.isfinite(r2_late) else "—")
        c5.metric("Best", f"{best_mode} (R²={r2_best:.3f})" if np.isfinite(r2_best) else best_mode)

        st.markdown("### Correlation Heatmap (Early-Fusion Features)")
        if X_ef.shape[1] >= 2:
            corr = X_ef.corr(numeric_only=True)
            heat = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index,
                                             zmin=-1, zmax=1, colorscale="RdBu", reversescale=True))
            heat.update_layout(height=380, margin=dict(l=10,r=10,t=20,b=10))
            st.plotly_chart(heat, use_container_width=True, key="a_corr_plot_v26")
            st.info(explain_corr(corr))

        st.markdown("### VIF (Early-Fusion Features)")
        vif_df = res["vif_before"]
        if not vif_df.empty:
            bar = go.Figure(data=[go.Bar(x=vif_df["feature"], y=vif_df["VIF"])])
            bar.update_layout(height=320, margin=dict(l=10,r=10,t=20,b=10))
            st.plotly_chart(bar, use_container_width=True, key="a_vif_plot_v26")
            st.dataframe(vif_df, use_container_width=True, key="a_vif_df_v26")
            st.info(explain_vif(vif_df))

        st.markdown("### Model (Early-Fusion, RidgeCV)")
        reg_df = res["reg_df"]
        if not reg_df.empty:
            fig = go.Figure(); fig.add_bar(x=reg_df["Feature"], y=reg_df["coef"])
            fig.update_layout(height=320, margin=dict(l=10,r=10,t=20,b=10))
            st.plotly_chart(fig, use_container_width=True, key="a_coef_plot_v26")
            st.caption(res["method_used"])
        else:
            st.write("Insufficient data for early-fusion regression.")

        st.markdown("### Overall diagnosis (Early-Fusion feature space)")
        st.warning(explain_overall(res["cond_before"], res["pc1_before"], providers))

        if res["lf_df"] is not None and not res["lf_df"].empty:
            st.markdown("### Late-Fusion Agent Reliability")
            st.dataframe(res["lf_df"], use_container_width=True, key="a_lf_df_v26")

        st.markdown("### Raw feature matrix (Early-Fusion)")
        st.dataframe(X_ef, use_container_width=True, height=260, key="a_raw_df_v26")

# ---------------- Footer ----------------
st.caption("v2.6 • Role-specific schemas, per-agent calibration, VIF→PCA→RidgeCV, late-fusion by skill • Designed & Developed by Jit")
