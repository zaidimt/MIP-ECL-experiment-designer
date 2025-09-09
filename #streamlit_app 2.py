#streamlit_app 2
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import itertools

# ---------- Constants ----------
AFF_LABELED = {
    'low': '≤2 kD*',
    'medium': '13 kD*',
    'high': '≥59 kD*'
}
AFF_KD_FROM_SIMPLE = {'low': 2, 'medium': 13, 'high': 59}
KD_TO_LABEL = {2: '≤2 kD*', 13: '13 kD*', 59: '≥59 kD*'}
AFFINITY_ORDERED = ['≤2 kD*', '13 kD*', '≥59 kD*']

# ---------- Load Data ----------
@st.cache_data
def load_data(path="App_data.xlsx"):
    df = pd.read_excel(path)
    if "protein.copy.nbr" in df.columns:
        df.rename(columns={"protein.copy.nbr": "analyte.copy.nbr"}, inplace=True)
    if "Affinity" in df.columns:
        df["Affinity_label"] = df["Affinity"].map(AFF_LABELED)
        df["Affinity_KD"] = df["Affinity"].map(AFF_KD_FROM_SIMPLE)
    else:
        raise ValueError("Missing 'Affinity' column")
    df["analyte.copy.nbr_fmt"] = df["analyte.copy.nbr"].apply(lambda x: f"{int(x):,}" if pd.notna(x) and x > 0 else "0")
    return df

df = load_data()

# ---------- Train / Cache Model ----------
features = ['analyte.copy.nbr', 'capture', 'probe', 'Affinity_KD']
y = df['log10_SN1']
X = df[features]

@st.cache_resource
def train_model(X, y):
    model = GradientBoostingRegressor()
    model.fit(X, y)
    return model

model = train_model(X, y)

# ---------- Precompute full prediction grid ----------
@st.cache_data
def generate_param_grid(df, features):
    analyte_vals = df['analyte.copy.nbr'].unique()
    capture_vals = df['capture'].unique()
    probe_vals = df['probe'].unique()
    affinity_vals = df['Affinity_KD'].unique()

    grid = pd.DataFrame(
        list(itertools.product(analyte_vals, capture_vals, probe_vals, affinity_vals)),
        columns=features
    )
    grid['Affinity_label'] = grid['Affinity_KD'].map(KD_TO_LABEL)
    grid['key'] = grid[features].astype(str).agg('-'.join, axis=1)
    return grid

param_grid = generate_param_grid(df, features)

df['key'] = df[features].astype(str).agg('-'.join, axis=1)
missing_combos = param_grid[~param_grid['key'].isin(df['key'])].copy()
missing_combos['log10_SN1'] = model.predict(missing_combos[features])
missing_combos['source'] = 'predicted'
df_measured = df.copy()
df_measured['source'] = 'measured'
df_combined = pd.concat([df_measured, missing_combos], ignore_index=True)

# ---------- Streamlit Layout ----------
st.set_page_config(page_title="MIP-ECL Experiment Designer", layout="wide")
st.title("🔬 MIP-ECL Experiment Designer")

st.markdown(
    """
    Interactive tool to predict S/N based on:
    - Analyte copy number
    - Capture concentration
    - Probe concentration
    - Binding affinity (kD*)
    """
)

# ---------- Sidebar Inputs ----------
st.sidebar.markdown("## 🔧 Input Panel")
with st.sidebar.expander("🎯 Predict S/N from known parameters", expanded=True):
    analyte_map = {f"{x:.0e}": x for x in sorted(df_combined['analyte.copy.nbr'].unique())}
    analyte_fmt = st.select_slider("Analyte copy number", options=list(analyte_map.keys()))
    analyte = analyte_map[analyte_fmt]

    capture = st.select_slider("Capture reagent conc. (µg/ml)", options=sorted(df_combined['capture'].unique()))
    probe = st.select_slider("Probe reagent conc. (µg/ml)", options=sorted(df_combined['probe'].unique()))
    affinity_label = st.select_slider("Affinity (kD*)", options=AFFINITY_ORDERED, value='13 kD*')
    kd = {v: k for k, v in KD_TO_LABEL.items()}[affinity_label]

with st.sidebar.expander("📈 Optimize parameters for target S/N", expanded=False):
    target_sn_linear = st.number_input("Target S/N value", value=10.0, step=1.0)
    target_sn = np.log10(target_sn_linear)
    selected_affinity = st.selectbox("Affinity (kD*)", options=["any"] + AFFINITY_ORDERED)
    analyte_target_fmt = st.select_slider("Analyte copy number", options=["any"] + list(analyte_map.keys()), value="any")
    analyte_target = analyte_map[analyte_target_fmt] if analyte_target_fmt != "any" else "any"

# ---------- Prediction ----------
def get_nearby_points(df_combined, analyte, capture, probe, kd, tol=0.3):
    mask = (
        (np.abs(df_combined['analyte.copy.nbr'] - analyte) <= tol * analyte) &
        (np.abs(df_combined['capture'] - capture) <= tol * capture) &
        (np.abs(df_combined['probe'] - probe) <= tol * probe) &
        (np.abs(df_combined['Affinity_KD'] - kd) <= tol * kd)
    )
    nearby = df_combined[mask].copy()
    if not nearby.empty:
        nearby['S/N'] = 10 ** nearby['log10_SN1']
    return nearby

exact_match = df_combined[
    (df_combined['analyte.copy.nbr'] == analyte) &
    (df_combined['capture'] == capture) &
    (df_combined['probe'] == probe) &
    (df_combined['Affinity_KD'] == kd)
]

with st.expander("### 🔍 Predicted S/N based on selected parameters", expanded=True):
    if not exact_match.empty:
        row = exact_match.iloc[0]
        sn = row['log10_SN1']
        st.metric(label="log10(S/N)", value=f"{sn:.2f}")
        st.metric(label="S/N", value=f"{10**sn:.2f}")
        st.caption(f"Source: {'Predicted' if row['source']=='predicted' else 'Measured'}")
    else:
        st.warning("No exact match found.")

    nearby = get_nearby_points(df_combined, analyte, capture, probe, kd)
    if not nearby.empty:
        st.dataframe(
            nearby[['analyte.copy.nbr_fmt','capture','probe','Affinity_label','log10_SN1','S/N','source']]
            .rename(columns={'analyte.copy.nbr_fmt':'Analyte Copy Number'})
            .sort_values('log10_SN1', ascending=False)
        )
    else:
        st.info("No nearby points found.")

# ---------- Optimization ----------
with st.expander("### 📈 Optimized parameters for target S/N", expanded=False):
    df_sn = df_combined.copy()
    if selected_affinity != "any":
        df_sn = df_sn[df_sn['Affinity_label'] == selected_affinity]
    if analyte_target != "any":
        df_sn = df_sn[np.isclose(df_sn['analyte.copy.nbr'], float(analyte_target), rtol=0.01)]
    matches = df_sn[np.isclose(df_sn['log10_SN1'], target_sn, atol=0.1)]
    if not matches.empty:
        matches = matches.copy()
        matches['S/N'] = 10 ** matches['log10_SN1']
        st.success(f"Parameter combinations close to target S/N ≈ {target_sn_linear:.2f}:")
        st.dataframe(
            matches[['analyte.copy.nbr_fmt','capture','probe','Affinity_label','log10_SN1','S/N','source']]
            .rename(columns={'analyte.copy.nbr_fmt':'Analyte Copy Number'})
            .sort_values('log10_SN1')
        )
    else:
        st.warning("No parameter sets found close to that target S/N.")

# ---------- 3D Visualization ----------
with st.expander("### 🌐 3D Parameter Space Visualization", expanded=False):
    custom_colorscale = [[0.0, '#FFDAB9'], [0.5, '#FF4500'], [1.0, '#800080']]
    for aff in AFFINITY_ORDERED:
        st.write(f"#### Affinity: {aff}")
        df_sub = df_combined[df_combined['Affinity_label']==aff].copy()
        df_sub['log10_analyte_copy_nbr'] = np.log10(df_sub['analyte.copy.nbr'].replace(0, np.nan))
        df_sub = df_sub.dropna(subset=['log10_analyte_copy_nbr'])
        if df_sub.empty:
            st.warning(f"No valid analyte copy numbers for {aff}")
            continue
        tick_vals = np.arange(int(np.floor(df_sub['log10_analyte_copy_nbr'].min())),
                              int(np.ceil(df_sub['log10_analyte_copy_nbr'].max()))+1)
        tick_texts = [f"10<sup>{i}</sup>" for i in tick_vals]
        fig = px.scatter_3d(df_sub,
                            x='log10_analyte_copy_nbr',
                            y='capture',
                            z='probe',
                            color='log10_SN1',
                            opacity=0.7,
                            color_continuous_scale=custom_colorscale,
                            labels={'log10_analyte_copy_nbr':'Analyte Copy Number (log10)',
                                    'capture':'Capture reagent conc. (µg/ml)',
                                    'probe':'Probe reagent conc. (µg/ml)',
                                    'log10_SN1':'log10(S/N)',
                                    'analyte.copy.nbr_fmt':'Analyte Copy Number'},
                            hover_data=['analyte.copy.nbr_fmt'])
        fig.update_layout(scene=dict(xaxis=dict(title='Analyte Copy Number', tickvals=tick_vals, ticktext=tick_texts)),
                          coloraxis_colorbar=dict(title=dict(text="log10(S/N)", font=dict(size=14)),
                                                  tickfont=dict(size=12), x=1.05, len=0.75, thickness=15),
                          margin=dict(l=0,r=0,b=0,t=30), showlegend=False)
        st.plotly_chart(fig)
