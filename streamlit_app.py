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
    'low':   'low (≤2 K<sub>D</sub><sup>*</sup>)',
    'medium':'medium (K<sub>D</sub><sup>*</sup> = 13)',
    'high':  'high (≥59 K<sub>D</sub><sup>*</sup>)'
}
AFF_KD_FROM_SIMPLE = {'low': 2, 'medium': 13, 'high': 59}
AFF_KD_FROM_LABELED = {
    'low (≤2 K<sub>D</sub><sup>*</sup>)': 2,
    'medium (K<sub>D</sub><sup>*</sup> = 13)': 13,
    'high (≥59 K<sub>D</sub><sup>*</sup>)': 59
}
KD_TO_LABELED = {
    2: 'low (≤2 K<sub>D</sub><sup>*</sup>)',
    13: 'medium (K<sub>D</sub><sup>*</sup> = 13)',
    59: 'high (≥59 K<sub>D</sub><sup>*</sup>)'
}
AFFINITY_ORDERED = [
    'low (≤2 K<sub>D</sub><sup>*</sup>)',
    'medium (K<sub>D</sub><sup>*</sup> = 13)',
    'high (≥59 K<sub>D</sub><sup>*</sup>)'
]

# ---------- Load data ----------
@st.cache_data
def load_data():
    df = pd.read_excel("App_data.xlsx")

    if "protein.copy.nbr" in df.columns:
        df.rename(columns={"protein.copy.nbr": "analyte.copy.nbr"}, inplace=True)

    if "Affinity" in df.columns:
        df["Affinity_label"] = df["Affinity"].map(AFF_LABELED)
        df["Affinity_KD"] = df["Affinity"].map(AFF_KD_FROM_SIMPLE)
    else:
        raise ValueError("Missing 'Affinity' column with values low|medium|high.")

    # Format analyte copy number in scientific notation
    df["analyte.copy.nbr_fmt"] = df["analyte.copy.nbr"].apply(
        lambda x: f"{x:.0e}" if pd.notna(x) and x > 0 else "0"
    )
    return df

df = load_data()

# ---------- Features & model ----------
features = ['analyte.copy.nbr', 'capture', 'probe', 'Affinity_KD']
y = df['log10_SN1']
X = df[features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = GradientBoostingRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# ---------- Prediction grid ----------
analyte_vals = df['analyte.copy.nbr'].unique()
capture_vals = df['capture'].unique()
probe_vals = df['probe'].unique()
affinity_vals = df['Affinity_KD'].unique()

param_grid = pd.DataFrame(
    list(itertools.product(analyte_vals, capture_vals, probe_vals, affinity_vals)),
    columns=features
)
param_grid['Affinity_label'] = param_grid['Affinity_KD'].map(KD_TO_LABELED)

df['key'] = df[features].astype(str).agg('-'.join, axis=1)
param_grid['key'] = param_grid[features].astype(str).agg('-'.join, axis=1)

missing_combos = param_grid[~param_grid['key'].isin(df['key'])].copy()
missing_combos['log10_SN1'] = model.predict(missing_combos[features])
missing_combos['source'] = 'predicted'

df_measured = df.copy()
df_measured['source'] = 'measured'
if 'Affinity_label' not in df_measured.columns:
    df_measured['Affinity_label'] = df_measured['Affinity_KD'].map(KD_TO_LABELED)

df_combined = pd.concat([df_measured, missing_combos], ignore_index=True)

# ✅ Always recompute analyte copy number formatting
df_combined['analyte.copy.nbr_fmt'] = df_combined['analyte.copy.nbr'].apply(
    lambda x: f"{x:.0e}" if pd.notna(x) and x > 0 else "0"
)

# ---------- Streamlit layout ----------
st.set_page_config(page_title="MIP-ECL Experiment Designer", layout="wide")
st.title("🔬 MIP-ECL EXPERIMENT DESIGNER")

st.markdown(
    """
    This interactive tool helps design **MIP-ECL experiments** by predicting 
    the expected signal-to-noise ratio (S/N) from four key parameters:
    - Analyte copy number  
    - Capture reagent concentration  
    - Probe reagent concentration  
    - Binding affinity (expressed as equilibrium dissociation constant, K<sub>D</sub><sup>*</sup>)  
    """,
    unsafe_allow_html=True
)

# ---------- Sidebar inputs ----------
st.sidebar.markdown("## 🔧 Input Panel")

with st.sidebar.expander("🎯 Predict S/N", expanded=True):
    unique_analytes = sorted(df_combined['analyte.copy.nbr'].unique())
    analyte_format_map = {f"{x:.0e}": x for x in unique_analytes}
    analyte_fmt = st.select_slider("Analyte copy number", options=list(analyte_format_map.keys()))
    analyte = analyte_format_map[analyte_fmt]

    capture = st.select_slider("Capture reagent conc. (µg/ml)", options=sorted(df_combined['capture'].unique()))
    probe = st.select_slider("Probe reagent conc. (µg/ml)", options=sorted(df_combined['probe'].unique()))
    affinity_label = st.select_slider("Affinity", options=AFFINITY_ORDERED, value=AFFINITY_ORDERED[1])
    kd = AFF_KD_FROM_LABELED[affinity_label]

# ---------- Model info ----------
st.markdown("## 🧠 Prediction Model")
st.info(
    f"Gradient Boosting Regressor performance:\n\n"
    f"- R² = {r2:.3f}\n"
    f"- MSE = {mse:.4f}\n\n"
    f"Measured values take precedence. Model only predicts missing combinations."
)

# ---------- 3D Visualization ----------
with st.expander("### 🌐 3D Parameter Space Visualization", expanded=False):
    custom_colorscale = [
        [0.0, '#FFDAB9'],
        [0.5, '#FF4500'],
        [1.0, '#800080']
    ]

    for aff in AFFINITY_ORDERED:
        st.markdown(f"#### Affinity: {aff}", unsafe_allow_html=True)
        df_sub = df_combined[df_combined['Affinity_label'] == aff].copy()

        df_sub['log10_analyte_copy_nbr'] = np.where(
            df_sub['analyte.copy.nbr'] > 0,
            np.log10(df_sub['analyte.copy.nbr']),
            np.nan
        )
        df_sub = df_sub.replace([np.inf, -np.inf], np.nan).dropna(subset=['log10_analyte_copy_nbr'])

        if df_sub.empty:
            st.warning(f"No valid analyte copy numbers for {aff}")
            continue

        tick_vals = np.arange(
            int(np.floor(df_sub['log10_analyte_copy_nbr'].min())),
            int(np.ceil(df_sub['log10_analyte_copy_nbr'].max())) + 1
        )

        sn_ticks = np.arange(
            int(np.floor(df_sub['log10_SN1'].min())),
            int(np.ceil(df_sub['log10_SN1'].max())) + 1
        )

        fig = px.scatter_3d(
            df_sub,
            x='log10_analyte_copy_nbr',
            y='capture',
            z='probe',
            color='log10_SN1',
            opacity=0.7,
            color_continuous_scale=custom_colorscale,
            labels={
                'log10_analyte_copy_nbr': 'Analyte Copy Number',
                'capture': 'Capture conc. (µg/ml)',
                'probe': 'Probe conc. (µg/ml)',
                'log10_SN1': 'log10(S/N)'
            },
            hover_data={'analyte.copy.nbr': ':.0e'}
        )

        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    tickmode='array',
                    tickvals=tick_vals.tolist(),
                    ticktext=[f"1e{int(i)}" for i in tick_vals],
                    title="Analyte Copy Number"
                ),
                yaxis=dict(title="Capture conc. (µg/ml)"),
                zaxis=dict(title="Probe conc. (µg/ml)")
            ),
            coloraxis_colorbar=dict(
                title="S/N",
                tickvals=sn_ticks.tolist(),
                ticktext=[f"{10**int(i):.0f}" for i in sn_ticks]
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            showlegend=False
        )

        st.plotly_chart(fig)
