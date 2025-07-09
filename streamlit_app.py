import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import itertools

# --- Load data ---
@st.cache_data
def load_data():
    return pd.read_excel("pcp_with_affinity.xlsx")

df = load_data()

# --- Map Affinity to numeric ---
affinity_map = {'low': 59, 'medium': 13, 'high': 2}
df['Affinity_KD'] = df['Affinity'].map(affinity_map)

# --- Define features and target ---
features = ['protein.copy.nbr', 'capture', 'probe', 'Affinity_KD']
X = df[features]
y = df['log10_SN1']

# --- Split data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = GradientBoostingRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# --- Generate parameter grid for prediction ---
cellnbr_vals = df['protein.copy.nbr'].unique()
capture_vals = df['capture'].unique()
probe_vals = df['probe'].unique()
affinity_vals = df['Affinity_KD'].unique()

param_grid = pd.DataFrame(
    list(itertools.product(cellnbr_vals, capture_vals, probe_vals, affinity_vals)),
    columns=features
)

inverse_affinity_map = {v: k for k, v in affinity_map.items()}
param_grid['Affinity'] = param_grid['Affinity_KD'].map(inverse_affinity_map)

df['key'] = df[features].astype(str).agg('-'.join, axis=1)
param_grid['key'] = param_grid[features].astype(str).agg('-'.join, axis=1)

missing_combos = param_grid[~param_grid['key'].isin(df['key'])].copy()
missing_combos['log10_SN1'] = model.predict(missing_combos[features])
missing_combos['source'] = 'predicted'

df['source'] = 'measured'
df_combined = pd.concat([df, missing_combos], ignore_index=True)

# --- FORMAT COLUMN FOR SCIENTIFIC NOTATION ---
df_combined['protein.copy.nbr_fmt'] = df_combined['protein.copy.nbr'].apply(lambda x: f"{x:.0e}")

# --- Streamlit layout ---
st.set_page_config(page_title="CRIS-CROS Experiment Designer", layout="wide")
st.title("🧪 CRIS-CROS Experiment Designer")

# --- Sidebar Inputs ---
st.sidebar.markdown("## 🔧 Input Panel")

with st.sidebar.expander("🎯 Predict S/N from known parameters", expanded=True):
    # Format dropdowns with scientific notation
    unique_cells = sorted(df_combined['protein.copy.nbr'].unique())
    cell_format_map = {f"{x:.0e}": x for x in unique_cells}
    cellnbr_fmt = st.select_slider("protein copy nbr", options=list(cell_format_map.keys()))
    cellnbr = cell_format_map[cellnbr_fmt]

    capture = st.select_slider("Capture concentration (µg/ml)", options=sorted(df_combined['capture'].unique()))
    probe = st.select_slider("Probe concentration (µg/ml)", options=sorted(df_combined['probe'].unique()))
    affinity_label = st.select_slider("Affinity (kD)", options=['high', 'medium', 'low'], value='medium')
    kd = affinity_map[affinity_label]

with st.sidebar.expander("📈 Optimize parameters for target S/N", expanded=False):
    target_sn_linear = st.number_input("Target S/N value", value=10.0, step=1.0)
    target_sn = np.log10(target_sn_linear)

    affinity_levels = ['low', 'medium', 'high']
    selected_affinity = st.selectbox("Affinity", options=["any"] + affinity_levels)

    all_cellnrs = sorted(df_combined['protein.copy.nbr'].unique())
    cell_map = {f"{x:.0e}": x for x in all_cellnrs}
    cellnbr_target_fmt = st.select_slider("Cell number", options=["any"] + list(cell_map.keys()), value="any")
    cellnbr_target = cell_map[cellnbr_target_fmt] if cellnbr_target_fmt != "any" else "any"

# --- Model Info ---
st.markdown("## 🧠 Model Info")
st.info("Using **Gradient Boosting Regressor** for prediction")
st.write(f"**Test MSE**: {mse:.4f} | **Test R²**: {r2:.4f}")

# --- Predicted S/N ---
with st.expander("### 🔍 Predicted S/N based on selected parameters", expanded=True):
    exact_match = df_combined[
        (np.isclose(df_combined['protein.copy.nbr'], cellnbr, rtol=0.01)) &
        (np.isclose(df_combined['capture'], capture, rtol=0.01)) &
        (np.isclose(df_combined['probe'], probe, rtol=0.01)) &
        (np.isclose(df_combined['Affinity_KD'], kd, rtol=0.01))
    ]

    if not exact_match.empty:
        row = exact_match.iloc[0]
        sn = row['log10_SN1']
        src = row['source']
        st.metric(label="log10(S/N)", value=f"{sn:.2f}")
        st.metric(label="S/N", value=f"{10**sn:.2f}")
        st.caption(f"Source: {'predicted' if src == 'predicted' else 'measured'}")
    else:
        st.warning("No exact match found.")

    st.markdown("#### 🧭 Nearby parameter space")
    tolerance = 0.3
    nearby = df_combined[
        (np.isclose(df_combined['protein.copy.nbr'], cellnbr, rtol=tolerance)) &
        (np.isclose(df_combined['capture'], capture, rtol=tolerance)) &
        (np.isclose(df_combined['probe'], probe, rtol=tolerance)) &
        (np.isclose(df_combined['Affinity_KD'], kd, rtol=tolerance))
    ]

    if not nearby.empty:
        nearby = nearby.copy()
        nearby['S/N'] = 10 ** nearby['log10_SN1']
        st.dataframe(nearby.rename(columns={'protein.copy.nbr_fmt': 'protein.copy.nbr'})[[
            'protein.copy.nbr_fmt', 'capture', 'probe', 'Affinity', 'log10_SN1', 'S/N', 'source'
        ]].sort_values('log10_SN1', ascending=False).rename(columns={'protein.copy.nbr_fmt': 'Protein Copy Nbr'}))
    else:
        st.info("No nearby points found within tolerance.")

# --- Optimization section ---
with st.expander("### 📈 Optimized parameters for target S/N", expanded=False):
    df_sn = df_combined.copy()
    if selected_affinity != "any":
        df_sn = df_sn[df_sn['Affinity'] == selected_affinity]
    if cellnbr_target != "any":
        df_sn = df_sn[np.isclose(df_sn['protein.copy.nbr'], float(cellnbr_target), rtol=0.01)]

    tolerance_sn = 0.1
    matches = df_sn[np.isclose(df_sn['log10_SN1'], target_sn, atol=tolerance_sn)]

    if not matches.empty:
        matches = matches.copy()
        matches['S/N'] = 10 ** matches['log10_SN1']
        st.success(f"Parameter combinations close to target S/N ≈ {target_sn_linear:.2f}:")
        st.dataframe(matches.rename(columns={'protein.copy.nbr_fmt': 'protein.copy.nbr'})[[
            'protein.copy.nbr_fmt', 'capture', 'probe', 'Affinity', 'log10_SN1', 'S/N', 'source'
        ]].sort_values('log10_SN1').rename(columns={'protein.copy.nbr_fmt': 'Protein Copy Nbr'}))
    else:
        st.warning("No parameter sets found close to that target S/N. Try adjusting inputs or tolerance.")

# --- 3D Visualization ---
with st.expander("### 🌐 3D Parameter Space Visualization", expanded=False):
    custom_colorscale = [
       [0.0, '#FFDAB9'],
       [0.5, '#FF4500'],
       [1.0, '#800080']
    ]

    for aff in ['low', 'medium', 'high']:
        st.write(f"#### Affinity: {aff.capitalize()}")
        df_sub = df_combined[df_combined['Affinity'] == aff].copy()

        df_sub = df_sub[df_sub['protein.copy.nbr'] > 0]
        df_sub['log10_protein_copy_nbr'] = np.log10(df_sub['protein.copy.nbr'])

        tick_vals = np.arange(
            int(df_sub['log10_protein_copy_nbr'].min()),
            int(df_sub['log10_protein_copy_nbr'].max()) + 1
        )
        tick_texts = [f"10<sup>{i}</sup>" for i in tick_vals]

        fig = px.scatter_3d(
            df_sub,
            x='log10_protein_copy_nbr',
            y='capture',
            z='probe',
            color='log10_SN1',
            opacity=0.7,
            color_continuous_scale=custom_colorscale,
            labels={
                'log10_protein_copy_nbr': 'Protein Copy Nbr (log10)',
                'capture': 'Capture (µg/ml)',
                'probe': 'Probe (µg/ml)',
                'log10_SN1': 'log10(S/N)',
                'protein.copy.nbr_fmt': 'Protein Copy Nbr'
            },
            hover_data=['protein.copy.nbr_fmt']
        )

        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    title='Protein Copy Nbr',
                    tickvals=tick_vals,
                    ticktext=tick_texts
                )
            ),
            coloraxis_colorbar=dict(
                title=dict(text="log10(S/N)", font=dict(size=14)),
                tickfont=dict(size=12),
                x=1.05,
                len=0.75,
                thickness=15
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            showlegend=False
        )

        st.plotly_chart(fig)