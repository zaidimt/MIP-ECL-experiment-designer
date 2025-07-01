import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import itertools

# Load data (cache)
@st.cache_data
def load_data():
    return pd.read_excel("pcp_with_affinity.xlsx")

df = load_data()

affinity_map = {'low': 59, 'medium': 13, 'high': 2}
df['Affinity_kD'] = df['Affinity'].map(affinity_map)

features = ['log10_cell.nbr', 'capture', 'probe', 'Affinity_kD']
X = df[features]
y = df['log10_SN1']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = GradientBoostingRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

cellnbr_vals = df['log10_cell.nbr'].unique()
capture_vals = df['capture'].unique()
probe_vals = df['probe'].unique()
affinity_vals = df['Affinity_kD'].unique()

param_grid = pd.DataFrame(
    list(itertools.product(cellnbr_vals, capture_vals, probe_vals, affinity_vals)),
    columns=features
)

inverse_affinity_map = {v: k for k, v in affinity_map.items()}
param_grid['Affinity'] = param_grid['Affinity_kD'].map(inverse_affinity_map)

df['key'] = df[features].astype(str).agg('-'.join, axis=1)
param_grid['key'] = param_grid[features].astype(str).agg('-'.join, axis=1)

missing_combos = param_grid[~param_grid['key'].isin(df['key'])].copy()
missing_combos['log10_SN1'] = model.predict(missing_combos[features])
missing_combos['source'] = 'predicted'

df['source'] = 'measured'
df_combined = pd.concat([df, missing_combos], ignore_index=True)

st.title("CRIS-CROS Experiment Designer")

with st.sidebar:
    st.subheader("🧠 Model Info & Performance")
    st.write("Model: Gradient Boosting Regressor")
    st.write(f"Test MSE: {mse:.4f}")
    st.write(f"Test R²: {r2:.4f}")
    st.info("Higher R² (closer to 1) indicates better prediction accuracy.")
    
    st.markdown("---")
    
    with st.container():
        st.header("🎛️ 1. Select Parameters to Predict S/N")
        cellnbr = st.select_slider("log10 (cell nbr)", options=sorted(df_combined['log10_cell.nbr'].unique()))
        capture = st.select_slider("Capture concentration (µg/ml)", options=sorted(df_combined['capture'].unique()))
        probe = st.select_slider("Probe concentration (µg/ml)", options=sorted(df_combined['probe'].unique()))
        affinity_label = st.select_slider("Affinity (kD)", options=['high', 'medium', 'low'], value='medium')
        kd = affinity_map[affinity_label]

    st.markdown("---")
    
    with st.container():
        st.header("🎯 2. Optimize for Target S/N")
        target_sn_linear = st.number_input("Target S/N value", value=10.0, step=0.1, min_value=0.01)
        target_sn = np.log10(target_sn_linear)
        affinity_levels = ['low', 'medium', 'high']
        selected_affinity = st.selectbox("Affinity (optional)", options=["any"] + affinity_levels)
        cellnbr_for_target = st.select_slider(
            "Cell number (log10) constraint (optional)", 
            options=["any"] + sorted(df_combined['log10_cell.nbr'].unique()), 
            value="any"
        )

# Main panel sections with emojis & clear headers
st.markdown("## 🎛️ Parameter Selection: Predicted S/N")
exact_match = df_combined[
    (np.isclose(df_combined['log10_cell.nbr'], cellnbr, rtol=0.01)) &
    (np.isclose(df_combined['capture'], capture, rtol=0.01)) &
    (np.isclose(df_combined['probe'], probe, rtol=0.01)) &
    (np.isclose(df_combined['Affinity_kD'], kd, rtol=0.01))
]

if not exact_match.empty:
    row = exact_match.iloc[0]
    sn_log = row['log10_SN1']
    sn_linear = 10**sn_log
    src = row['source']
    if src == 'predicted':
        st.info(f"Predicted log10(S/N): {sn_log:.2f} (model-generated)")
        st.info(f"Predicted S/N: {sn_linear:.2f}")
    else:
        st.success(f"Measured log10(S/N): {sn_log:.2f}")
        st.success(f"Measured S/N: {sn_linear:.2f}")
else:
    st.warning("No exact match found. Try adjusting the sliders.")

st.markdown("### Nearby Parameter Space (optional)")
tolerance = 0.3
nearby = df_combined[
    (np.isclose(df_combined['log10_cell.nbr'], cellnbr, rtol=tolerance)) &
    (np.isclose(df_combined['capture'], capture, rtol=tolerance)) &
    (np.isclose(df_combined['probe'], probe, rtol=tolerance)) &
    (np.isclose(df_combined['Affinity_kD'], kd, rtol=tolerance))
]

if not nearby.empty:
    st.dataframe(nearby[[
        'log10_cell.nbr', 'capture', 'probe', 'Affinity', 'log10_SN1', 'source'
    ]].sort_values('log10_SN1', ascending=False))
else:
    st.info("No nearby points found within tolerance.")

st.markdown("## 🎯 Target S/N Optimization")
if selected_affinity != "any":
    df_sn = df_combined[df_combined['Affinity'] == selected_affinity]
else:
    df_sn = df_combined.copy()

if cellnbr_for_target != "any":
    df_sn = df_sn[np.isclose(df_sn['log10_cell.nbr'], cellnbr_for_target, rtol=0.01)]

tolerance_sn = 0.1
matches = df_sn[np.isclose(df_sn['log10_SN1'], target_sn, atol=tolerance_sn)]

if not matches.empty:
    matches_display = matches.copy()
    matches_display['S/N'] = 10 ** matches_display['log10_SN1']
    st.dataframe(matches_display[[
        'log10_cell.nbr', 'capture', 'probe', 'Affinity', 'log10_SN1', 'S/N', 'source'
    ]].sort_values('log10_SN1'))
else:
    st.warning("No parameter sets found close to that target S/N. Try adjusting tolerance or constraints.")

st.markdown("## 📊 3D Parameter Space Visualization")

custom_colorscale = [
   [0.0, '#FFDAB9'],
   [0.5, '#FF4500'],
   [1.0, '#800080']
]

for aff in ['low', 'medium', 'high']:
    st.write(f"### Affinity: {aff.capitalize()}")
    df_sub = df_combined[df_combined['Affinity'] == aff]

    fig = px.scatter_3d(
        df_sub,
        x='log10_cell.nbr',
        y='capture',
        z='probe',
        color='log10_SN1',
        opacity=0.7,
        color_continuous_scale=custom_colorscale,
        labels={
            'log10_cell.nbr': 'log10 (cell nbr)',
            'capture': 'Capture (µg/ml)',
            'probe': 'Probe (µg/ml)',
            'log10_SN1': 'log10(S/N)'
        }
    )

    fig.update_layout(
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





