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
df['Affinity_kD'] = df['Affinity'].map(affinity_map)

# --- Define features and target ---
features = ['log10_cell.nbr', 'capture', 'probe', 'Affinity_kD']
X = df[features]
y = df['log10_SN1']

# --- Split data for evaluation ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train model ---
model = GradientBoostingRegressor()
model.fit(X_train, y_train)

# --- Predict on test set for evaluation ---
y_pred = model.predict(X_test)

# --- Calculate evaluation metrics ---
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# --- Generate full parameter grid for prediction ---
cellnbr_vals = df['log10_cell.nbr'].unique()
capture_vals = df['capture'].unique()
probe_vals = df['probe'].unique()
affinity_vals = df['Affinity_kD'].unique()

param_grid = pd.DataFrame(
    list(itertools.product(cellnbr_vals, capture_vals, probe_vals, affinity_vals)),
    columns=features
)

# --- Add Affinity label to param_grid for plotting ---
inverse_affinity_map = {v: k for k, v in affinity_map.items()}
param_grid['Affinity'] = param_grid['Affinity_kD'].map(inverse_affinity_map)

# --- Tag existing combos ---
df['key'] = df[features].astype(str).agg('-'.join, axis=1)
param_grid['key'] = param_grid[features].astype(str).agg('-'.join, axis=1)

# --- Predict missing combos ---
missing_combos = param_grid[~param_grid['key'].isin(df['key'])].copy()
missing_combos['log10_SN1'] = model.predict(missing_combos[features])
missing_combos['source'] = 'predicted'

df['source'] = 'measured'
df_combined = pd.concat([df, missing_combos], ignore_index=True)

# --- Streamlit app layout ---
st.title("CRIS-CROS Experiment Designer")

# Show model evaluation metrics in sidebar
st.sidebar.subheader("Model Performance")
st.sidebar.write(f"Test MSE: {mse:.4f}")
st.sidebar.write(f"Test R²: {r2:.4f}")
st.sidebar.info("Higher R² (closer to 1) indicates better prediction accuracy.")

# Sidebar filters
st.sidebar.header("Select Parameters")
cellnbr = st.sidebar.select_slider("log10 (cell nbr)", options=sorted(df_combined['log10_cell.nbr'].unique()))
capture = st.sidebar.select_slider("Capture concentration (µg/ml)", options=sorted(df_combined['capture'].unique()))
probe = st.sidebar.select_slider("Probe concentration (µg/ml)", options=sorted(df_combined['probe'].unique()))
affinity_label = st.sidebar.select_slider("Affinity (kD)", options=['high', 'medium', 'low'], value='medium')
kd = affinity_map[affinity_label]

# --- Exact match filtering ---
exact_match = df_combined[
    (np.isclose(df_combined['log10_cell.nbr'], cellnbr, rtol=0.01)) &
    (np.isclose(df_combined['capture'], capture, rtol=0.01)) &
    (np.isclose(df_combined['probe'], probe, rtol=0.01)) &
    (np.isclose(df_combined['Affinity_kD'], kd, rtol=0.01))
]

st.subheader("Predicted log10(S/N):")
if not exact_match.empty:
    row = exact_match.iloc[0]
    sn = row['log10_SN1']
    src = row['source']
    if src == 'predicted':
        st.info(f"Predicted log10(S/N): {sn:.2f} (model-generated)")
    else:
        st.success(f"Measured log10(S/N): {sn:.2f}")
else:
    st.warning("No match found.")

# --- Nearby parameter space ---
st.subheader("Nearby parameter space (optional)")
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

# --- 3D visualization ---
st.subheader("3D Parameter Space Visualization")

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