import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import itertools

# --- Load data ---
@st.cache_data(ttl=600)
def load_data():
    return pd.read_excel("pcp_with_affinity.xlsx")

df = load_data()

# --- Affinity mapping ---
affinity_map = {'low': 59, 'medium': 13, 'high': 2}
affinity_levels = ['low', 'medium', 'high']
df['Affinity_kD'] = df['Affinity'].map(affinity_map)

# --- Model training ---
features = ['log10_cell.nbr', 'capture', 'probe', 'Affinity_kD']
X = df[features]
y = df['log10_SN1']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = GradientBoostingRegressor()
model.fit(X_train, y_train)

# --- Evaluation ---
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# --- Full parameter space ---
cellnbr_vals = df['log10_cell.nbr'].unique()
capture_vals = df['capture'].unique()
probe_vals = df['probe'].unique()
affinity_vals = df['Affinity_kD'].unique()

param_grid = pd.DataFrame(
    list(itertools.product(cellnbr_vals, capture_vals, probe_vals, affinity_vals)),
    columns=features
)

# --- Inverse map for labels ---
inverse_affinity_map = {v: k for k, v in affinity_map.items()}
param_grid['Affinity'] = param_grid['Affinity_KD'] = param_grid['Affinity_kD'].map(inverse_affinity_map)

# --- Identify missing combinations ---
df['key'] = df[features].astype(str).agg('-'.join, axis=1)
param_grid['key'] = param_grid[features].astype(str).agg('-'.join, axis=1)

missing_combos = param_grid[~param_grid['key'].isin(df['key'])].copy()
missing_combos['log10_SN1'] = model.predict(missing_combos[features])
missing_combos['source'] = 'predicted'

df['source'] = 'measured'
df_combined = pd.concat([df, missing_combos], ignore_index=True)

# --- Streamlit app layout ---
st.title("CRIS-CROS Experiment Designer")

# Sidebar: Model performance
st.sidebar.subheader("Model Performance")
st.sidebar.write(f"Test MSE: {mse:.4f}")
st.sidebar.write(f"Test R²: {r2:.4f}")
st.sidebar.info("Higher R² (closer to 1) means better model prediction.")

# Sidebar: Parameter selectors
st.sidebar.header("Select Parameters")
cellnbr = st.sidebar.select_slider("log10 (cell nbr)", options=sorted(df_combined['log10_cell.nbr'].unique()))
capture = st.sidebar.select_slider("Capture concentration (µg/ml)", options=sorted(df_combined['capture'].unique()))
probe = st.sidebar.select_slider("Probe concentration (µg/ml)", options=sorted(df_combined['probe'].unique()))
affinity_label = st.sidebar.select_slider("Affinity (kD)", options=affinity_levels, value='medium')
kd = affinity_map[affinity_label]

# --- Exact match prediction ---
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

# --- Nearby space ---
st.subheader("Nearby Parameter Space (Optional)")
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

# --- Target SN optimization ---
st.subheader("Optimize by Target log10(S/N)")

target_sn = st.sidebar.number_input("Target log10(S/N) value", value=1.0, step=0.1)
tolerance_sn = st.sidebar.slider("SN Match Tolerance", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
selected_affinity = st.sidebar.selectbox("Affinity (optional)", options=["any"] + affinity_levels)

if selected_affinity != "any":
    df_sn = df_combined[df_combined['Affinity'] == selected_affinity]
else:
    df_sn = df_combined.copy()

matches = df_sn[np.isclose(df_sn['log10_SN1'], target_sn, atol=tolerance_sn)]

if not matches.empty:
    st.success(f"Parameter combinations near log10(S/N) = {target_sn}:")
    st.dataframe(matches[[
        'log10_cell.nbr', 'capture', 'probe', 'Affinity', 'log10_SN1', 'source'
    ]].sort_values('log10_SN1'))
    st.download_button("Download matches as CSV", matches.to_csv(index=False), file_name="target_sn_matches.csv")
else:
    st.warning("No combinations found close to that SN target.")

# --- 3D plot ---
st.subheader("3D Parameter Space Visualization")

custom_colorscale = [
   [0.0, '#FFDAB9'],  # Light orange (peach puff)
   [0.5, '#FF4500'],  # Red-orange
   [1.0, '#800080']   # Purple
]

for aff in affinity_levels:
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
