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
    return pd.read_excel("App_data.xlsx")

df = load_data()

# --- Map Affinity to numeric (ordered: Low -> Medium -> High) ---
affinity_map = {
    'low (≤2 kD*)': 2,
    'medium (13 kD*)': 13,
    'high (≥59 kD*)': 59
}
df['Affinity_KD'] = df['Affinity'].map({'low': 2, 'medium': 13, 'high': 59})

# --- Define features and target ---
features = ['protein.copy.nbr', 'capture', 'probe', 'Affinity_KD']
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
analyte_vals = df['protein.copy.nbr'].unique()
capture_vals = df['capture'].unique()
probe_vals = df['probe'].unique()
affinity_vals = df['Affinity_KD'].unique()

param_grid = pd.DataFrame(
    list(itertools.product(analyte_vals, capture_vals, probe_vals, affinity_vals)),
    columns=features
)

# --- Add Affinity label to param_grid for plotting ---
inverse_affinity_map = {2: 'low (≤2 kD*)', 13: 'medium (13 kD*)', 59: 'high (≥59 kD*)'}
param_grid['Affinity'] = param_grid['Affinity_KD'].map(inverse_affinity_map)

# --- Tag existing combos ---
df['key'] = df[features].astype(str).agg('-'.join, axis=1)
param_grid['key'] = param_grid[features].astype(str).agg('-'.join, axis=1)

# --- Predict missing combos ---
missing_combos = param_grid[~param_grid['key'].isin(df['key'])].copy()
missing_combos['log10_SN1'] = model.predict(missing_combos[features])
missing_combos['source'] = 'predicted'

df['source'] = 'measured'
df_combined = pd.concat([df, missing_combos], ignore_index=True)

# --- Format analyte copy number ---
df_combined['analyte.copy.nbr_fmt'] = df_combined['protein.copy.nbr'].apply(lambda x: f"{x:.0e}")

# --- Streamlit app layout ---
st.set_page_config(page_title="MIP-ECL Experiment Designer", layout="wide")

st.title("🔬 MIP-ECL EXPERIMENT DESIGNER")

st.markdown(
    """
    This interactive tool helps design **MIP-ECL experiments** by predicting 
    the expected signal-to-noise ratio (S/N) from four key parameters:
    - Analyte copy number  
    - Capture reagent concentration  
    - Probe reagent concentration  
    - Binding affinity (expressed as equilibrium dissociation constant, kD*)  

    Use the panels on the left to predict outcomes for specific conditions or 
    explore optimized parameter combinations.  
    """
)

# --- Sidebar: Inputs ---
st.sidebar.markdown("## 🔧 Input Panel")

with st.sidebar.expander("🎯 Predict S/N from known parameters", expanded=True):
    unique_analytes = sorted(df_combined['protein.copy.nbr'].unique())
    analyte_format_map = {f"{x:.0e}": x for x in unique_analytes}
    analyte_fmt = st.select_slider("Analyte copy number", options=list(analyte_format_map.keys()))
    analyte = analyte_format_map[analyte_fmt]

    capture = st.select_slider("Capture reagent conc. (µg/ml)", options=sorted(df_combined['capture'].unique()))
    probe = st.select_slider("Probe reagent concentration (µg/ml)", options=sorted(df_combined['probe'].unique()))
    affinity_ordered = ['low (≤2 kD*)', 'medium (13 kD*)', 'high (≥59 kD*)']
    affinity_label = st.select_slider("Affinity", options=affinity_ordered, value='medium (13 kD*)')
    kd = affinity_map[affinity_label]

with st.sidebar.expander("📈 Optimize parameters for target S/N", expanded=False):
    target_sn_linear = st.number_input("Target S/N value", value=10.0, step=1.0)
    target_sn = np.log10(target_sn_linear)

    affinity_levels = ['low (≤2 kD*)', 'medium (13 kD*)', 'high (≥59 kD*)']
    selected_affinity = st.selectbox("Affinity", options=["any"] + affinity_levels)

    all_analytes = sorted(df_combined['protein.copy.nbr'].unique())
    analyte_map = {f"{x:.0e}": x for x in all_analytes}
    analyte_target_fmt = st.select_slider("Analyte copy number", options=["any"] + list(analyte_map.keys()), value="any")
    analyte_target = analyte_map[analyte_target_fmt] if analyte_target_fmt != "any" else "any"

# --- Main: Model Info ---
st.markdown("## 🧠 Prediction Model")
st.info(
    f"This app uses a **Gradient Boosting Regressor** to predict the signal-to-noise ratio (S/N) "
    f"from analyte copy number, capture reagent concentration, probe reagent concentration, and binding affinity.\n\n"
    f"🔹 **Important:** Predictions are only made for parameter combinations that were **not measured experimentally**. "
    f"Measured values always take precedence, and the app labels each data point as either:\n"
    f"- **measured** (experimental data)\n"
    f"- **predicted** (model estimate)\n\n"
    f"Model performance on test data:\n"
    f"- **Coefficient of Determination (R²):** {r2:.3f}\n"
    f"- **Mean Squared Error (MSE):** {mse:.4f}"
)

# --- Main: S/N prediction ---
with st.expander("### 🔍 Predicted S/N based on selected parameters", expanded=True):
    exact_match = df_combined[
        (np.isclose(df_combined['protein.copy.nbr'], analyte, rtol=0.01)) &
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
        st.caption(f"Source: {'Predicted by model' if src == 'predicted' else 'Measured experimentally'}")
    else:
        st.warning("No exact match found.")

    st.markdown("#### 🧭 Nearby parameter space")
    tolerance = 0.3
    nearby = df_combined[
        (np.isclose(df_combined['protein.copy.nbr'], analyte, rtol=tolerance)) &
        (np.isclose(df_combined['capture'], capture, rtol=tolerance)) &
        (np.isclose(df_combined['probe'], probe, rtol=tolerance)) &
        (np.isclose(df_combined['Affinity_KD'], kd, rtol=tolerance))
    ]

    if not nearby.empty:
        nearby = nearby.copy()
        nearby['S/N'] = 10 ** nearby['log10_SN1']
        st.dataframe(
            nearby[[
                'analyte.copy.nbr_fmt', 'capture', 'probe', 'Affinity', 'log10_SN1', 'S/N', 'source'
            ]].rename(columns={'analyte.copy.nbr_fmt': 'Analyte Copy Number'}).sort_values('log10_SN1', ascending=False)
        )
    else:
        st.info("No nearby points found within tolerance.")

# --- Main: Optimization section ---
with st.expander("### 📈 Optimized parameters for target S/N", expanded=False):
    df_sn = df_combined.copy()
    if selected_affinity != "any":
        df_sn = df_sn[df_sn['Affinity'] == selected_affinity]
    if analyte_target != "any":
        df_sn = df_sn[np.isclose(df_sn['protein.copy.nbr'], float(analyte_target), rtol=0.01)]

    tolerance_sn = 0.1
    matches = df_sn[np.isclose(df_sn['log10_SN1'], target_sn, atol=tolerance_sn)]

    if not matches.empty:
        matches = matches.copy()
        matches['S/N'] = 10 ** matches['log10_SN1']
        st.success(f"Parameter combinations close to target S/N ≈ {target_sn_linear:.2f}:")
        st.dataframe(
            matches[[
                'analyte.copy.nbr_fmt', 'capture', 'probe', 'Affinity', 'log10_SN1', 'S/N', 'source'
            ]].rename(columns={'analyte.copy.nbr_fmt': 'Analyte Copy Number'}).sort_values('log10_SN1')
        )
    else:
        st.warning("No parameter sets found close to that target S/N. Try adjusting inputs or tolerance.")

# --- Main: 3D visualization ---
with st.expander("### 🌐 3D Parameter Space Visualization", expanded=False):
    custom_colorscale = [
       [0.0, '#FFDAB9'],
       [0.5, '#FF4500'],
       [1.0, '#800080']
    ]

    for aff in ['low (≤2 kD*)', 'medium (13 kD*)', 'high (≥59 kD*)']:
        st.write(f"#### Affinity: {aff.capitalize()}")
        df_sub = df_combined[df_combined['Affinity'] == aff].copy()

        df_sub = df_sub[df_sub['protein.copy.nbr'] > 0]
        df_sub['log10_analyte_copy_nbr'] = np.log10(df_sub['protein.copy.nbr'])

        tick_vals = np.arange(
            int(df_sub['log10_analyte_copy_nbr'].min()),
            int(df_sub['log10_analyte_copy_nbr'].max()) + 1
        )
        tick_texts = [f"10<sup>{i}</sup>" for i in tick_vals]

        fig = px.scatter_3d(
            df_sub,
            x='log10_analyte_copy_nbr',
            y='capture',
            z='probe',
            color='log10_SN1',
            opacity=0.7,
            color_continuous_scale=custom_colorscale,
            labels={
                'log10_analyte_copy_nbr': 'Analyte Copy Number (log10)',
                'capture': 'Capture reagent conc. (µg/ml)',
                'probe': 'Probe reagent concentration (µg/ml)',
                'log10_SN1': 'log10(S/N)',
                'analyte.copy.nbr_fmt': 'Analyte Copy Number'
            },
            hover_data=['analyte.copy.nbr_fmt']
        )

        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    title='Analyte Copy Number',
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
            margin=dict(l=0, r=0, b=0, t=30)
        )

        st.plotly_chart(fig)