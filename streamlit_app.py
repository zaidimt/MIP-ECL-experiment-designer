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
    'low':   '(kD* ≥ 53.6)',
    'medium':'(kD* = 13)',
    'high':  '(kD* ≤ 1.3)'
}
AFF_KD_FROM_SIMPLE = {'low': 53.6, 'medium': 13, 'high': 1.3}
AFF_KD_FROM_LABELED = {'(kD* ≥ 53.6)': 53.6, '(kD* = 13)': 13, '(kD* ≤ 1.3)': 1.3}
KD_TO_LABELED = {53.6: '(kD* ≥ 53.6)', 13: 'medium (kD* = 13)', 1.3: '(kD* ≤ 1.3)'}

AFFINITY_ORDERED = ['(kD* ≤ 1.3)', '(kD* = 13)', '(kD* ≥ 53.6))']

# ---------- Load data ----------
@st.cache_data
def load_data():
    df = pd.read_excel("App_data.xlsx")

    # Rename once for consistency
    if "protein.copy.nbr" in df.columns:
        df.rename(columns={"protein.copy.nbr": "analyte.copy.nbr"}, inplace=True)

    # Create canonical affinity fields for measured data
    # Expecting df['Affinity'] to be 'low'|'medium'|'high'
    if "Affinity" in df.columns:
        df["Affinity_label"] = df["Affinity"].map(AFF_LABELED)
        df["Affinity_KD"] = df["Affinity"].map(AFF_KD_FROM_SIMPLE)
    else:
        raise ValueError("Missing 'Affinity' column with values low|medium|high.")

    # Prettify analyte copy number for hover
    df["analyte.copy.nbr_fmt"] = df["analyte.copy.nbr"].apply(
        lambda x: f"{int(x):,}" if pd.notna(x) and x > 0 else "0"
    )
    return df

df = load_data()

# ---------- Define features & target ----------
features = ['analyte.copy.nbr', 'capture', 'probe', 'Affinity_KD']
y = df['log10SN']
X = df[features]

# ---------- Train/test split & model ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = GradientBoostingRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# ---------- Ensure measured data has numeric Affinity_KD ----------
df_measured = df.copy()
df_measured['Affinity_KD'] = df_measured['Affinity'].map(AFF_KD_FROM_SIMPLE)
df_measured['Affinity_label'] = df_measured['Affinity'].map(AFF_LABELED)
df_measured['source'] = 'measured'

# ---------- Generate full parameter grid for prediction ----------
analyte_vals = df['analyte.copy.nbr'].unique()
capture_vals = df['capture'].unique()
probe_vals = df['probe'].unique()
affinity_vals = df['Affinity_KD'].unique()  # numeric: 2, 13, 59

param_grid = pd.DataFrame(
    list(itertools.product(analyte_vals, capture_vals, probe_vals, affinity_vals)),
    columns=features
)
# Add canonical label to the grid
param_grid['Affinity_label'] = param_grid['Affinity_KD'].map(KD_TO_LABELED)

# ---------- Function to check if a row is already measured ----------
def is_missing(row, df_measured):
    return not ((np.isclose(df_measured['analyte.copy.nbr'], row['analyte.copy.nbr'])) &
                (np.isclose(df_measured['capture'], row['capture'])) &
                (np.isclose(df_measured['probe'], row['probe'])) &
                (df_measured['Affinity_KD'] == row['Affinity_KD'])).any()

# ---------- Select missing rows ----------
missing_combos = param_grid[param_grid.apply(lambda r: is_missing(r, df_measured), axis=1)].copy()

# ---------- Predict log10SN for missing combos ----------
missing_combos['log10SN'] = model.predict(missing_combos[features])
missing_combos['source'] = 'predicted'

# ---------- Format analyte copy number for display ----------
missing_combos['analyte.copy.nbr_fmt'] = missing_combos['analyte.copy.nbr'].apply(
    lambda x: f"{int(x):,}" if pd.notna(x) and x > 0 else "0"
)
df_measured['analyte.copy.nbr_fmt'] = df_measured['analyte.copy.nbr'].apply(
    lambda x: f"{int(x):,}" if pd.notna(x) and x > 0 else "0"
)

# ---------- Combine measured + predicted ----------
df_combined = pd.concat([df_measured, missing_combos], ignore_index=True)

# ---------- App layout ----------
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

# ---------- Sidebar: Inputs ----------
st.sidebar.markdown("## 🔧 Input Panel")

with st.sidebar.expander("🎯 Predict S/N from known parameters", expanded=True):
    unique_analytes = sorted(df_combined['analyte.copy.nbr'].unique())
    analyte_format_map = {f"{x:.0e}": x for x in unique_analytes}
    analyte_fmt = st.select_slider("Analyte copy number", options=list(analyte_format_map.keys()))
    analyte = analyte_format_map[analyte_fmt]

    capture = st.select_slider("Capture reagent conc. (µg/ml)", options=sorted(df_combined['capture'].unique()))
    probe = st.select_slider("Probe reagent concentration (µg/ml)", options=sorted(df_combined['probe'].unique()))
    affinity_label = st.select_slider("Affinity (kD*)", options=AFFINITY_ORDERED, value='medium (kD* = 13)')
    kd = AFF_KD_FROM_LABELED[affinity_label]

with st.sidebar.expander("📈 Optimize parameters for target S/N", expanded=False):
    target_sn_linear = st.number_input("Target S/N value", value=10.0, step=1.0)
    target_sn = np.log10(target_sn_linear)

    selected_affinity = st.selectbox("Affinity (kD*)", options=["any"] + AFFINITY_ORDERED)

    all_analytes = sorted(df_combined['analyte.copy.nbr'].unique())
    analyte_map = {f"{x:.0e}": x for x in all_analytes}
    analyte_target_fmt = st.select_slider("Analyte copy number", options=["any"] + list(analyte_map.keys()), value="any")
    analyte_target = analyte_map[analyte_target_fmt] if analyte_target_fmt != "any" else "any"

# ---------- Model Info ----------
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

# ---------- S/N prediction for selected parameters ----------
with st.expander("### 🔍 Predicted S/N based on selected parameters", expanded=True):
    exact_match = df_combined[
        (np.isclose(df_combined['analyte.copy.nbr'], analyte, rtol=0.01)) &
        (np.isclose(df_combined['capture'], capture, rtol=0.01)) &
        (np.isclose(df_combined['probe'], probe, rtol=0.01)) &
        (np.isclose(df_combined['Affinity_KD'], kd, rtol=0.01))
    ]

    if not exact_match.empty:
        row = exact_match.iloc[0]
        sn = row['log10SN']
        src = row['source']
        st.metric(label="log10(S/N)", value=f"{sn:.2f}")
        st.metric(label="S/N", value=f"{10**sn:.2f}")
        st.caption(f"Source: {'Predicted by model' if src == 'predicted' else 'Measured experimentally'}")
    else:
        st.warning("No exact match found.")

    st.markdown("#### 🧭 Nearby parameter space")
    tolerance = 0.3
    nearby = df_combined[
        (np.isclose(df_combined['analyte.copy.nbr'], analyte, rtol=tolerance)) &
        (np.isclose(df_combined['capture'], capture, rtol=tolerance)) &
        (np.isclose(df_combined['probe'], probe, rtol=tolerance)) &
        (np.isclose(df_combined['Affinity_KD'], kd, rtol=tolerance))
    ]

    if not nearby.empty:
        nearby = nearby.copy()
        nearby['S/N'] = 10 ** nearby['log10SN']
        # format analyte in scientific notation
        nearby['analyte.copy.nbr_sci'] = nearby['analyte.copy.nbr'].apply(lambda x: f"{x:.2e}")
        st.dataframe(
            nearby[[
                'analyte.copy.nbr_sci', 'capture', 'probe', 'Affinity_label', 'log10SN', 'S/N', 'source'
            ]].rename(columns={'analyte.copy.nbr_sci': 'Analyte Copy Number',
                               'Affinity_label': 'Affinity (kD*)'}).sort_values('log10SN', ascending=False)
        )
    else:
        st.info("No nearby points found within tolerance.")

# ---------- Optimization ----------
with st.expander("### 📈 Optimized parameters for target S/N", expanded=False):
    df_sn = df_combined.copy()
    if selected_affinity != "any":
        df_sn = df_sn[df_sn['Affinity_label'] == selected_affinity]
    if analyte_target != "any":
        df_sn = df_sn[np.isclose(df_sn['analyte.copy.nbr'], float(analyte_target), rtol=0.01)]

    tolerance_sn = 0.1
    matches = df_sn[np.isclose(df_sn['log10SN'], target_sn, atol=tolerance_sn)]

    if not matches.empty:
        matches = matches.copy()
        matches['S/N'] = 10 ** matches['log10SN']
        # format analyte in scientific notation
        matches['analyte.copy.nbr_sci'] = matches['analyte.copy.nbr'].apply(lambda x: f"{x:.2e}")
        st.success(f"Parameter combinations close to target S/N ≈ {target_sn_linear:.2f}:")
        st.dataframe(
            matches[[
                'analyte.copy.nbr_sci', 'capture', 'probe', 'Affinity_label', 'log10SN', 'S/N', 'source'
            ]].rename(columns={'analyte.copy.nbr_sci': 'Analyte Copy Number',
                               'Affinity_label': 'Affinity (kD*)'}).sort_values('log10SN')
        )
    else:
        st.warning("No parameter sets found close to that target S/N. Try adjusting inputs or tolerance.")

# ---------- 3D Visualization ----------
with st.expander("### 🌐 3D Parameter Space Visualization", expanded=False):
    custom_colorscale = [
        [0.0, '#FFDAB9'],
        [0.5, '#FF4500'],
        [1.0, '#800080']
    ]

    for aff in AFFINITY_ORDERED:
        st.write(f"#### Affinity: {aff.capitalize()}")
        df_sub = df_combined[df_combined['Affinity_label'] == aff].copy()

        # Guard against log10(0) and invalids
        df_sub['log10_analyte_copy_nbr'] = np.where(
            df_sub['analyte.copy.nbr'] > 0,
            np.log10(df_sub['analyte.copy.nbr']),
            np.nan
        )
        # scientific notation for hover
        df_sub['analyte.copy.nbr_sci'] = df_sub['analyte.copy.nbr'].apply(lambda x: f"{x:.2e}")
        df_sub = df_sub.replace([np.inf, -np.inf], np.nan).dropna(subset=['log10_analyte_copy_nbr'])

        if df_sub.empty:
            st.warning(f"No valid analyte copy numbers for {aff}")
            continue

        tick_vals = np.arange(
            int(np.floor(df_sub['log10_analyte_copy_nbr'].min())),
            int(np.ceil(df_sub['log10_analyte_copy_nbr'].max())) + 1
        )
        tick_texts = [f"10<sup>{i}</sup>" for i in tick_vals]

        fig = px.scatter_3d(
            df_sub,
            x='log10_analyte_copy_nbr',
            y='capture',
            z='probe',
            color='log10SN',
            opacity=0.7,
            color_continuous_scale=custom_colorscale,
            labels={
                'log10_analyte_copy_nbr': 'Analyte Copy Number (log10)',
                'capture': 'Capture reagent conc. (µg/ml)',
                'probe': 'Probe reagent concentration (µg/ml)',
                'log10SN': 'log10(S/N)',
                'analyte.copy.nbr_sci': 'Analyte Copy Number'
            },
            hover_data=['analyte.copy.nbr_sci']
        )

        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    title='Analyte Copy Number',
                    tickvals=tick_vals.tolist(),
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
