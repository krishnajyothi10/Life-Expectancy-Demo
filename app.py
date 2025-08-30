# Guard against anything that might unset sys.modules['warnings']
import sys as _sys, warnings as _warnings
if 'warnings' not in _sys.modules:
    _sys.modules['warnings'] = _warnings

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px   # you’re using plotly later
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# --- LOAD & CLEAN DATA (TIDY, NO PIVOT) ---
st.title("Life Expectancy Prediction App")

df = pd.read_csv("Life Expectancy Data.csv")

# Standardize column names (strip and unify spaces/case)
df.columns = [c.strip() for c in df.columns]

# If your file is the WHO/Kaggle one, it usually has these columns:
# 'Country', 'Status', 'Life expectancy', 'Adult Mortality', 'infant deaths',
# 'Alcohol', 'percentage expenditure', 'Hepatitis B', 'Measles ', ' BMI ',
# 'under-five deaths ', 'Polio', 'Total expenditure', 'Diphtheria ', 'HIV/AIDS',
# 'GDP', 'Population', ' thinness  1-19 years', ... etc.
# We’ll align the few you reference later:

rename_map = {
    'Life expectancy': 'Life Expectancy',
    'Hepatitis B': 'Immunization HepB3',
    'Diphtheria ': 'Immunization DPT',
    'percentage expenditure': 'Health Expenditure',  # note: this is per Kaggle WHO def
    'Population': 'population',
    'Country': 'Country',
    'Status': 'Status',
}
for k, v in rename_map.items():
    if k in df.columns:
        df.rename(columns={k: v}, inplace=True)

# Keep only columns we use later if they exist
wanted = [
    'Country', 'Status', 'CO2 Emission', 'Health Expenditure', 'GDP',
    'Immunization DPT', 'Immunization HepB3', 'Immunization Measles',
    'Life Expectancy', 'Infant Death', 'Maternal Death', 'Primary Education',
    'population'
]
# Some of these may not exist in your file; keep what’s present.
present = [c for c in wanted if c in df.columns]
data = df[present].copy()

# Fix obvious string placeholders -> NaN and coerce numerics
data.replace(["..", "..."], np.nan, inplace=True)
for c in data.columns:
    if c not in ['Country', 'Status']:
        data[c] = pd.to_numeric(data[c], errors='coerce')

# Optional: drop last 5 rows ONLY if you know they’re footer notes. Otherwise remove this.
# data = data.iloc[:-5]

# Merge population if you need from an external file and if not already present
if 'population' not in data.columns:
    pop = pd.read_csv("Life Expectancy Data.csv")  # must contain columns: Country, population
    pop.columns = [c.strip() for c in pop.columns]
    data = pd.merge(data, pop[['Country', 'population']], on='Country', how='left')

# Clean country text
if 'Country' in data.columns:
    data['Country'] = data['Country'].astype(str).str.replace(',', '', regex=False)

# Final NA drop for modeling/plots
data.dropna(subset=['Life Expectancy'], inplace=True)

st.dataframe(data)

st.write("")
row9_space1, row9_space2, row9_space3 = st.columns((0.5, 1, 0.5))
with row9_space2:
    num = data.select_dtypes(include='number')
    corr = num.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_alpha(0)
    sns.heatmap(corr, annot=True, square=True)
    st.pyplot(fig)

# ------- helper: robust bubble scatter that handles NaNs in size -------
import numpy as np
import plotly.express as px
import pandas as pd  # ensure pd is in scope here

def make_bubble(df, xcol, ycol, sizecol='population', namecol='Country',
                log_x=False, min_size=6, max_size=40):
    # Ensure numeric
    for c in [xcol, ycol, sizecol]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Keep rows with valid x,y
    d = df.dropna(subset=[xcol, ycol]).copy()

    # Size handling
    if sizecol not in d.columns:
        d['_size'] = min_size
    else:
        s = d[sizecol].copy()
        # Fill NaNs with median; ensure strictly positive
        if s.isna().any():
            med = s.median(skipna=True)
            if not np.isfinite(med) or med <= 0:
                med = 1.0
            s = s.fillna(med)
        s = s.clip(lower=1e-6)
        # Winsorize extremes
        lo, hi = s.quantile([0.05, 0.95])
        if np.isfinite(lo): s = s.clip(lower=lo)
        if np.isfinite(hi): s = s.clip(upper=hi)
        # Map to visual size range
        s_vis = np.log1p(s)
        s_vis = (s_vis - s_vis.min()) / (s_vis.max() - s_vis.min() + 1e-9)
        s_vis = s_vis * (max_size - min_size) + min_size
        d['_size'] = s_vis

    if d.empty:
        return None

    fig = px.scatter(
        d,
        x=xcol, y=ycol,
        size="_size",
        hover_name=namecol if namecol in d.columns else None,
        log_x=log_x
    )
    return fig
# ----------------------------------------------------------------------

import numpy as np
import plotly.express as px

row3_space1, row3_1, row3_space2, row3_2, row3_space3 = st.columns((0.1, 1, 0.1, 1, 0.1))
# --- Your two plots ---
with row3_1:
    st.subheader("Scatter Plot: Health Expenditure vs Life Expectancy")
    fig1 = make_bubble(
        data,
        xcol="Health Expenditure",
        ycol="Life Expectancy",
        sizecol="population",
        namecol="Country",
        log_x=True
    )
    if fig1:
        st.plotly_chart(fig1, theme="streamlit", use_container_width=True)

with row3_2:
    st.subheader("Scatter Plot: Immunization HepB3 vs Life Expectancy")
    fig2 = make_bubble(
        data,
        xcol="Immunization HepB3",
        ycol="Life Expectancy",
        sizecol="population",
        namecol="Country",
        log_x=False
    )
    if fig2:
        st.plotly_chart(fig2, theme="streamlit", use_container_width=True)

def make_bubble(df, xcol, ycol, sizecol='population', namecol='Country',
                log_x=False, min_size=6, max_size=40):
    # Ensure numeric
    for c in [xcol, ycol, sizecol]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Keep rows with valid x,y
    d = df.dropna(subset=[xcol, ycol]).copy()

    # Handle size column
    if sizecol not in d.columns:
        # fallback: constant size if population missing
        d['_size'] = min_size
    else:
        # start from numeric size, allow zeros but not NaN/negatives
        s = d[sizecol].copy()

        # Option A (strict): drop NaNs and nonpositive
        # d = d[(~s.isna()) & (s > 0)].copy()
        # s = d[sizecol]

        # Option B (lenient): fill NaNs with median and clamp to >0
        if s.isna().any():
            med = s.median(skipna=True)
            if np.isnan(med) or med <= 0:
                med = 1.0
            s = s.fillna(med)
        s = s.clip(lower=1e-6)

        # Optional: de-winsorize extreme sizes
        lo, hi = s.quantile([0.05, 0.95])
        s = s.clip(lower=lo if np.isfinite(lo) else s.min(),
                   upper=hi if np.isfinite(hi) else s.max())

        # Map to a visual size scale
        # (log transform helps when population spans many orders)
        s_vis = np.log1p(s)
        s_vis = (s_vis - s_vis.min()) / (s_vis.max() - s_vis.min() + 1e-9)
        s_vis = s_vis * (max_size - min_size) + min_size
        d['_size'] = s_vis

    if d.empty:
        st.info(f"No data available for plotting {xcol} vs {ycol}.")
        return None

    fig = px.scatter(
        d,
        x=xcol, y=ycol,
        size="_size",
        hover_name=namecol if namecol in d.columns else None,
        log_x=log_x
    )
    return fig

# --- Your two plots ---
# 5-column layout (spacer, plot, spacer, plot, spacer)
row3_space1, row3_1, row3_space2, row3_2, row3_space3 = st.columns((0.1, 1, 0.1, 1, 0.1))

with row3_1:
    st.subheader("Scatter Plot: Health Expenditure vs Life Expectancy")
    fig1 = make_bubble(
        data,
        xcol="Health Expenditure",
        ycol="Life Expectancy",
        sizecol="population",
        namecol="Country",
        log_x=True
    )
    if fig1 is None:
        st.info("No rows available to plot after cleaning.")
    else:
        st.plotly_chart(
            fig1,
            theme="streamlit",
            use_container_width=True,
            key="chart_health_vs_life"   # ✅ unique key
        )

with row3_2:
    st.subheader("Scatter Plot: Immunization HepB3 vs Life Expectancy")
    fig2 = make_bubble(
        data,
        xcol="Immunization HepB3",
        ycol="Life Expectancy",
        sizecol="population",
        namecol="Country",
        log_x=False
    )
    if fig2 is None:
        st.info("No rows available to plot after cleaning.")
    else:
        st.plotly_chart(
            fig2,
            theme="streamlit",
            use_container_width=True,
            key="chart_hepb3_vs_life"   # ✅ another unique key
        )



import numbers, streamlit as st
st.write("numbers module path:", getattr(numbers, "__file__", "builtin"))
if "site-packages" in str(getattr(numbers, "__file__", "")).lower():
    st.error("A third-party 'numbers' package is installed. Run: pip uninstall -y numbers")
