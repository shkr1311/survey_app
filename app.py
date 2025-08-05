import streamlit as st
import pandas as pd
from cleaning import auto_clean, rule_clean
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from jinja2 import Environment, FileSystemLoader
import tempfile

st.set_page_config(page_title='Smart Survey Data Cleaner', layout='wide')

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        return pd.read_excel(uploaded_file)
    else:
        return None

def get_col_types(df):
    return {col: str(df[col].dtype) for col in df.columns}

def margin_of_error(data, weights, confidence=0.95):
    from scipy.stats import norm
    mu = np.average(data, weights=weights)
    variance = np.average((data - mu)**2, weights=weights)
    std_err = np.sqrt(variance / len(data))
    interval = norm.ppf((1 + confidence) / 2) * std_err
    return interval

###################### Streamlit UI ########################

st.title("🔬 Smart Survey Data Cleaner")
st.write("**Upload, clean, validate, and analyze your survey data with ease!**")

lang = st.radio("Language", options=["English", "Hindi"], key="lang")
# (For production: use Streamlit ecosystem translation solutions or messages dict)

uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=['csv', 'xlsx'])

if uploaded_file:
    df = load_data(uploaded_file)
    st.subheader("Raw Data Preview")
    st.dataframe(df.head())
    st.write(f"Shape: {df.shape}")

    if st.button("Auto Clean Data"):
        cleaned, log, outliers, types = auto_clean(df.copy())
        st.session_state['cleaned'] = cleaned
        st.session_state['clean_log'] = log
        st.session_state['outliers'] = outliers
        st.success("Data cleaned successfully!")

if 'cleaned' in st.session_state:
    cleaned = st.session_state['cleaned']
    log = st.session_state['clean_log']
    outliers = st.session_state['outliers']

    st.subheader("Missing % per column")
    nulls = cleaned.isnull().mean().mul(100).round(1).to_dict()
    st.write({k: f"{v}%" for k, v in nulls.items()})

    st.subheader("Create/Load Validation Rules")
    rule_input = st.text_area("Paste rules as JSON", value='{"age": [">", 0], "income": ["<", 1000000]}')
    if st.button("Apply Validation Rules"):
        import json
        try:
            rules = json.loads(rule_input)
            validated, val_log, removed_by_rule = rule_clean(cleaned.copy(), rules)
            st.session_state['validated'] = validated
            st.session_state['validation_log'] = val_log
            st.session_state['removed_by_rule'] = removed_by_rule
            st.success("Validation rules applied!")
        except Exception as e:
            st.error(f"Invalid rules JSON: {e}")

if 'validated' in st.session_state:
    validated = st.session_state['validated']
    val_log = st.session_state['validation_log']

    st.subheader("Weighting")
    weight_col = st.selectbox("Select weight column", options=[None] + list(validated.columns))
    est_col = st.selectbox("Select numeric column for estimate", options=validated.select_dtypes('number').columns)

    if st.button("Calculate Weighted Estimates"):
        if weight_col:
            data = validated[est_col].dropna()
            weights = validated.loc[data.index, weight_col].replace(0, np.nan).dropna()
            # aligning indices & drop NaNs
            common_idx = data.index.intersection(weights.index)
            data = data.loc[common_idx]
            weights = weights.loc[common_idx]
            wmean = np.average(data, weights=weights)
            wsum = np.dot(data, weights)
            moe = margin_of_error(data, weights)
        else:
            wmean = validated[est_col].mean()
            wsum = validated[est_col].sum()
            moe = margin_of_error(validated[est_col].dropna(), np.ones(len(validated[est_col].dropna())))
        st.write({
            "Weighted mean": wmean,
            "Weighted sum": wsum,
            "95% margin of error": moe
        })

    st.subheader("Preview Cleaned Data")
    st.dataframe(validated.head())
    st.info(f"Final cleaned rows: {validated.shape[0]}")

    st.subheader("Download Cleaned Data")
    csv = validated.to_csv(index=False).encode()
    st.download_button("Download CSV", csv, file_name="cleaned_data.csv", mime="text/csv")

    ################ Visualization ###################
    st.subheader("Data Visualizations")

    col_to_plot = st.selectbox("Column for Before/After Chart", options=validated.select_dtypes('number').columns)
    if col_to_plot:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        sns.boxplot(y=st.session_state['cleaned'][col_to_plot], ax=axs[0])
        axs[0].set_title('Before Cleaning')
        sns.boxplot(y=validated[col_to_plot], ax=axs[1])
        axs[1].set_title('After Cleaning/Validation')
        st.pyplot(fig)

    # Histogram
    fig = px.histogram(validated, x=col_to_plot, nbins=30, title="Distribution After Cleaning")
    st.plotly_chart(fig)

    st.subheader("Summary Charts")
    st.bar_chart(pd.DataFrame({'Missing %': list(nulls.values())}, index=nulls.keys()))
    st.write("Rows removed (outliers):", len(st.session_state['outliers']))
    st.write("Rows removed (validation):", len(st.session_state['removed_by_rule']))

    #################### Generate Report #################
    st.subheader("Generate & Export Report")

    if st.button("Create PDF/HTML Report"):
        # Jinja2 templating
        env = Environment(loader=FileSystemLoader('assets'))
        template = env.get_template('report_template.html')
        estimates = [
            {"name": est_col, "unweighted": validated[est_col].mean(), "weighted": wmean}
        ]
        context = dict(
            shape=validated.shape,
            nulls=nulls,
            log=st.session_state['clean_log'],
            val_log=val_log,
            estimates=estimates,
            cleaned_rows=validated.shape[0],
            removed_rows=len(st.session_state['outliers']) + len(st.session_state['removed_by_rule'])
        )
        html_content = template.render(**context)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
            tmp.write(html_content.encode())
            html_path = tmp.name
        with open(html_path, "rb") as file:
            st.download_button("Download HTML Report", file, file_name="data_report.html")

        # Convert to PDF
        import pdfkit
        pdf_path = html_path.replace('.html', '.pdf')
        try:
            pdfkit.from_file(html_path, pdf_path)
            with open(pdf_path, "rb") as file:
                st.download_button("Download PDF Report", file, file_name="data_report.pdf")
        except Exception as e:
            st.warning(f"PDF generation failed: {e}")

    st.subheader("Cleaning Log")
    for step in st.session_state['clean_log']:
        st.write("-", step)
    st.subheader("Validation Log")
    for step in val_log:
        st.write("-", step)
