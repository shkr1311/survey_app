import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy.stats import norm
from scipy import stats
import json
from datetime import datetime

st.set_page_config(page_title='Smart Survey Data Cleaner', layout='wide')

# ==================== MULTILINGUAL SUPPORT ====================
TRANSLATIONS = {
    "English": {
        "title": "Smart Survey Data Cleaner",
        "subtitle": "Upload, clean, validate, and analyze your survey data with ease!",
        "upload": "Upload a CSV or Excel file",
        "raw_preview": "Raw Data Preview",
        "shape": "Shape",
        "auto_clean": "Auto Clean Data",
        "clean_success": "Data cleaned successfully!",
        "missing_pct": "Missing % per column",
        "col_types": "Detected Column Types",
        "validation_rules": "Create/Load Validation Rules",
        "paste_rules": "Paste rules as JSON",
        "apply_rules": "Apply Validation Rules",
        "rules_applied": "Validation rules applied!",
        "invalid_json": "Invalid rules JSON",
        "weighting": "Weighting & Statistical Estimates",
        "select_weight": "Select weight column (optional)",
        "select_numeric": "Select numeric column for estimate",
        "calc_estimates": "Calculate Weighted Estimates",
        "weighted_mean": "Weighted Mean",
        "weighted_sum": "Weighted Sum",
        "margin_error": "95% Margin of Error",
        "unweighted_mean": "Unweighted Mean",
        "preview_clean": "Preview Cleaned Data",
        "final_rows": "Final cleaned rows",
        "download_data": "Download Cleaned Data",
        "download_csv": "Download CSV",
        "visualizations": "Data Visualizations",
        "select_column": "Select column for visualization",
        "before_clean": "Before Cleaning",
        "after_clean": "After Cleaning/Validation",
        "distribution": "Distribution After Cleaning",
        "summary_charts": "Summary Statistics",
        "rows_removed_outliers": "Rows removed (outliers)",
        "rows_removed_validation": "Rows removed (validation)",
        "original_rows": "Original rows",
        "generate_report": "Generate & Export Report",
        "create_report": "Create HTML Report",
        "download_html": "Download HTML Report",
        "cleaning_log": "Cleaning Log",
        "validation_log": "Validation Log",
        "no_data": "Please upload a file first",
        "select_lang": "Language",
        "rule_examples": "Rule Examples",
        "rule_info": "Use JSON format: {\"column_name\": [\"operator\", value]}",
        "operators": "Operators: '>', '<', '>=', '<=', '==', '!=', 'in', 'not in'",
        "example_rules": "Example: {\"age\": [\">\", 0], \"status\": [\"in\", [\"active\", \"pending\"]]}",
        "data_quality": "Data Quality Overview",
        "retention_rate": "Data Retention Rate",
        "missing_analysis": "Missing Data Analysis",
        "type_distribution": "Column Type Distribution",
        "no_missing": "No missing data in the cleaned dataset!",
    },
    "Hindi": {
        "title": "🔬 स्मार्ट सर्वेक्षण डेटा क्लीनर",
        "subtitle": "अपने सर्वेक्षण डेटा को आसानी से अपलोड, साफ़, मान्य और विश्लेषण करें!",
        "upload": "CSV या Excel फ़ाइल अपलोड करें",
        "raw_preview": "कच्चा डेटा पूर्वावलोकन",
        "shape": "आकार",
        "auto_clean": "स्वतः डेटा साफ़ करें",
        "clean_success": "डेटा सफलतापूर्वक साफ़ किया गया!",
        "missing_pct": "प्रति कॉलम गुम %",
        "col_types": "पहचाने गए कॉलम प्रकार",
        "validation_rules": "सत्यापन नियम बनाएं/लोड करें",
        "paste_rules": "JSON के रूप में नियम पेस्ट करें",
        "apply_rules": "सत्यापन नियम लागू करें",
        "rules_applied": "सत्यापन नियम लागू किए गए!",
        "invalid_json": "अमान्य नियम JSON",
        "weighting": "भारांकन और सांख्यिकीय अनुमान",
        "select_weight": "वेट कॉलम चुनें (वैकल्पिक)",
        "select_numeric": "अनुमान के लिए संख्यात्मक कॉलम चुनें",
        "calc_estimates": "भारित अनुमान की गणना करें",
        "weighted_mean": "भारित औसत",
        "weighted_sum": "भारित योग",
        "margin_error": "95% त्रुटि का मार्जिन",
        "unweighted_mean": "अभारित औसत",
        "preview_clean": "साफ़ किए गए डेटा का पूर्वावलोकन",
        "final_rows": "अंतिम साफ़ पंक्तियाँ",
        "download_data": "साफ़ किया गया डेटा डाउनलोड करें",
        "download_csv": "CSV डाउनलोड करें",
        "visualizations": "डेटा दृश्यावलोकन",
        "select_column": "दृश्यावलोकन के लिए कॉलम चुनें",
        "before_clean": "सफाई से पहले",
        "after_clean": "सफाई/सत्यापन के बाद",
        "distribution": "सफाई के बाद वितरण",
        "summary_charts": "सारांश सांख्यिकी",
        "rows_removed_outliers": "पंक्तियाँ हटाई गईं (आउटलायर)",
        "rows_removed_validation": "पंक्तियाँ हटाई गईं (सत्यापन)",
        "original_rows": "मूल पंक्तियाँ",
        "generate_report": "रिपोर्ट तैयार और निर्यात करें",
        "create_report": "HTML रिपोर्ट बनाएं",
        "download_html": "HTML रिपोर्ट डाउनलोड करें",
        "cleaning_log": "सफाई लॉग",
        "validation_log": "सत्यापन लॉग",
        "no_data": "कृपया पहले एक फ़ाइल अपलोड करें",
        "select_lang": "भाषा",
        "rule_examples": "नियम उदाहरण",
        "rule_info": "JSON प्रारूप उपयोग करें: {\"column_name\": [\"operator\", value]}",
        "operators": "ऑपरेटर: '>', '<', 'in', 'not in'",
        "example_rules": "उदाहरण: {\"age\": [\">\", 0], \"status\": [\"in\", [\"active\", \"pending\"]]}",
    }
}

def t(key, lang="English"):
    """Translation helper function"""
    return TRANSLATIONS.get(lang, TRANSLATIONS["English"]).get(key, key)

# ==================== CLEANING FUNCTIONS ====================

def log_step(logs, step):
    """Add step to cleaning log"""
    logs.append(step)

def auto_clean(df):
    """Automatic data cleaning with comprehensive logging"""
    logs = []
    removed_rows = pd.DataFrame()
    original_shape = df.shape

    # Detect column types
    types = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            types[col] = 'numeric'
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            types[col] = 'datetime'
        else:
            types[col] = 'categorical'
    log_step(logs, f"Detected column types: {types}")

    # Handle missing values based on column type and null percentage
    for col in df.columns:
        coltype = types[col]
        null_pct = df[col].isnull().mean()
        
        if coltype == 'numeric':
            if null_pct < 0.1:
                df[col].fillna(df[col].mean(), inplace=True)
                log_step(logs, f"{col}: <10% null, filled with mean")
            elif null_pct < 0.3:
                df[col].fillna(df[col].median(), inplace=True)
                log_step(logs, f"{col}: 10–30% null, filled with median")
            else:
                df.drop(columns=[col], inplace=True)
                log_step(logs, f"{col}: >30% null, column dropped")
        elif coltype == 'categorical':
            mode = df[col].mode(dropna=True)
            if not mode.empty:
                df[col].fillna(mode[0], inplace=True)
                log_step(logs, f"{col}: categorical, filled nulls with mode ({mode[0]})")
            else:
                df[col].fillna('Unknown', inplace=True)
                log_step(logs, f"{col}: categorical, filled nulls with 'Unknown'")

    # Remove outliers using IQR method for numeric columns
    for col, coltype in types.items():
        if coltype == 'numeric' and col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            low, high = Q1 - 1.5*IQR, Q3 + 1.5*IQR
            mask = (df[col] < low) | (df[col] > high)
            removed = df[mask]
            if not removed.empty:
                removed_rows = pd.concat([removed_rows, removed])
                df = df[~mask]
                log_step(logs, f"{col}: removed {mask.sum()} outliers using IQR (range: {low:.2f} to {high:.2f})")

    log_step(logs, f"Cleaning complete: {original_shape[0]} → {len(df)} rows")
    return df.reset_index(drop=True), logs, removed_rows, types

def rule_clean(df, rules):
    """Apply validation rules to dataframe"""
    logs = []
    removed_rows = pd.DataFrame()
    keep_mask = pd.Series([True] * len(df), index=df.index)

    for col, (rule, val) in rules.items():
        if col not in df.columns:
            logs.append(f"Rule on {col}: column not found, skipped")
            continue
        
        if rule == '>':
            mask = df[col] > val
        elif rule == '<':
            mask = df[col] < val
        elif rule == '>=':
            mask = df[col] >= val
        elif rule == '<=':
            mask = df[col] <= val
        elif rule == '==':
            mask = df[col] == val
        elif rule == '!=':
            mask = df[col] != val
        elif rule == 'in':
            mask = df[col].isin(val)
        elif rule == 'not in':
            mask = ~df[col].isin(val)
        else:
            logs.append(f"Rule on {col}: unknown operator '{rule}', skipped")
            continue

        bad_mask = ~mask
        if bad_mask.sum() > 0:
            logs.append(f"{col}: removed {bad_mask.sum()} rows by rule '{rule}' {val}")
            removed_rows = pd.concat([removed_rows, df[bad_mask]])
        keep_mask = keep_mask & mask

    cleaned = df[keep_mask]
    logs.append(f"Validation complete: {len(df)} → {len(cleaned)} rows")
    return cleaned.reset_index(drop=True), logs, removed_rows

def margin_of_error(data, weights, confidence=0.95):
    """Calculate margin of error for weighted estimate"""
    mu = np.average(data, weights=weights)
    variance = np.average((data - mu)**2, weights=weights)
    std_err = np.sqrt(variance / len(data))
    interval = norm.ppf((1 + confidence) / 2) * std_err
    return interval

# ==================== DATA LOADING ====================

@st.cache_data
def load_data(uploaded_file):
    """Load CSV or Excel file with error handling"""
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            return pd.read_excel(uploaded_file)
        else:
            return None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# ==================== REPORT GENERATION ====================

def generate_html_report(context, lang="English"):
    """Generate comprehensive HTML report"""
    html = f"""
    <!DOCTYPE html>
    <html lang="{'hi' if lang == 'Hindi' else 'en'}">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{t('generate_report', lang)}</title>
        <style>
            * {{ box-sizing: border-box; margin: 0; padding: 0; }}
            body {{ 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0; 
                padding: 20px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }}
            .container {{ 
                max-width: 1200px;
                margin: 0 auto;
                background: white; 
                padding: 40px; 
                border-radius: 12px; 
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            }}
            h1 {{ 
                color: #2c3e50; 
                border-bottom: 4px solid #667eea; 
                padding-bottom: 15px;
                margin-bottom: 10px;
                font-size: 2.5em;
            }}
            h2 {{ 
                color: #34495e; 
                margin-top: 35px;
                margin-bottom: 20px;
                font-size: 1.8em;
                border-left: 5px solid #667eea;
                padding-left: 15px;
            }}
            .timestamp {{ 
                color: #7f8c8d; 
                font-size: 0.95em;
                margin-bottom: 30px;
                font-style: italic;
            }}
            table {{ 
                border-collapse: collapse; 
                width: 100%; 
                margin: 25px 0;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}
            th, td {{ 
                border: 1px solid #e0e0e0; 
                padding: 14px; 
                text-align: left; 
            }}
            th {{ 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                font-weight: 600;
                text-transform: uppercase;
                font-size: 0.9em;
                letter-spacing: 0.5px;
            }}
            tr:nth-child(even) {{ background-color: #f8f9fa; }}
            tr:hover {{ background-color: #e3f2fd; transition: background-color 0.3s; }}
            .metric-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            .metric-card {{
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                border-top: 4px solid #667eea;
            }}
            .metric-card h3 {{
                color: #667eea;
                font-size: 1em;
                margin-bottom: 10px;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            .metric-card .value {{
                font-size: 2em;
                font-weight: bold;
                color: #2c3e50;
            }}
            .log-item {{ 
                padding: 10px; 
                margin: 8px 0;
                border-bottom: 1px solid #ecf0f1;
                padding-left: 20px;
                position: relative;
            }}
            .log-item:before {{
                content: "→";
                position: absolute;
                left: 0;
                color: #667eea;
                font-weight: bold;
            }}
            .log-section {{
                background: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                margin: 20px 0;
            }}
            .footer {{
                margin-top: 50px;
                padding-top: 20px;
                border-top: 2px solid #ecf0f1;
                text-align: center;
                color: #7f8c8d;
                font-size: 0.9em;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{t('title', lang)}</h1>
            <p class="timestamp">📅 {t('generate_report', lang)}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>📊 {t('summary_charts', lang)}</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <h3>{t('original_rows', lang)}</h3>
                    <div class="value">{context.get('original_rows', 'N/A')}</div>
                </div>
                <div class="metric-card">
                    <h3>{t('final_rows', lang)}</h3>
                    <div class="value">{context['cleaned_rows']}</div>
                </div>
                <div class="metric-card">
                    <h3>{t('rows_removed_outliers', lang)}</h3>
                    <div class="value">{context.get('outliers_removed', 0)}</div>
                </div>
                <div class="metric-card">
                    <h3>{t('rows_removed_validation', lang)}</h3>
                    <div class="value">{context.get('validation_removed', 0)}</div>
                </div>
            </div>
    """
    
    # Column Types
    if context.get('col_types'):
        html += f"""
            <h2>🏷️ {t('col_types', lang)}</h2>
            <table>
                <tr><th>Column</th><th>Type</th></tr>
        """
        for col, ctype in context['col_types'].items():
            html += f"<tr><td>{col}</td><td>{ctype}</td></tr>"
        html += "</table>"
    
    # Missing Values
    html += f"""
            <h2>❓ {t('missing_pct', lang)}</h2>
            <table>
                <tr><th>Column</th><th>Missing %</th></tr>
    """
    
    for col, pct in context.get('nulls', {}).items():
        html += f"<tr><td>{col}</td><td>{pct}%</td></tr>"
    
    html += "</table>"
    
    # Cleaning Log
    html += f"""
            <h2>🧹 {t('cleaning_log', lang)}</h2>
            <div class="log-section">
    """
    
    for log_item in context.get('clean_log', []):
        html += f'<div class="log-item">{log_item}</div>'
    
    html += "</div>"
    
    # Validation Log
    if context.get('validation_log'):
        html += f"""
            <h2>✅ {t('validation_log', lang)}</h2>
            <div class="log-section">
        """
        
        for log_item in context['validation_log']:
            html += f'<div class="log-item">{log_item}</div>'
        
        html += "</div>"
    
    # Estimates
    if context.get('estimates'):
        html += f"""
            <h2>📈 {t('calc_estimates', lang)}</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
        """
        for key, value in context['estimates'].items():
            if isinstance(value, (int, float)):
                html += f"<tr><td>{key}</td><td>{value:.4f}</td></tr>"
            else:
                html += f"<tr><td>{key}</td><td>{value}</td></tr>"
        html += "</table>"
    
    html += f"""
            <div class="footer">
                <p>Generated by {t('title', lang)}</p>
                <p>Powered by Streamlit & Python</p>
            </div>
        </div>
    </body>
    </html>
    """
    return html

# ==================== MAIN APP ====================

# Sidebar for language selection
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/data-cleaning.png", width=100)
    lang = st.radio(
        t("select_lang", "English"), 
        options=["English", "Hindi"], 
        key="lang"
    )
    st.markdown("---")
    st.markdown("### " + t("rule_examples", lang))
    st.info(t("operators", lang))
    st.code(t("example_rules", lang), language="json")

# Main title
st.title(t("title", lang))
st.markdown(f"**{t('subtitle', lang)}**")

# File upload
uploaded_file = st.file_uploader(t("upload", lang), type=['csv', 'xlsx'])

if uploaded_file:
    df = load_data(uploaded_file)
    
    if df is not None:
        st.session_state['original_shape'] = df.shape
        
        # Raw data preview
        st.subheader(t("raw_preview", lang))
        st.dataframe(df.head(100), use_container_width=True)
        st.info(f"📏 {t('shape', lang)}: {df.shape[0]} rows × {df.shape[1]} columns")

        # Auto clean button
        if st.button(t("auto_clean", lang), type="primary", use_container_width=True):
            with st.spinner('🔄 Cleaning data...'):
                cleaned, log, outliers, types = auto_clean(df.copy())
                st.session_state['cleaned'] = cleaned
                st.session_state['clean_log'] = log
                st.session_state['outliers'] = outliers
                st.session_state['col_types'] = types
                st.session_state['original_df'] = df.copy()
                st.success("✅ " + t("clean_success", lang))
                st.rerun()

# Display cleaning results
if 'cleaned' in st.session_state:
    cleaned = st.session_state['cleaned']
    log = st.session_state['clean_log']
    outliers = st.session_state['outliers']
    col_types = st.session_state.get('col_types', {})

    # Column types
    st.subheader(t("col_types", lang))
    types_df = pd.DataFrame(list(col_types.items()), columns=['Column', 'Type'])
    st.dataframe(types_df, use_container_width=True)

    # Missing values
    st.subheader(t("missing_pct", lang))
    nulls = cleaned.isnull().mean().mul(100).round(2).to_dict()
    null_df = pd.DataFrame(list(nulls.items()), columns=['Column', 'Missing %'])
    st.dataframe(null_df, use_container_width=True)

    # Validation rules
    st.subheader(t("validation_rules", lang))
    st.info(t("rule_info", lang))
    
    rule_input = st.text_area(
        t("paste_rules", lang), 
        value='{"age": [">", 0]}',
        height=100
    )
    
    if st.button(t("apply_rules", lang), type="primary"):
        try:
            rules = json.loads(rule_input)
            validated, val_log, removed_by_rule = rule_clean(cleaned.copy(), rules)
            st.session_state['validated'] = validated
            st.session_state['validation_log'] = val_log
            st.session_state['removed_by_rule'] = removed_by_rule
            st.success("✅ " + t("rules_applied", lang))
            st.rerun()
        except json.JSONDecodeError as e:
            st.error(f"❌ {t('invalid_json', lang)}: {e}")
        except Exception as e:
            st.error(f"❌ Error: {e}")

# Display validated data
if 'validated' in st.session_state:
    validated = st.session_state['validated']
    val_log = st.session_state.get('validation_log', [])

    # Weighting section
    st.subheader(t("weighting", lang))
    
    col1, col2 = st.columns(2)
    
    with col1:
        weight_col = st.selectbox(
            t("select_weight", lang), 
            options=[None] + list(validated.columns)
        )
    
    with col2:
        numeric_cols = validated.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            est_col = st.selectbox(t("select_numeric", lang), options=numeric_cols)
        else:
            st.warning("⚠️ No numeric columns available")
            est_col = None

    if est_col and st.button(t("calc_estimates", lang), type="primary"):
        try:
            data = validated[est_col].dropna()
            
            if weight_col and weight_col in validated.columns:
                weights = validated.loc[data.index, weight_col].replace(0, np.nan).dropna()
                common_idx = data.index.intersection(weights.index)
                data = data.loc[common_idx]
                weights = weights.loc[common_idx]
                
                if len(data) > 0:
                    wmean = np.average(data, weights=weights)
                    wsum = np.dot(data, weights)
                    moe = margin_of_error(data.values, weights.values)
                else:
                    wmean = wsum = moe = 0
            else:
                wmean = data.mean()
                wsum = data.sum()
                if len(data) > 0:
                    moe = margin_of_error(data.values, np.ones(len(data)))
                else:
                    moe = 0
            
            st.session_state['estimates'] = {
                t('unweighted_mean', lang): validated[est_col].mean(),
                t('weighted_mean', lang): wmean,
                t('weighted_sum', lang): wsum,
                t('margin_error', lang): moe
            }
            
            # Display metrics
            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.metric(t("unweighted_mean", lang), f"{validated[est_col].mean():.2f}")
            with metric_cols[1]:
                st.metric(t("weighted_mean", lang), f"{wmean:.2f}")
            with metric_cols[2]:
                st.metric(t("weighted_sum", lang), f"{wsum:.2f}")
            with metric_cols[3]:
                st.metric(t("margin_error", lang), f"±{moe:.2f}")
                
        except Exception as e:
            st.error(f"❌ Error calculating estimates: {e}")

    # Preview cleaned data
    st.subheader(t("preview_clean", lang))
    st.dataframe(validated.head(100), use_container_width=True)
    st.info(f"✅ {t('final_rows', lang)}: **{validated.shape[0]}** / {st.session_state.get('original_shape', (0,0))[0]}")

    # Download section
    st.subheader(t("download_data", lang))
    csv = validated.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 " + t("download_csv", lang), 
        csv, 
        file_name=f"cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", 
        mime="text/csv",
        use_container_width=True
    )

    # Visualizations
    st.subheader(t("visualizations", lang))
    
    if numeric_cols:
        # Filter out ID columns for better visualization
        meaningful_cols = [col for col in numeric_cols if col.lower() not in ['id', 'index']]
        
        if not meaningful_cols:
            meaningful_cols = numeric_cols
        
        col_to_plot = st.selectbox(
            t("select_column", lang), 
            options=meaningful_cols,
            index=0 if meaningful_cols else 0
        )
        
        if col_to_plot:
            # Before/After comparison with better styling
            fig, axs = plt.subplots(1, 2, figsize=(14, 6))
            
            if 'original_df' in st.session_state and col_to_plot in st.session_state['original_df'].columns:
                sns.boxplot(y=st.session_state['original_df'][col_to_plot], ax=axs[0], color='#ff6b6b')
                axs[0].set_title(t("before_clean", lang), fontsize=14, fontweight='bold')
                axs[0].set_ylabel(col_to_plot, fontsize=12)
                axs[0].grid(True, alpha=0.3)
            
            sns.boxplot(y=validated[col_to_plot], ax=axs[1], color='#51cf66')
            axs[1].set_title(t("after_clean", lang), fontsize=14, fontweight='bold')
            axs[1].set_ylabel(col_to_plot, fontsize=12)
            axs[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Enhanced Histogram with statistics
            col_stats = validated[col_to_plot].describe()
            
            fig2 = px.histogram(
                validated, 
                x=col_to_plot, 
                nbins=20,
                title=f"{t('distribution', lang)} - {col_to_plot}",
                labels={col_to_plot: col_to_plot, 'count': 'Frequency'},
                color_discrete_sequence=['#667eea']
            )
            
            # Add mean line
            fig2.add_vline(
                x=col_stats['mean'], 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Mean: {col_stats['mean']:.2f}",
                annotation_position="top"
            )
            
            # Add median line
            fig2.add_vline(
                x=col_stats['50%'], 
                line_dash="dash", 
                line_color="green",
                annotation_text=f"Median: {col_stats['50%']:.2f}",
                annotation_position="bottom"
            )
            
            fig2.update_layout(
                showlegend=False,
                height=500,
                xaxis_title=col_to_plot,
                yaxis_title="Frequency (Count)"
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
            # Display statistics
            st.markdown("### 📊 Statistical Summary")
            stat_cols = st.columns(5)
            with stat_cols[0]:
                st.metric("Mean", f"{col_stats['mean']:.2f}")
            with stat_cols[1]:
                st.metric("Median", f"{col_stats['50%']:.2f}")
            with stat_cols[2]:
                st.metric("Std Dev", f"{col_stats['std']:.2f}")
            with stat_cols[3]:
                st.metric("Min", f"{col_stats['min']:.2f}")
            with stat_cols[4]:
                st.metric("Max", f"{col_stats['max']:.2f}")

    # Summary Charts
    st.subheader(t("summary_charts"))