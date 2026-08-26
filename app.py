import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import difflib
import io

st.set_page_config(page_title="InsightForge", page_icon="🔥", layout="wide")

st.markdown("""
<style>
.hero {
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 55%, #C026D3 100%);
    padding: 2rem 2rem 1.6rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 30px rgba(124,58,237,0.25);
}
.hero h1 {
    color: white;
    font-size: 2.1rem;
    margin: 0 0 0.3rem 0;
}
.hero p {
    color: rgba(255,255,255,0.9);
    font-size: 1rem;
    margin: 0;
}
.footer {
    text-align: center;
    color: #888;
    font-size: 0.85rem;
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid rgba(128,128,128,0.2);
}

/* --- Buttons --- */
.stButton > button, .stDownloadButton > button {
    border-radius: 10px;
    border: none;
    background: linear-gradient(135deg, #4F46E5, #7C3AED);
    color: white;
    font-weight: 600;
    padding: 0.5rem 1.2rem;
    transition: opacity 0.15s ease, transform 0.15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    opacity: 0.88;
    transform: translateY(-1px);
    color: white;
}

/* --- Tabs --- */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background-color: #F3F0FF;
    border-radius: 10px 10px 0 0;
    padding: 8px 16px;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #4F46E5, #7C3AED) !important;
    color: white !important;
}

/* --- KPI cards --- */
.kpi-card {
    border-radius: 14px;
    padding: 1rem 1.2rem;
    color: white;
    box-shadow: 0 4px 14px rgba(0,0,0,0.12);
    margin-bottom: 0.5rem;
}
.kpi-card .kpi-label {
    font-size: 0.85rem;
    opacity: 0.9;
}
.kpi-card .kpi-value {
    font-size: 1.6rem;
    font-weight: 700;
    margin-top: 4px;
}
</style>
<div class="hero">
    <h1>🔥 InsightForge</h1>
    <p>Upload your data → Clean it → Get automated business insights, instantly.</p>
</div>
""", unsafe_allow_html=True)

KPI_GRADIENTS = [
    "linear-gradient(135deg, #4F46E5, #7C3AED)",
    "linear-gradient(135deg, #7C3AED, #C026D3)",
    "linear-gradient(135deg, #0EA5E9, #4F46E5)",
    "linear-gradient(135deg, #DB2777, #C026D3)",
    "linear-gradient(135deg, #059669, #0EA5E9)",
]


def render_kpi_cards(kpis):
    """Render a row of colorful gradient KPI cards. kpis = list of (icon, label, value) tuples."""
    cols = st.columns(len(kpis))
    for i, (col, (icon, label, value)) in enumerate(zip(cols, kpis)):
        gradient = KPI_GRADIENTS[i % len(KPI_GRADIENTS)]
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="background:{gradient};">
                <div class="kpi-label">{icon} {label}</div>
                <div class="kpi-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)


@st.cache_data
def load_file(file_bytes, file_name):
    """Cached file loader so repeated Streamlit reruns don't re-parse the file every time."""
    if file_name.endswith('.csv'):
        return pd.read_csv(io.BytesIO(file_bytes))
    else:
        return pd.read_excel(io.BytesIO(file_bytes))


def detect_type_mismatches(df):
    """Flag text columns that are mostly numeric-looking (e.g. '100', '250') but stored as text."""
    issues = []
    for col in df.select_dtypes(include=['object']).columns:
        non_null = df[col].dropna()
        if len(non_null) == 0:
            continue
        converted = pd.to_numeric(non_null, errors='coerce')
        pct_numeric = converted.notna().mean()
        if pct_numeric > 0.8:
            issues.append((col, round(pct_numeric * 100, 1)))
    return issues


def is_id_like_numeric(df, col):
    """Heuristic for NUMERIC columns: name suggests an ID/index, or values are literally a
    row-number sequence (0..n-1 or 1..n). Uniqueness alone is NOT used here, since real metrics
    like Sales/Revenue are often highly unique too."""
    name_flag = any(k in col.lower() for k in ['id', 'index', 'unnamed', 'code'])
    seq_flag = False
    s = df[col].dropna()
    if len(s) == len(df) and pd.api.types.is_integer_dtype(s):
        sorted_vals = s.sort_values().reset_index(drop=True).values
        if np.array_equal(sorted_vals, np.arange(len(s))) or np.array_equal(sorted_vals, np.arange(1, len(s) + 1)):
            seq_flag = True
    return name_flag or seq_flag


def is_id_like_categorical(df, col):
    """Heuristic for TEXT columns: name suggests an ID/index, or nearly every value is unique
    (e.g. Order ID strings) -- unlike numeric metrics, unique text columns are usually identifiers."""
    name_flag = any(k in col.lower() for k in ['id', 'index', 'unnamed', 'code'])
    unique_flag = df[col].nunique() >= 0.95 * len(df)
    return name_flag or unique_flag


def rank_numeric_cols(df, cols):
    """Put meaningful metrics (Age, Sales, Qty...) before ID/index-like columns for dashboard defaults."""
    non_id = [c for c in cols if not is_id_like_numeric(df, c)]
    id_like = [c for c in cols if is_id_like_numeric(df, c)]
    return non_id + id_like


def rank_categorical_cols(df, cols):
    """Put low-cardinality real categories (Region, Category...) before ID-like text columns."""
    return sorted(cols, key=lambda c: (is_id_like_categorical(df, c), df[c].nunique()))


COLOR_THEMES = {
    "Default": px.colors.qualitative.Plotly,
    "Vivid": px.colors.qualitative.Vivid,
    "Pastel": px.colors.qualitative.Pastel,
    "Bold": px.colors.qualitative.Bold,
    "Set2": px.colors.qualitative.Set2,
    "Dark24": px.colors.qualitative.Dark24,
}


# ---------- Smart Analysis: business-field auto-detection ----------
CANONICAL_FIELDS = {
    "Sales": ["sales", "revenue", "amount", "total amount", "order value", "sale amount", "net sales", "revenue amount", "turnover"],
    "Profit": ["profit", "margin", "net profit", "net income", "earnings"],
    "Order Date": ["order date", "date", "purchase date", "invoice date", "transaction date"],
    "Region": ["region", "zone", "area"],
    "Country": ["country", "nation"],
    "Category": ["category", "product category", "segment"],
    "Product": ["product", "product name", "item", "sku"],
    "Employee": ["employee", "salesperson", "sales rep", "agent", "staff"],
    "Order ID": ["order id", "invoice id", "transaction id"],
}
NUMERIC_FIELDS = {"Sales", "Profit"}
DATE_FIELDS = {"Order Date"}
CATEGORICAL_FIELDS = {"Region", "Country", "Category", "Product", "Employee"}

ANALYSES = {
    "Monthly Sales Trend": ["Order Date", "Sales"],
    "Region-wise Sales": ["Region", "Sales"],
    "Country-wise Sales": ["Country", "Sales"],
    "Top Products": ["Product", "Sales"],
    "Profit Analysis": ["Profit"],
    "Category Performance": ["Category", "Sales"],
    "Employee Performance": ["Employee", "Sales"],
}


# words that automatically DISQUALIFY a column for a field, even if other words match
# (e.g. a "Discount_Amount" column should never be picked as the "Sales" field just because
# both contain the word "amount")
NEGATIVE_KEYWORDS = {
    "Sales": {"discount", "tax", "cost", "refund", "return", "shipping", "fee"},
    "Profit": {"tax"},
}


def _norm(s):
    return s.lower().strip().replace('_', ' ').replace('-', ' ')


def _word_score(field, alias, col_norm, col_words):
    alias_norm = _norm(alias)
    alias_words = set(alias_norm.split())
    if col_norm == alias_norm:
        return 1.0
    if field.lower() in col_words:          # column literally contains the field's own name -> strong signal
        return 0.95
    if alias_words.issubset(col_words):
        return 0.9
    if col_words.issubset(alias_words) and col_words:
        return 0.75
    return 0


def auto_detect_column_mapping(df):
    """Guess which real column corresponds to each canonical business field
    (Sales, Profit, Region, Order Date...) using name matching + dtype checks.
    Word-based matches always win over fuzzy string similarity, and disqualifying
    keywords (e.g. 'discount' can never satisfy 'Sales') are excluded up front."""
    mapping = {}
    for field, aliases in CANONICAL_FIELDS.items():
        neg = NEGATIVE_KEYWORDS.get(field, set())
        best_match, best_score = None, 0

        # Pass 1: reliable word-based matching
        for col in df.columns:
            col_norm = _norm(col)
            col_words = set(col_norm.split())
            if col_words & neg:
                continue
            for alias in aliases:
                score = _word_score(field, alias, col_norm, col_words)
                if score > best_score:
                    best_score, best_match = score, col

        # Pass 2: only fall back to fuzzy similarity if word-matching found nothing
        if best_score == 0:
            for col in df.columns:
                col_norm = _norm(col)
                col_words = set(col_norm.split())
                if col_words & neg:
                    continue
                for alias in aliases:
                    score = difflib.SequenceMatcher(None, col_norm, _norm(alias)).ratio()
                    if score > best_score:
                        best_score, best_match = score, col
            if best_score < 0.75:  # stricter cutoff for the fuzzy fallback only
                best_match = None

        if best_match is not None:
            if field in NUMERIC_FIELDS and not pd.api.types.is_numeric_dtype(df[best_match]):
                best_match = None
            elif field in CATEGORICAL_FIELDS and pd.api.types.is_numeric_dtype(df[best_match]):
                best_match = None
            elif field in DATE_FIELDS:
                parsed = pd.to_datetime(df[best_match], errors='coerce')
                if parsed.notna().mean() < 0.7:
                    best_match = None
            elif field == "Order ID" and df[best_match].dtype == object:
                # only exclude if a TEXT column actually looks like real dates
                parsed = pd.to_datetime(df[best_match], errors='coerce')
                if parsed.notna().mean() > 0.7:
                    best_match = None
        mapping[field] = best_match
    return mapping


def get_available_missing_analyses(mapping):
    """Split the named analyses into ones we have enough mapped columns for, and ones we don't."""
    available, missing = [], []
    for name, needed_fields in ANALYSES.items():
        missing_fields = [f for f in needed_fields if not mapping.get(f)]
        if missing_fields:
            missing.append((name, missing_fields))
        else:
            available.append(name)
    return available, missing


def clean_dataframe(df):
    """1-click cleaning: drop duplicates, trim text, fill missing text/numbers."""
    cleaned = df.copy().drop_duplicates()
    for col in cleaned.select_dtypes(include=['object']).columns:
        cleaned[col] = cleaned[col].astype(str).str.strip()
        cleaned[col] = cleaned[col].replace(['nan', 'None', 'NaN', '<NA>', ''], np.nan)
        cleaned[col] = cleaned[col].fillna('Unknown')
    for col in cleaned.select_dtypes(include=['number']).columns:
        if cleaned[col].isnull().sum() > 0:
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())
    return cleaned


uploaded_file = st.sidebar.file_uploader("📁 Upload CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    try:
        file_bytes = uploaded_file.getvalue()
        try:
            df = load_file(file_bytes, uploaded_file.name)
        except ImportError:
            st.error("⚠️ Legacy .xls files need the `xlrd` package. Run `pip install xlrd`, "
                     "or re-save your file as .xlsx and re-upload.")
            st.stop()

        if 'file_signature' not in st.session_state:
            st.session_state.file_signature = None

        file_signature = (uploaded_file.name, uploaded_file.size)
        is_new_file = st.session_state.file_signature != file_signature

        if is_new_file or 'clean_df' not in st.session_state or st.sidebar.button("🔄 Reset Data"):
            st.session_state.clean_df = df.copy()
            st.session_state.file_signature = file_signature
            st.session_state.pop("smart_continue", None)  # re-show the missing-analysis gate for the new file

        tab_audit, tab_clean, tab_dashboard, tab_insights, tab_smart, tab_custom, tab_story = st.tabs([
            "🔍 Data Audit", "✨ Auto-Clean", "📈 Auto Dashboard", "🚀 Auto Insights",
            "🧠 Smart Analysis", "🎨 Custom Chart Builder", "📝 AI Business Story"
        ])

        # TAB 1: DATA AUDIT
        with tab_audit:
            st.subheader("Data Health Overview")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Rows", f"{len(df):,}")
            col2.metric("Total Columns", f"{len(df.columns)}")
            col3.metric("Duplicate Rows", f"{int(df.duplicated().sum())}")
            col4.metric("Missing Values", f"{int(df.isnull().sum().sum())}")

            missing_pct = (df.isnull().sum() / len(df) * 100).round(1)
            missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
            if len(missing_pct) > 0:
                st.write("**Missing values by column:**")
                st.dataframe(
                    missing_pct.rename("Missing %").reset_index().rename(columns={'index': 'Column'}),
                    width='stretch', hide_index=True
                )

            mismatches = detect_type_mismatches(df)
            if mismatches:
                st.write("**⚠️ Possible type mismatches (numbers stored as text):**")
                for col, pct in mismatches:
                    st.write(f"- `{col}` — {pct}% of values look numeric")

            st.write("---")
            st.dataframe(df.head(8), width='stretch')

        # TAB 2: AUTO CLEAN
        with tab_clean:
            st.subheader("1-Click Automated Cleaning Pipeline")
            st.caption("Drops duplicate rows, trims whitespace, fills missing text with 'Unknown', "
                        "fills missing numbers with the column median.")
            if st.button("🧹 Execute Auto-Cleaning", type="primary", width='stretch'):
                st.session_state.clean_df = clean_dataframe(df)
                removed = len(df) - len(st.session_state.clean_df)
                st.success(f"✅ Dataset cleaned! Removed {removed:,} duplicate row(s).")

            curr_data = st.session_state.clean_df
            st.dataframe(curr_data.head(8), width='stretch')

            dl1, dl2 = st.columns(2)
            csv_buf = io.StringIO()
            curr_data.to_csv(csv_buf, index=False)
            dl1.download_button(
                "⬇️ Download Cleaned CSV", csv_buf.getvalue(),
                f"cleaned_{uploaded_file.name.split('.')[0]}.csv", "text/csv",
                width='stretch'
            )

            excel_buf = io.BytesIO()
            curr_data.to_excel(excel_buf, index=False, engine='openpyxl')
            dl2.download_button(
                "⬇️ Download Cleaned Excel", excel_buf.getvalue(),
                f"cleaned_{uploaded_file.name.split('.')[0]}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width='stretch'
            )

        active_df = st.session_state.clean_df
        numeric_cols = rank_numeric_cols(active_df, active_df.select_dtypes(include=['number']).columns.tolist())
        categorical_cols = rank_categorical_cols(active_df, active_df.select_dtypes(include=['object', 'category']).columns.tolist())

        # TAB 3: AUTO DASHBOARD
        with tab_dashboard:
            if len(numeric_cols) == 0:
                st.info("No numeric columns found — the auto dashboard needs at least one numeric column.")
            else:
                dash_kpis = [("📊", f"Total {col}", f"{active_df[col].sum():,.1f}") for col in numeric_cols[:4]]
                render_kpi_cards(dash_kpis)
                g1, g2 = st.columns(2)
                with g1:
                    st.plotly_chart(
                        px.histogram(active_df, x=numeric_cols[0], marginal="box"),
                        width='stretch'
                    )
                with g2:
                    if len(categorical_cols) > 0:
                        grp = (
                            active_df.groupby(categorical_cols[0])[numeric_cols[0]]
                            .sum().reset_index()
                            .sort_values(by=numeric_cols[0], ascending=False).head(10)
                        )
                        st.plotly_chart(px.bar(grp, x=categorical_cols[0], y=numeric_cols[0]), width='stretch')
                    else:
                        st.info("No categorical columns found for grouping.")

        # TAB 4: AUTO INSIGHTS (region-wise / country-wise / employee-wise breakdowns, fully interactive)
        with tab_insights:
            if len(numeric_cols) == 0 or len(categorical_cols) == 0:
                st.info("Need at least one numeric column and one categorical column (e.g. Region, "
                        "Country, Employee) to generate auto insights.")
            else:
                st.subheader("🚀 Automated Insights Dashboard")
                ctrl1, ctrl2, ctrl3, ctrl4 = st.columns(4)
                metric = ctrl1.selectbox("Metric to analyze", numeric_cols, key="insights_metric")
                chart_style = ctrl2.selectbox("Chart style", ["Bar", "Line", "Pie / Donut", "Area"], key="insights_style")
                color_theme_name = ctrl3.selectbox("Color theme", list(COLOR_THEMES.keys()), key="insights_color")
                top_n = ctrl4.slider("Top N per chart", 3, 20, 8, key="insights_topn")
                colors = COLOR_THEMES[color_theme_name]

                default_dims = categorical_cols[:4]
                selected_dims = st.multiselect(
                    "Break down by (pick any dimensions — Region, Country, Employee, Category...)",
                    categorical_cols, default=default_dims, key="insights_dims"
                )

                if not selected_dims:
                    st.info("Select at least one dimension above to see insights.")
                else:
                    st.write("### 🔎 Key Insights")
                    total_metric = active_df[metric].sum()
                    for dim in selected_dims:
                        grp = active_df.groupby(dim)[metric].sum().sort_values(ascending=False)
                        if len(grp) == 0:
                            continue
                        top_cat, top_val = grp.index[0], grp.iloc[0]
                        pct = (top_val / total_metric * 100) if total_metric else 0
                        st.write(f"- **{dim}**: `{top_cat}` leads with **{top_val:,.1f}** total "
                                 f"{metric} ({pct:.1f}% of overall).")

                    st.write("---")
                    st.write(f"### 📊 {metric} breakdown by dimension")

                    cols_per_row = 2
                    for i in range(0, len(selected_dims), cols_per_row):
                        row_dims = selected_dims[i:i + cols_per_row]
                        row_cols = st.columns(len(row_dims))
                        for rc, dim in zip(row_cols, row_dims):
                            grp = (
                                active_df.groupby(dim)[metric].sum()
                                .sort_values(ascending=False).head(top_n).reset_index()
                            )
                            with rc:
                                st.caption(f"{metric} by {dim}")
                                try:
                                    if chart_style == "Bar":
                                        fig = px.bar(grp, x=dim, y=metric, color=dim, color_discrete_sequence=colors)
                                    elif chart_style == "Line":
                                        fig = px.line(grp, x=dim, y=metric, markers=True, color_discrete_sequence=colors)
