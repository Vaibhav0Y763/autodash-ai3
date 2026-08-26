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
                                    elif chart_style == "Pie / Donut":
                                        fig = px.pie(grp, names=dim, values=metric, hole=0.4, color_discrete_sequence=colors)
                                    else:  # Area
                                        fig = px.area(grp, x=dim, y=metric, color_discrete_sequence=colors)
                                    fig.update_layout(showlegend=False, height=350, margin=dict(t=30, b=30, l=10, r=10))
                                    st.plotly_chart(fig, width='stretch')
                                except Exception as insight_err:
                                    st.warning(f"Couldn't render chart for {dim}: {insight_err}")

        # TAB 5: SMART ANALYSIS (business-field mapping + named analyses + missing-data warnings)
        with tab_smart:
            st.subheader("🧠 Smart Analysis")
            st.caption("InsightForge tries to auto-detect business fields (Sales, Profit, Region, Order Date...) "
                       "from your column names. Adjust any dropdown below if a guess looks wrong.")

            auto_mapping = auto_detect_column_mapping(active_df)
            map_cols = st.columns(3)
            field_names = list(CANONICAL_FIELDS.keys())
            user_mapping = {}
            col_options = ["(Not used)"] + active_df.columns.tolist()
            for idx, field in enumerate(field_names):
                default_col = auto_mapping.get(field)
                default_idx = col_options.index(default_col) if default_col in col_options else 0
                chosen = map_cols[idx % 3].selectbox(field, col_options, index=default_idx, key=f"map_{field}")
                user_mapping[field] = None if chosen == "(Not used)" else chosen

            available, missing = get_available_missing_analyses(user_mapping)

            if missing:
                warn_lines = "\n".join(f"- **{name}** → needs `{', '.join(fields)}`" for name, fields in missing)
                st.warning(f"⚠️ Some recommended analyses cannot be generated:\n\n{warn_lines}")

            show_dashboard = (not missing) or st.session_state.get("smart_continue", False)
            if missing and not show_dashboard:
                if st.button("Continue & Skip Missing Analyses", type="primary", key="smart_continue_btn"):
                    st.session_state.smart_continue = True
                    st.rerun()
            else:
                st.write("---")
                st.write("### 📌 Key Metrics")
                sales_col, profit_col, order_col, date_col = (
                    user_mapping["Sales"], user_mapping["Profit"], user_mapping["Order ID"], user_mapping["Order Date"]
                )
                kpis = [("📋", "Total Rows", f"{len(active_df):,}")]
                if order_col:
                    kpis.append(("🧾", "Total Orders", f"{active_df[order_col].nunique():,}"))
                if sales_col:
                    kpis.append(("💰", "Total Sales", f"{active_df[sales_col].sum():,.0f}"))
                if profit_col:
                    kpis.append(("📈", "Total Profit", f"{active_df[profit_col].sum():,.0f}"))
                if sales_col and order_col:
                    aov = active_df[sales_col].sum() / active_df[order_col].nunique()
                    kpis.append(("🎯", "Avg Order Value", f"{aov:,.1f}"))

                render_kpi_cards(kpis)

                if available:
                    st.write("---")
                    st.write("### 📊 Smart Analyses")
                    color_theme = st.selectbox("Color theme", list(COLOR_THEMES.keys()), key="smart_color")
                    colors = COLOR_THEMES[color_theme]

                    analysis_cols = st.columns(2)
                    slot = 0
                    for name in available:
                        with analysis_cols[slot % 2]:
                            with st.container(border=True):
                                st.caption(name)
                                try:
                                    if name == "Monthly Sales Trend":
                                        tmp = active_df.copy()
                                        tmp["_period"] = pd.to_datetime(tmp[date_col], errors='coerce').dt.to_period('M').astype(str)
                                        trend = tmp.dropna(subset=["_period"]).groupby("_period")[sales_col].sum().reset_index()
                                        fig = px.line(trend, x="_period", y=sales_col, markers=True, color_discrete_sequence=colors)
                                    elif name == "Region-wise Sales":
                                        grp = active_df.groupby(user_mapping["Region"])[sales_col].sum().sort_values(ascending=False).reset_index()
                                        fig = px.bar(grp, x=user_mapping["Region"], y=sales_col, color=user_mapping["Region"], color_discrete_sequence=colors)
                                    elif name == "Country-wise Sales":
                                        grp = active_df.groupby(user_mapping["Country"])[sales_col].sum().sort_values(ascending=False).reset_index()
                                        fig = px.bar(grp, x=user_mapping["Country"], y=sales_col, color=user_mapping["Country"], color_discrete_sequence=colors)
                                    elif name == "Top Products":
                                        grp = active_df.groupby(user_mapping["Product"])[sales_col].sum().sort_values(ascending=False).head(10).reset_index()
                                        fig = px.bar(grp, x=user_mapping["Product"], y=sales_col, color_discrete_sequence=colors)
                                    elif name == "Profit Analysis":
                                        fig = px.histogram(active_df, x=profit_col, marginal="box", color_discrete_sequence=colors)
                                    elif name == "Category Performance":
                                        grp = active_df.groupby(user_mapping["Category"])[sales_col].sum().sort_values(ascending=False).reset_index()
                                        fig = px.pie(grp, names=user_mapping["Category"], values=sales_col, hole=0.4, color_discrete_sequence=colors)
                                    elif name == "Employee Performance":
                                        grp = active_df.groupby(user_mapping["Employee"])[sales_col].sum().sort_values(ascending=False).reset_index()
                                        fig = px.bar(grp, x=user_mapping["Employee"], y=sales_col, color=user_mapping["Employee"], color_discrete_sequence=colors)
                                    fig.update_layout(showlegend=False, height=340, margin=dict(t=20, b=20, l=10, r=10))
                                    st.plotly_chart(fig, width='stretch')
                                except Exception as smart_err:
                                    st.warning(f"Couldn't render {name}: {smart_err}")
                        slot += 1

                    # export a plain-text insights summary
                    summary_lines = [f"InsightForge — Smart Analysis Summary", "=" * 40]
                    for icon, label, value in kpis:
                        summary_lines.append(f"{label}: {value}")
                    summary_lines.append("")
                    summary_lines.append("Analyses included: " + ", ".join(available))
                    if missing:
                        summary_lines.append("Analyses skipped (missing data): " + ", ".join(m[0] for m in missing))
                    st.download_button(
                        "⬇️ Export Insights Summary (.txt)", "\n".join(summary_lines),
                        "insightforge_summary.txt", "text/plain"
                    )
                else:
                    st.info("No analyses available yet — map at least one metric + dimension above.")

        # TAB 6: CUSTOM CHART BUILDER
        with tab_custom:
            c1, c2, c3, c4 = st.columns(4)
            c_type = c1.selectbox("Chart Type", ["Bar Chart", "Line Chart", "Scatter Plot", "Pie / Donut", "Box Plot"])
            x_ax = c2.selectbox("X-Axis", active_df.columns.tolist())
            y_options = numeric_cols if numeric_cols else active_df.columns.tolist()
            y_ax = c3.selectbox("Y-Axis", y_options)
            # Only offer Sum/Average when the Y column is actually numeric, avoids crashes
            agg_options = ["Sum", "Average", "Count", "None"] if y_ax in numeric_cols else ["Count", "None"]
            agg = c4.selectbox("Aggregation", agg_options)

            try:
                if agg == "Sum":
                    plot_df = active_df.groupby(x_ax)[y_ax].sum().reset_index()
                elif agg == "Average":
                    plot_df = active_df.groupby(x_ax)[y_ax].mean().reset_index()
                elif agg == "Count":
                    plot_df = active_df.groupby(x_ax)[y_ax].count().reset_index()
                else:
                    plot_df = active_df

                if c_type == "Bar Chart":
                    fig = px.bar(plot_df, x=x_ax, y=y_ax)
                elif c_type == "Line Chart":
                    fig = px.line(plot_df, x=x_ax, y=y_ax, markers=True)
                elif c_type == "Pie / Donut":
                    fig = px.pie(plot_df.head(10), names=x_ax, values=y_ax, hole=0.4)
                elif c_type == "Scatter Plot":
                    fig = px.scatter(active_df, x=x_ax, y=y_ax)
                elif c_type == "Box Plot":
                    fig = px.box(active_df, x=x_ax, y=y_ax)
                st.plotly_chart(fig, width='stretch')
            except Exception as chart_err:
                st.warning(f"Couldn't build this chart with the selected options: {chart_err}")

        # TAB 7: STORYTELLING
        with tab_story:
            st.subheader("Automated Executive Summary")
            dup_removed = max(len(df) - len(active_df), 0)
            st.write(f"- Dataset has **{len(active_df):,} rows** and **{len(active_df.columns)} features**.")
            if dup_removed > 0:
                st.write(f"- **{dup_removed:,} duplicate rows** were removed during cleaning.")
            if len(numeric_cols) > 0:
                top_col = numeric_cols[0]
                st.write(
                    f"- Primary metric **`{top_col}`** ranges from **{active_df[top_col].min():,.2f}** "
                    f"to **{active_df[top_col].max():,.2f}**, averaging **{active_df[top_col].mean():,.2f}**."
                )
                if len(categorical_cols) > 0:
                    top_group = active_df.groupby(categorical_cols[0])[top_col].sum().idxmax()
                    st.write(f"- **`{top_group}`** leads all `{categorical_cols[0]}` groups by total `{top_col}`.")
            st.write("**Suggested next actions:**")
            st.write("- Investigate columns with high missing-value percentages before deeper analysis.")
            st.write("- Validate any flagged type-mismatch columns before using them in calculations.")

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("👉 Please upload a CSV or Excel file in the sidebar to get started.")

st.markdown("""
<div class="footer">
    Developed by <strong>Vaibhav Saini</strong> · sainivaibhav535@gmail.com
</div>
""", unsafe_allow_html=True)
