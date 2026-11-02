
from pathlib import Path
import sys

import joblib
import pandas as pd
import streamlit as st


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fuse_decoder import decode_fuse_map, load_schema
from validator import validate_record


MODEL_PATH = PROJECT_ROOT / "models" / "best_fuse_error_model.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "mcu_fuse_map_audit.csv"

FUSE_FIELDS = [
    "BG_TRIM",
    "HSI_TRIM",
    "LDO_TRIM",
    "TS_CAL1",
    "FLASH_SIZE",
    "PERIPH_EN",
    "TEMP_GRADE",
    "DEBUG_LOCK",
    "RDP",
    "RESERVED",
]

DERIVATIVES = ["MCU-A", "MCU-B", "MCU-C"]


# ============================================================
# 2. LOAD RESOURCES
# ============================================================

st.set_page_config(
    page_title="MCU Fuse Map Integrity Auditor",
    page_icon="🔬",
    layout="wide",
)


# ============================================================
# PREMIUM SEMICONDUCTOR DASHBOARD THEME
# ============================================================
BACKGROUND_IMAGE_URL = (
    "https://images.unsplash.com/photo-1518770660439-4636190af475"
    "?auto=format&fit=crop&w=2200&q=85"
)

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {{
        --navy: #061326;
        --panel: rgba(4, 23, 46, 0.88);
        --cyan: #11c5ff;
        --muted: #a9c7e8;
    }}

    html, body, [class*="css"], [data-testid="stAppViewContainer"] {{
        font-family: 'Inter', sans-serif;
    }}

    [data-testid="stAppViewContainer"] {{
        background:
            linear-gradient(115deg, rgba(2, 12, 28, .94), rgba(2, 19, 41, .82)),
            url("{BACKGROUND_IMAGE_URL}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: #edf7ff;
    }}

    [data-testid="stHeader"] {{
        background: rgba(2, 13, 29, .72);
        border-bottom: 1px solid rgba(17, 197, 255, .24);
    }}

    [data-testid="stSidebar"] {{
        background:
            linear-gradient(180deg, rgba(2, 13, 31, .98), rgba(3, 25, 49, .94)),
            url("{BACKGROUND_IMAGE_URL}");
        background-size: cover;
        border-right: 1px solid rgba(17, 197, 255, .32);
    }}

    [data-testid="stSidebar"] * {{
        color: #e7f4ff;
    }}

    .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1600px;
    }}

    h1, h2, h3 {{
        color: #f4f9ff !important;
        letter-spacing: -.025em;
    }}

    p, label, .stMarkdown {{
        color: #d8e9fa;
    }}

    [data-testid="stMetric"] {{
        background: linear-gradient(145deg, rgba(7, 36, 67, .94), rgba(3, 20, 42, .92));
        border: 1px solid rgba(17, 197, 255, .55);
        border-radius: 16px;
        padding: 18px 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, .22);
    }}

    [data-testid="stMetricLabel"] {{
        color: #c1d9f2 !important;
    }}

    [data-testid="stMetricValue"] {{
        color: #ffffff !important;
        font-weight: 800;
    }}

    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: rgba(3, 22, 44, .78);
        border: 1px solid rgba(30, 144, 210, .42);
        border-radius: 16px;
    }}

    .hero {{
        padding: 26px 28px;
        margin: 0 0 22px 0;
        border: 1px solid rgba(17, 197, 255, .42);
        border-radius: 20px;
        background:
            linear-gradient(90deg, rgba(2, 16, 36, .96), rgba(3, 31, 59, .62)),
            url("{BACKGROUND_IMAGE_URL}");
        background-position: center 42%;
        background-size: cover;
        box-shadow: 0 16px 42px rgba(0, 0, 0, .25);
    }}

    .hero-eyebrow {{
        color: #19d2ff;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .13em;
        font-size: .78rem;
    }}

    .hero-title {{
        color: #f7fbff;
        font-size: clamp(1.8rem, 3.3vw, 3rem);
        font-weight: 800;
        line-height: 1.12;
        margin: 8px 0;
    }}

    .hero-title span {{ color: #18caff; }}

    .hero-subtitle {{
        color: #c4def8;
        font-size: 1rem;
    }}

    .section-kicker {{
        color: #20caff;
        text-transform: uppercase;
        letter-spacing: .12em;
        font-size: .75rem;
        font-weight: 700;
    }}

    .stButton > button, .stDownloadButton > button {{
        border-radius: 10px;
        border: 1px solid rgba(17, 197, 255, .7);
        background: linear-gradient(110deg, #087fe5, #08b9f4);
        color: white;
        font-weight: 700;
        transition: .18s ease-in-out;
    }}

    .stButton > button:hover, .stDownloadButton > button:hover {{
        border-color: #7be4ff;
        box-shadow: 0 0 18px rgba(17, 197, 255, .28);
        color: white;
        transform: translateY(-1px);
    }}

    [data-baseweb="select"] > div, [data-baseweb="input"] > div,
    [data-baseweb="textarea"] > div {{
        background: rgba(3, 20, 40, .92);
        border-color: rgba(77, 157, 210, .55);
        border-radius: 10px;
    }}

    [data-testid="stDataFrame"], [data-testid="stTable"] {{
        border: 1px solid rgba(17, 197, 255, .3);
        border-radius: 12px;
        overflow: hidden;
    }}

    [data-testid="stAlert"] {{
        border-radius: 12px;
    }}

    hr {{
        border-color: rgba(17, 197, 255, .25);
    }}

    @media (max-width: 900px) {{
        .block-container {{ padding-left: 1rem; padding-right: 1rem; }}
        .hero {{ padding: 20px; }}
    }}

    [data-testid="stMetric"]:nth-child(1) {{ border-color: rgba(17, 197, 255, .8); }}
[data-testid="stMetric"] [data-testid="stMetricDelta"] {{ color: #27e7b1 !important; }}
[data-testid="stMarkdownContainer"] h3 {{ letter-spacing: -.02em; }}
[data-testid="stSidebar"] [role="radiogroup"] label {{
    padding: 8px 10px;
    border-radius: 9px;
}}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {{
    background: rgba(9, 127, 229, .22);
}}

    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_resources():
    schema = load_schema()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)
    return schema, model


@st.cache_data
def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    return pd.read_csv(DATA_PATH)


try:
    schema, model = load_resources()
except Exception as exc:
    st.error(f"Could not load the auditor resources: {exc}")
    st.stop()


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def normalize_column(name):
    """Normalize a column name for flexible matching."""
    return "".join(
        character.lower()
        for character in str(name)
        if character.isalnum()
    )


def detect_column(columns, aliases):
    """Find a column matching one of the supplied aliases."""
    normalized = {
        normalize_column(column): column
        for column in columns
    }

    for alias in aliases:
        if alias in normalized:
            return normalized[alias]

    return None


def suggest_derivative(value):
    """Suggest a supported derivative from a common alias."""
    normalized = normalize_column(value)

    aliases = {
        "mcua": "MCU-A",
        "mcu_a": "MCU-A",
        "a": "MCU-A",
        "mcub": "MCU-B",
        "mcu_b": "MCU-B",
        "b": "MCU-B",
        "mcuc": "MCU-C",
        "mcu_c": "MCU-C",
        "c": "MCU-C",
    }

    return aliases.get(normalized, "Unmapped")


def analyze(die_id, derivative, fuse_hex):
    """
    Decode and validate a fuse map, then get the ML prediction.

    Returns:
        decoded fields, rule findings, ML prediction,
        estimated ML error probability
    """
    record = {
        "die_id": str(die_id),
        "derivative": str(derivative),
        "fuse_map_hex": str(fuse_hex).strip(),
    }

    decoded, findings = validate_record(record, schema)

    missing_fields = [
        field for field in FUSE_FIELDS
        if field not in decoded
    ]

    if missing_fields:
        raise ValueError(
            f"Decoder did not return expected fields: {missing_fields}"
        )

    features = pd.DataFrame([
        {
            **{
                field: decoded[field]
                for field in FUSE_FIELDS
            },
            "derivative": derivative,
        }
    ])

    prediction = int(model.predict(features)[0])

    probability = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        classes = list(model.classes_)

        if 1 in classes:
            probability = float(
                probabilities[classes.index(1)]
            )

    return decoded, findings, prediction, probability


def findings_text(findings):
    """Convert validator findings to readable text."""
    if not findings:
        return "No implemented rule violations"

    messages = []

    for finding in findings:
        if isinstance(finding, dict):
            rule = finding.get("rule", "Unknown rule")
            actual = finding.get("actual", "N/A")
            expected = finding.get("expected", "N/A")

            messages.append(
                f"{rule} (actual={actual}, expected={expected})"
            )
        else:
            messages.append(str(finding))

    return "; ".join(messages)


def probability_text(value):
    """Format an optional probability."""
    if value is None or pd.isna(value):
        return "Unavailable"

    return f"{value:.1%}"


@st.cache_data(show_spinner=False)
def get_dashboard_audit(data):
    """Audit the reference dataset once and cache dashboard counts/results."""
    rows = []
    for _, row in data.iterrows():
        try:
            decoded, findings, prediction, probability = analyze(
                row.get("die_id", row.get("record_id", "DATASET_ROW")),
                row.get("derivative", "MCU-A"),
                row.get("fuse_map_hex", ""),
            )
            rows.append({
                "die_id": row.get("die_id", row.get("record_id", "DATASET_ROW")),
                "derivative": row.get("derivative", "Unknown"),
                "rule_flag": int(bool(findings)),
                "ml_prediction": int(prediction),
                "audit_status": "OK",
                "rule_findings": findings_text(findings),
                "ml_error_probability": probability,
            })
        except Exception as exc:
            rows.append({
                "die_id": row.get("die_id", row.get("record_id", "DATASET_ROW")),
                "derivative": row.get("derivative", "Unknown"),
                "rule_flag": None,
                "ml_prediction": None,
                "audit_status": f"ERROR: {exc}",
                "rule_findings": "",
                "ml_error_probability": None,
            })
    return pd.DataFrame(rows)


# ============================================================
# 4. APP HEADER AND NAVIGATION
# ============================================================

st.sidebar.markdown(
    """
    <div style="padding: 10px 2px 18px 2px;">
      <div style="font-size: 1.55rem; font-weight: 800; color: #f4faff;">
        <span style="color:#12caff;">▣</span> FuseGuard <span style="color:#12caff;">AI</span>
      </div>
      <div style="font-size:.82rem; color:#a9c7e8;">MCU Fuse Map Integrity Auditor</div>
      <div style="font-size:.72rem; color:#7fafd9; margin-top:10px;">DECODE · VALIDATE · PREDICT · AUDIT</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="hero-eyebrow">AI-powered configuration analysis</div>
      <div class="hero-title">MCU Fuse Map <span>Integrity Auditor</span></div>
      <div class="hero-subtitle">Decode. Validate. Detect. Report. — a smart workspace for MCU fuse-map inspection.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.warning(
    "Research prototype. Verify all findings against official MCU "
    "documentation. ML predictions are not a safety guarantee."
)

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard",
        "Single Fuse Map Audit",
        "Batch CSV Audit",
        "Dataset Overview",
        "Reports",
        "Settings",
    ],
)


# ============================================================
# 5. SINGLE FUSE MAP AUDIT
# ============================================================

if page == "Dashboard":
    st.markdown('<div class="section-kicker">OVERVIEW / AUDIT CONSOLE</div>', unsafe_allow_html=True)

    try:
        reference_data = load_dataset()
        total_records = len(reference_data)
        audit_summary = get_dashboard_audit(reference_data)

        successful = int((audit_summary["audit_status"] == "OK").sum())
        processing_errors = total_records - successful
        rule_flagged = int((audit_summary["rule_flag"].fillna(0) == 1).sum())
        ml_flagged = int((audit_summary["ml_prediction"].fillna(-1) == 1).sum())
        clean_records = max(total_records - rule_flagged, 0)
        rule_pct = (rule_flagged / total_records * 100) if total_records else 0
        ml_pct = (ml_flagged / total_records * 100) if total_records else 0

        # KPI cards
        st.markdown("### System Overview")
        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric("TOTAL RECORDS", f"{total_records:,}", "Reference dataset")
        with metric_cols[1]:
            st.metric("RULE VIOLATIONS", f"{rule_flagged:,}", f"{rule_pct:.1f}% of total")
        with metric_cols[2]:
            st.metric("ML-PREDICTED ERRORS", f"{ml_flagged:,}", f"{ml_pct:.1f}% of total")
        with metric_cols[3]:
            st.metric("PROCESSING ERRORS", f"{processing_errors:,}", "Rows that could not be audited")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        chart_col, insights_col, model_col = st.columns([1.12, 1.0, 0.78], gap="medium")

        with chart_col:
            with st.container(border=True):
                st.markdown("### <span style='color:#19caff'>◉</span> Error Distribution", unsafe_allow_html=True)
                chart_data = pd.DataFrame({
                    "Category": ["No rule flags", "Rule-flagged"],
                    "Records": [clean_records, rule_flagged],
                }).set_index("Category")
                st.bar_chart(chart_data, color="#11c5ff", height=230)
                st.caption(
                    f"{clean_records:,} records had no implemented rule flags; "
                    f"{rule_flagged:,} triggered at least one rule."
                )

        with insights_col:
            with st.container(border=True):
                st.markdown("### <span style='color:#19caff'>✧</span> Key Insights", unsafe_allow_html=True)
                st.markdown(
                    f"""
                    <div style="display:flex;gap:12px;align-items:flex-start;padding:9px 0;border-bottom:1px solid #163a59;">
                      <div style="font-size:1.35rem;color:#34e7a1;">●</div>
                      <div><b>Processing status</b><br><span style="color:#a9c7e8;font-size:.88rem;">{successful:,} of {total_records:,} records audited successfully.</span></div>
                    </div>
                    <div style="display:flex;gap:12px;align-items:flex-start;padding:9px 0;border-bottom:1px solid #163a59;">
                      <div style="font-size:1.35rem;color:#ffbf47;">▲</div>
                      <div><b>Rule-based findings</b><br><span style="color:#a9c7e8;font-size:.88rem;">{rule_flagged:,} records triggered implemented validation rules.</span></div>
                    </div>
                    <div style="display:flex;gap:12px;align-items:flex-start;padding:9px 0;border-bottom:1px solid #163a59;">
                      <div style="font-size:1.35rem;color:#c4a1ff;">✧</div>
                      <div><b>ML risk signal</b><br><span style="color:#a9c7e8;font-size:.88rem;">{ml_flagged:,} records were predicted as potential errors.</span></div>
                    </div>
                    <div style="display:flex;gap:12px;align-items:flex-start;padding:9px 0;">
                      <div style="font-size:1.35rem;color:#19caff;">ⓘ</div>
                      <div><b>Interpret results carefully</b><br><span style="color:#a9c7e8;font-size:.88rem;">A prediction is a review signal, not proof of a hardware defect.</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with model_col:
            with st.container(border=True):
                st.markdown("### <span style='color:#19caff'>▥</span> Model Performance", unsafe_allow_html=True)
                st.caption("Held-out test set")
                comparison_path = PROJECT_ROOT / "models" / "model_comparison.csv"
                if comparison_path.exists():
                    comparison = pd.read_csv(comparison_path)
                    st.dataframe(comparison, use_container_width=True, hide_index=True, height=225)
                else:
                    st.info("Model comparison file not found. Run src/train_model.py to generate it.")
                st.caption("Use held-out test metrics for model evaluation; full-dataset counts are audit summaries.")

        with st.container(border=True):
            title_col, search_col = st.columns([1.2, 1])
            with title_col:
                st.markdown("### <span style='color:#19caff'>▤</span> Recent Audit Results", unsafe_allow_html=True)
            with search_col:
                search_term = st.text_input("Search records", placeholder="Search Die ID or derivative...", label_visibility="collapsed")

            recent = audit_summary.copy()
            if search_term.strip():
                term = search_term.strip().lower()
                recent = recent[
                    recent["die_id"].astype(str).str.lower().str.contains(term, na=False)
                    | recent["derivative"].astype(str).str.lower().str.contains(term, na=False)
                    | recent["rule_findings"].astype(str).str.lower().str.contains(term, na=False)
                ]

            recent["Rule Check"] = recent["rule_flag"].map({1: "WARNING", 0: "PASS"}).fillna("NOT AUDITED")
            recent["ML Risk"] = recent["ml_prediction"].map({1: "PREDICTED ERROR", 0: "NO ERROR PREDICTED"}).fillna("UNAVAILABLE")
            recent["ML Probability"] = recent["ml_error_probability"].apply(probability_text)
            display_cols = ["die_id", "derivative", "Rule Check", "ML Risk", "ML Probability", "rule_findings", "audit_status"]
            st.dataframe(
                recent[display_cols].head(12),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "die_id": st.column_config.TextColumn("Die ID"),
                    "derivative": st.column_config.TextColumn("Derivative"),
                    "Rule Check": st.column_config.TextColumn("Rule Check"),
                    "ML Risk": st.column_config.TextColumn("ML Risk"),
                    "ML Probability": st.column_config.TextColumn("ML Error Probability"),
                    "rule_findings": st.column_config.TextColumn("Rule Findings", width="large"),
                    "audit_status": st.column_config.TextColumn("Processing Status"),
                },
            )
            st.download_button(
                "⬇ Download Dashboard Audit CSV",
                data=audit_summary.to_csv(index=False).encode("utf-8"),
                file_name="mcu_dashboard_audit_summary.csv",
                mime="text/csv",
                type="primary",
            )

        st.caption("Prototype notice: confirm findings against official MCU documentation and independent hardware data before production use.")

    except Exception as exc:
        st.error(f"Could not build the dashboard: {exc}")

elif page == "Single Fuse Map Audit":

    st.header("Single Fuse Map Audit")

    st.write(
        "Enter a fuse-map hexadecimal value to decode it, "
        "check implemented validation rules, and obtain an ML prediction."
    )

    default_derivative = "MCU-A"

    try:
        sample_data = load_dataset()

        if (
            "derivative" in sample_data.columns
            and not sample_data.empty
        ):
            available = [
                value for value in DERIVATIVES
                if value in sample_data["derivative"].dropna().values
            ]

            if available:
                default_derivative = available[0]

    except Exception:
        pass

    derivative = st.selectbox(
        "MCU derivative",
        DERIVATIVES,
        index=DERIVATIVES.index(default_derivative),
    )

    die_id = st.text_input(
        "Die ID",
        value="MANUAL_TEST_001",
    )

    default_hex = ""

    try:
        if (
            not sample_data.empty
            and "derivative" in sample_data.columns
            and "fuse_map_hex" in sample_data.columns
        ):
            matching = sample_data[
                sample_data["derivative"] == derivative
            ]

            if not matching.empty:
                default_hex = str(
                    matching.iloc[0]["fuse_map_hex"]
                )

    except Exception:
        pass

    fuse_hex = st.text_input(
        "Fuse map (hexadecimal)",
        value=default_hex,
        help="Enter a hexadecimal fuse-map value supported by your decoder.",
    )

    if st.button("Analyze Fuse Map", type="primary"):

        if not die_id.strip():
            st.error("Please enter a Die ID.")

        elif not fuse_hex.strip():
            st.error("Please enter a fuse-map hexadecimal value.")

        else:
            try:
                decoded, findings, prediction, probability = analyze(
                    die_id,
                    derivative,
                    fuse_hex,
                )

                st.subheader("Audit Summary")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Rule-based result",
                    "FLAGGED" if findings else "NO RULE FLAGS",
                )

                col2.metric(
                    "ML prediction",
                    "ERROR" if prediction == 1 else "NO ERROR",
                )

                col3.metric(
                    "ML error probability",
                    probability_text(probability),
                )

                st.caption(
                    "The probability is the model's estimated probability "
                    "for its error class, not a guarantee of actual risk."
                )

                st.subheader("Rule Validation Findings")

                if findings:
                    for finding in findings:
                        if isinstance(finding, dict):
                            rule = finding.get("rule", "Unknown rule")
                            actual = finding.get("actual", "N/A")
                            expected = finding.get("expected", "N/A")

                            st.error(
                                f"{rule} — actual: {actual}; "
                                f"expected: {expected}"
                            )
                        else:
                            st.error(str(finding))
                else:
                    st.success(
                        "No violations were detected by the implemented rules."
                    )

                st.subheader("Decoded Fuse Fields")

                decoded_df = pd.DataFrame([
                    {
                        "Field": field,
                        "Decoded Value": decoded[field],
                    }
                    for field in FUSE_FIELDS
                ])

                st.dataframe(
                    decoded_df,
                    use_container_width=True,
                    hide_index=True,
                )

                single_result = {
                    "die_id": die_id,
                    "derivative": derivative,
                    "fuse_map_hex": fuse_hex,
                    "rule_flag": int(bool(findings)),
                    "rule_findings": findings_text(findings),
                    "ml_prediction": prediction,
                    "ml_error_probability": probability,
                }

                for field, value in decoded.items():
                    single_result[f"decoded_{field}"] = value

                result_csv = pd.DataFrame([single_result]).to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "Download Single Audit CSV",
                    data=result_csv,
                    file_name="single_fuse_audit_result.csv",
                    mime="text/csv",
                )

            except Exception as exc:
                st.error(
                    f"Could not analyze this fuse map: {exc}"
                )


# ============================================================
# 6. BATCH CSV AUDIT
# ============================================================

elif page == "Batch CSV Audit":

    st.header("Batch CSV Audit")

    st.write(
        "Upload a new CSV file containing MCU fuse-map records. "
        "The file does not need to be one of the two project CSVs, "
        "and it does not need training labels such as has_error or "
        "error_type. It must contain a column identifying the MCU "
        "derivative (or an equivalent column) and a column containing "
        "the fuse-map hexadecimal value. You can map different column "
        "names below."
    )

    with st.expander("Input requirements and limitations"):
        st.markdown(
            "- **Required information:** MCU derivative and fuse-map hex value.\n"
            "- **Optional information:** a die/chip/record ID; otherwise the app generates row IDs.\n"
            "- **Supported derivatives:** MCU-A, MCU-B, and MCU-C. You can map source labels to these values.\n"
            "- **Training labels are not required:** `has_error` and `error_type` are not needed for inference.\n"
            "- **Not an arbitrary-CSV auditor:** unrelated CSV files or fuse formats that do not match the configured schema cannot be meaningfully audited.\n"
            "- The loaded model and rule checks use this project's existing schema and trained model; uploading a CSV does not retrain the model."
        )

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        key="batch_csv_upload",
    )

    if uploaded_file is not None:

        try:
            df = pd.read_csv(uploaded_file)

            if df.empty:
                st.warning(
                    "This CSV contains no data rows."
                )

            else:
                st.success(
                    f"Loaded {len(df):,} rows and "
                    f"{len(df.columns)} columns."
                )

                with st.expander("Preview uploaded CSV"):
                    st.dataframe(
                        df.head(10),
                        use_container_width=True,
                    )

                    st.write(
                        "Available columns:",
                        list(df.columns),
                    )

                columns = list(df.columns)

                derivative_aliases = [
                    "derivative",
                    "mcu",
                    "mcutype",
                    "chiptype",
                    "device",
                    "devicetype",
                    "partnumber",
                    "model",
                    "microcontroller",
                ]

                fuse_aliases = [
                    "fusemaphex",
                    "fusehex",
                    "fusemap",
                    "fusehexvalue",
                    "fusemapvalue",
                    "fusemapdata",
                    "rawfuse",
                    "fusedata",
                    "fusevalue",
                    "hex",
                    "rawhex",
                ]

                die_aliases = [
                    "dieid",
                    "die",
                    "chipid",
                    "deviceid",
                    "serialnumber",
                    "unitid",
                ]

                detected_derivative = detect_column(
                    columns,
                    derivative_aliases,
                )

                detected_fuse = detect_column(
                    columns,
                    fuse_aliases,
                )

                detected_die = detect_column(
                    columns,
                    die_aliases,
                )

                st.subheader("Map CSV Columns")

                st.caption(
                    "The app suggests columns based on their names. "
                    "Check the preview and verify each selection."
                )

                derivative_index = (
                    columns.index(detected_derivative)
                    if detected_derivative in columns
                    else 0
                )

                fuse_index = (
                    columns.index(detected_fuse)
                    if detected_fuse in columns
                    else min(1, len(columns) - 1)
                )

                derivative_col = st.selectbox(
                    "Column identifying the MCU derivative",
                    options=columns,
                    index=derivative_index,
                    key="batch_derivative_col",
                )

                fuse_col = st.selectbox(
                    "Column containing the fuse-map hex value",
                    options=columns,
                    index=fuse_index,
                    key="batch_fuse_col",
                )

                die_options = ["Generate row IDs"] + columns

                die_index = (
                    die_options.index(detected_die)
                    if detected_die in columns
                    else 0
                )

                die_col = st.selectbox(
                    "Column containing the Die ID (optional)",
                    options=die_options,
                    index=die_index,
                    key="batch_die_col",
                )

                # Detect distinct derivative values.
                raw_derivatives = (
                    df[derivative_col]
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .unique()
                    .tolist()
                )

                derivative_mapping = {}

                st.subheader("Confirm MCU Derivative Mapping")

                if not raw_derivatives:
                    st.error(
                        "No derivative values were found in the selected "
                        "column. Check your column selection."
                    )

                else:
                    st.caption(
                        "Map each distinct input value to MCU-A, MCU-B, "
                        "or MCU-C. Unmapped values will be reported as "
                        "invalid rows."
                    )

                    for i, raw_value in enumerate(raw_derivatives):

                        suggestion = suggest_derivative(raw_value)

                        choices = [
                            "Unmapped",
                            "MCU-A",
                            "MCU-B",
                            "MCU-C",
                        ]

                        suggested_index = (
                            choices.index(suggestion)
                            if suggestion in choices
                            else 0
                        )

                        derivative_mapping[raw_value] = st.selectbox(
                            f"CSV value '{raw_value}' represents",
                            options=choices,
                            index=suggested_index,
                            key=f"derivative_mapping_{i}",
                        )

                st.info(
                    "This auditor processes MCU fuse-map records only. "
                    "An unrelated CSV can be uploaded and previewed, but "
                    "it cannot be meaningfully audited unless it contains "
                    "the required MCU information."
                )

                selected_core_columns = [derivative_col, fuse_col]
                if die_col != "Generate row IDs":
                    selected_core_columns.append(die_col)

                if derivative_col == fuse_col:
                    st.error(
                        "Select different columns for derivative and fuse map."
                    )

                elif len(set(selected_core_columns)) != len(selected_core_columns):
                    st.error(
                        "The Die ID column must be different from the derivative "
                        "and fuse-map columns. Choose 'Generate row IDs' or select "
                        "a separate ID column."
                    )

                elif not raw_derivatives:
                    st.error(
                        "No derivative values were found. Choose the correct "
                        "derivative column before running the audit."
                    )

                elif st.button(
                    "Run Batch Audit",
                    type="primary",
                    key="run_batch_audit",
                ):

                    results = []

                    progress = st.progress(0)
                    status = st.empty()

                    total_rows = len(df)

                    for i, (_, row) in enumerate(df.iterrows()):

                        result = row.to_dict()

                        raw_derivative = (
                            str(row[derivative_col]).strip()
                            if pd.notna(row[derivative_col])
                            else ""
                        )

                        derivative = derivative_mapping.get(
                            raw_derivative,
                            "Unmapped",
                        )

                        fuse_value = row[fuse_col]

                        fuse_hex = (
                            str(fuse_value).strip()
                            if pd.notna(fuse_value)
                            else ""
                        )

                        if (
                            die_col != "Generate row IDs"
                            and pd.notna(row[die_col])
                        ):
                            die_id = str(row[die_col]).strip()
                        else:
                            die_id = f"CSV_ROW_{i + 1}"

                        result["audit_die_id"] = die_id
                        result["audit_derivative"] = derivative
                        result["audit_fuse_map_hex"] = fuse_hex

                        try:
                            if derivative not in DERIVATIVES:
                                raise ValueError(
                                    "Unmapped or unsupported derivative: "
                                    f"'{raw_derivative}'"
                                )

                            if not fuse_hex:
                                raise ValueError(
                                    "Fuse-map value is empty."
                                )

                            decoded, findings, prediction, probability = (
                                analyze(
                                    die_id,
                                    derivative,
                                    fuse_hex,
                                )
                            )

                            result["rule_flag"] = int(bool(findings))
                            result["rule_findings"] = findings_text(
                                findings
                            )
                            result["ml_prediction"] = prediction
                            result["ml_error_probability"] = probability
                            result["audit_status"] = "OK"

                            for field, value in decoded.items():
                                result[f"decoded_{field}"] = value

                        except Exception as exc:
                            result["rule_flag"] = None
                            result["rule_findings"] = ""
                            result["ml_prediction"] = None
                            result["ml_error_probability"] = None
                            result["audit_status"] = f"ERROR: {exc}"

                        results.append(result)

                        progress.progress(
                            (i + 1) / total_rows
                        )

                        status.text(
                            f"Processed {i + 1:,} of {total_rows:,} rows"
                        )

                    results_df = pd.DataFrame(results)

                    successful = int(
                        (results_df["audit_status"] == "OK").sum()
                    )

                    failed = total_rows - successful

                    flagged = int(
                        (
                            results_df["rule_flag"].fillna(0) == 1
                        ).sum()
                    )

                    predicted_errors = int(
                        (
                            results_df["ml_prediction"].fillna(-1) == 1
                        ).sum()
                    )

                    st.subheader("Batch Audit Summary")

                    col1, col2, col3, col4 = st.columns(4)

                    col1.metric(
                        "Total Rows",
                        f"{total_rows:,}",
                    )

                    col2.metric(
                        "Successfully Audited",
                        f"{successful:,}",
                    )

                    col3.metric(
                        "Rows with Processing Errors",
                        f"{failed:,}",
                    )

                    col4.metric(
                        "Rule-Flagged Rows",
                        f"{flagged:,}",
                    )

                    st.metric(
                        "ML-Predicted Error Rows",
                        f"{predicted_errors:,}",
                    )

                    st.subheader("Audit Results")

                    status_filter = st.selectbox(
                        "Filter results",
                        [
                            "All rows",
                            "Successfully audited",
                            "Processing errors",
                            "Rule-flagged rows",
                            "ML-predicted errors",
                        ],
                    )

                    filtered_df = results_df.copy()

                    if status_filter == "Successfully audited":
                        filtered_df = filtered_df[
                            filtered_df["audit_status"] == "OK"
                        ]

                    elif status_filter == "Processing errors":
                        filtered_df = filtered_df[
                            filtered_df["audit_status"] != "OK"
                        ]

                    elif status_filter == "Rule-flagged rows":
                        filtered_df = filtered_df[
                            filtered_df["rule_flag"].fillna(0) == 1
                        ]

                    elif status_filter == "ML-predicted errors":
                        filtered_df = filtered_df[
                            filtered_df["ml_prediction"].fillna(-1) == 1
                        ]

                    st.dataframe(
                        filtered_df,
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.download_button(
                        "Download Complete Audit Results CSV",
                        data=results_df.to_csv(
                            index=False
                        ).encode("utf-8"),
                        file_name="mcu_fuse_audit_results.csv",
                        mime="text/csv",
                    )

                    error_df = results_df[
                        results_df["audit_status"] != "OK"
                    ]

                    if not error_df.empty:
                        st.download_button(
                            "Download Rows Needing Review",
                            data=error_df.to_csv(
                                index=False
                            ).encode("utf-8"),
                            file_name="mcu_fuse_rows_needing_review.csv",
                            mime="text/csv",
                        )

                    st.caption(
                        "A row marked ERROR was not successfully audited. "
                        "Rule flags and ML predictions are separate signals. "
                        "A negative ML prediction does not prove a fuse map "
                        "is valid. The uploaded CSV is evaluated using the "
                        "existing trained model and configured MCU schema; "
                        "the model is not retrained from the uploaded file."
                    )

        except Exception as exc:
            st.error(
                f"Could not read this CSV file: {exc}"
            )


# ============================================================
# 7. DATASET OVERVIEW
# ============================================================

elif page == "Dataset Overview":

    st.header("Dataset Overview")

    try:
        data = load_dataset()

        st.write(
            "This page summarizes the project's reference dataset, "
            "not the CSV currently uploaded in Batch CSV Audit."
        )

        if data.empty:
            st.warning("The reference dataset is empty.")

        else:
            total_records = len(data)

            if "has_error" in data.columns:
                error_values = pd.to_numeric(
                    data["has_error"],
                    errors="coerce",
                )

                error_count = int(
                    (error_values == 1).sum()
                )

                clean_count = int(
                    (error_values == 0).sum()
                )

                error_rate = (
                    error_count / total_records * 100
                )

            else:
                error_count = 0
                clean_count = 0
                error_rate = 0.0

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Total Records",
                f"{total_records:,}",
            )

            col2.metric(
                "Error Records",
                f"{error_count:,}",
            )

            col3.metric(
                "Clean Records",
                f"{clean_count:,}",
            )

            col4.metric(
                "Error Rate",
                f"{error_rate:.2f}%",
            )

            if "derivative" in data.columns:
                st.subheader("Records by MCU Derivative")

                derivative_counts = (
                    data["derivative"]
                    .fillna("Unknown")
                    .value_counts()
                )

                st.bar_chart(derivative_counts)

            if "error_type" in data.columns:
                st.subheader("Error Category Distribution")

                error_data = data.copy()

                if "has_error" in error_data.columns:
                    error_data = error_data[
                        pd.to_numeric(
                            error_data["has_error"],
                            errors="coerce",
                        ) == 1
                    ]

                error_counts = (
                    error_data["error_type"]
                    .fillna("Unknown")
                    .value_counts()
                )

                if not error_counts.empty:
                    st.bar_chart(error_counts)
                else:
                    st.info(
                        "No error categories are available."
                    )

            st.subheader("Dataset Preview")

            st.dataframe(
                data.head(20),
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download Reference Dataset",
                data=data.to_csv(index=False).encode("utf-8"),
                file_name="mcu_fuse_reference_dataset.csv",
                mime="text/csv",
            )

    except Exception as exc:
        st.error(
            f"Could not load the reference dataset: {exc}"
        )


elif page == "Reports":
    st.markdown('<div class="section-kicker">Exports / Documentation</div>', unsafe_allow_html=True)
    st.subheader("Audit Reports")
    st.write("Download the reference dataset or run a batch audit to generate a detailed record-level report.")
    try:
        report_data = load_dataset()
        st.metric("Reference dataset rows", f"{len(report_data):,}")
        st.download_button(
            "Download Reference Dataset CSV",
            data=report_data.to_csv(index=False).encode("utf-8"),
            file_name="mcu_fuse_reference_dataset.csv",
            mime="text/csv",
            type="primary",
        )
    except Exception as exc:
        st.error(f"Could not load report data: {exc}")
    st.info("Detailed audit-result downloads are available after running Single Fuse Map Audit or Batch CSV Audit.")

elif page == "Settings":
    st.markdown('<div class="section-kicker">Configuration / About</div>', unsafe_allow_html=True)
    st.subheader("Auditor Settings & Information")
    st.markdown(
        "- **Supported derivatives:** MCU-A, MCU-B, MCU-C\n"
        "- **Input format:** hexadecimal fuse-map values and compatible CSV files\n"
        "- **Model mode:** offline inference using the saved local model\n"
        "- **Schema:** the configured fuse-field schema shipped with this project"
    )
    st.warning("Do not treat ML predictions as proof of a hardware fault or a clean bill of health. Confirm findings against official MCU specifications.")
    st.caption("For a different MCU family or fuse layout, update and validate the schema and model before use.")


# ============================================================
# 8. FOOTER
# ============================================================

st.divider()

st.caption(
    "AI-Based MCU Fuse Map Integrity Auditor | "
    "Research prototype. Confirm all findings against official "
    "hardware specifications before taking engineering action."
)
