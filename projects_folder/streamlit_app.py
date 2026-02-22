import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

SHOW_RUN_LOG = os.getenv("SHOW_RUN_LOG", "false").strip().lower() == "true"


# =========================================================
# Path resolution (supports running from project root or /streamlit subfolder)
# =========================================================
THIS_FILE = Path(__file__).resolve()

if (THIS_FILE.parent / "src" / "app.py").exists():
    # file is in project root (rare in your setup)
    PROJECT_ROOT = THIS_FILE.parent
else:
    # file is in project_root/streamlit/streamlit_app.py (your setup)
    PROJECT_ROOT = THIS_FILE.parent.parent

APP_SCRIPT = PROJECT_ROOT / "src" / "app.py"
RUNS_ROOT = PROJECT_ROOT / "outputs" / "streamlit_runs"
RUNS_ROOT.mkdir(parents=True, exist_ok=True)


# =========================================================
# App / Page
# =========================================================
st.set_page_config(
    page_title="Ad Campaign Action Planner",
    layout="wide",
    initial_sidebar_state="collapsed",  # סיידבר סגור בדיפולט
)

# --- UI tweaks: blue accents / no red action buttons ---
st.markdown(
    """
    <style>
    /* Primary button (Analyze) -> blue */
    div.stButton > button[kind="primary"] {
        background-color: #1d4ed8 !important;
        border: 1px solid #2563eb !important;
        color: white !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #2563eb !important;
        border-color: #3b82f6 !important;
    }

    /* Secondary buttons (including downloads in many Streamlit versions) -> blue outline */
    div.stButton > button[kind="secondary"],
    div.stDownloadButton > button {
        border: 1px solid #2563eb !important;
        color: #bfdbfe !important;
        background: rgba(37, 99, 235, 0.08) !important;
    }
    div.stButton > button[kind="secondary"]:hover,
    div.stDownloadButton > button:hover {
        border-color: #3b82f6 !important;
        color: white !important;
        background: rgba(37, 99, 235, 0.18) !important;
    }

    /* Focus rings / inputs */
    button:focus, input:focus, textarea:focus, [role="tab"]:focus {
        box-shadow: 0 0 0 0.15rem rgba(59,130,246,0.35) !important;
        outline: none !important;
    }

    /* ===== Tabs (text + hover + selected + underline) ===== */

    /* Default tab label color */
    .stTabs [data-baseweb="tab"] [data-testid="stMarkdownContainer"] p {
        color: #e5e7eb !important;
    }

    /* Hover tab label color */
    .stTabs [data-baseweb="tab"]:hover [data-testid="stMarkdownContainer"] p {
        color: #93c5fd !important;
    }

    /* Active tab label color */
    .stTabs [data-baseweb="tab"][aria-selected="true"] [data-testid="stMarkdownContainer"] p {
        color: #93c5fd !important;
        font-weight: 600 !important;
    }

    /* Active tab underline (real indicator in BaseWeb) */
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #3b82f6 !important;
    }

    /* Fallback for some Streamlit versions (if underline is border-based) */
    button[role="tab"][aria-selected="true"] {
        color: #93c5fd !important;
        border-bottom-color: #3b82f6 !important;
    }

    /* Optional: links hover (if any text turns red because it's a link) */
    a, a:visited {
        color: #93c5fd !important;
    }
    a:hover, a:active, a:focus {
        color: #60a5fa !important;
    }

    /* Expander header subtle blue tint on hover */
    [data-testid="stExpander"] details summary:hover {
        background: rgba(37, 99, 235, 0.06);
        border-radius: 8px;
    }

    /* ===== Custom file-help expander with inline info tooltip ===== */
    .file-help-wrap {
        margin: 0.35rem 0 0.75rem 0;
    }

    .file-help {
        border: 1px solid rgba(148, 163, 184, 0.20);
        border-radius: 10px;
        background: rgba(15, 23, 42, 0.25);
        overflow: visible; /* IMPORTANT: allow tooltip to overflow */
    }

    .file-help summary {
        list-style: none;
        cursor: pointer;
        padding: 0.7rem 0.9rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        user-select: none;
        position: relative;
        border-radius: 10px;
    }

    .file-help summary::-webkit-details-marker {
        display: none;
    }

    .file-help summary:hover {
        background: rgba(37, 99, 235, 0.06);
    }

    .file-help-chevron {
        width: 10px;
        height: 10px;
        border-right: 2px solid #93c5fd;
        border-bottom: 2px solid #93c5fd;
        transform: rotate(-45deg);
        transition: transform 0.18s ease;
        margin-top: -2px;
        flex: 0 0 auto;
    }

    .file-help[open] .file-help-chevron {
        transform: rotate(45deg);
        margin-top: -4px;
    }

    .file-help-title {
        color: #e5e7eb;
        font-weight: 600;
        font-size: 0.95rem;
        line-height: 1.2;
    }

    /* info icon + tooltip */
    .file-help-info {
        position: relative;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 22px;
        height: 22px;
        border-radius: 999px;
        border: 1px solid rgba(59, 130, 246, 0.35);
        background: rgba(37, 99, 235, 0.10);
        color: #bfdbfe;
        flex: 0 0 auto;
        margin-left: 2px;
    }

    .file-help-info svg {
        width: 14px;
        height: 14px;
        display: block;
    }

    /* Tooltip on hover/focus */
    .file-help-tooltip {
        visibility: hidden;
        opacity: 0;
        transition: opacity 0.15s ease;
        position: absolute;
        top: 28px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 9999;
        width: 430px;
        max-width: min(430px, 82vw);
        background: #0b1220;
        border: 1px solid rgba(59, 130, 246, 0.30);
        border-radius: 10px;
        padding: 10px 12px;
        color: #dbeafe;
        line-height: 1.4;
        font-size: 0.86rem;
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.35);
        pointer-events: none;
        text-align: left;
        white-space: normal;
    }

    .file-help-info:hover .file-help-tooltip,
    .file-help-info:focus-within .file-help-tooltip {
        visibility: visible;
        opacity: 1;
    }

    /* Body */
    .file-help-body {
        border-top: 1px solid rgba(148, 163, 184, 0.12);
        padding: 0.75rem 0.95rem 0.8rem 0.95rem;
        color: #e5e7eb;
        font-size: 0.9rem;
        line-height: 1.45;
        background: rgba(2, 6, 23, 0.15);
        border-radius: 0 0 10px 10px;
    }

    .file-help-body ul {
        margin: 0.35rem 0 0.6rem 1.1rem;
        padding: 0;
    }

    .file-help-body li {
        margin: 0.2rem 0;
    }

    .file-help-body code {
        background: rgba(148, 163, 184, 0.12);
        border: 1px solid rgba(148, 163, 184, 0.15);
        padding: 0.05rem 0.35rem;
        border-radius: 6px;
        color: #dbeafe;
    }
        /* Make the popover button look like a small dark icon button */
    div[data-testid="stPopover"] button {
        min-height: 2.2rem !important;
        height: 2.2rem !important;
        padding: 0 !important;
        border-radius: 10px !important;
        border: 1px solid rgba(59,130,246,0.30) !important;
        background: rgba(37,99,235,0.08) !important;
        color: #bfdbfe !important;
        font-size: 0.95rem !important;
    }
    div[data-testid="stPopover"] button:hover {
        border-color: #3b82f6 !important;
        background: rgba(37,99,235,0.16) !important;
        color: #dbeafe !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Ad Campaign Action Planner")
st.caption("Upload Excel/CSV → Analyze → Get Executive Action Plan")

# שורה אחת: expander + אייקון מידע
# --- Upload guidance (visible short text + expander) ---
st.markdown(
    f"""
    <div style="
        margin: 0.25rem 0 0.45rem 0;
        padding: 0.65rem 0.8rem;
        border-radius: 10px;
        border: 1px solid rgba(59,130,246,0.22);
        background: rgba(37,99,235,0.08);
        color: #dbeafe;
        font-size: 0.92rem;
        line-height: 1.4;
    ">
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("**Required Fields:** requests, responses, impressions, revenue + one entity field "
    "(advertiser/publisher) and one supply field (supplier/bundle/site/app). "
    "**Recommended:** bundle_id, date, geo/country, device, format, campaign_id, "
    "and IVT fields (SIVT/GIVT). "
    "_Click for more info below._", expanded=False):
    st.markdown(
        """
**Required metrics** (exact names not required if common aliases exist):
- `requests`
- `responses`
- `impressions`
- `revenue`

**Identity fields** (at least one entity + one supply identifier):
- **Entity (at least one):** `advertiser_id` / `advertiser_name` / `publisher_id` / `publisher_name`
- **Supply (at least one):** `supplier_id` / `bundle_id` / `site` / `app`

**Recommended for deeper optimization:**
- IVT / fraud signals: `sivt`, `givt` (or equivalent aliases)
- Segmentation: `date`, `country`/`geo`, `device`, `format`
- Commercial dimensions: `campaign_id`, `bundle_id`, `supplier_id`, `advertiser_id`

**Formatting tips:**
- First row = headers
- No merged cells
- Metric columns should be numeric (the app auto-cleans common formats like commas / `$` / `%`)
- Avoid totals/summary rows mixed into raw data
- In multi-sheet Excel, if no sheet is selected, the app tries to auto-pick the best sheet
"""
    )

# =========================================================
# Helpers
# =========================================================
def latest_file(folder: Path, pattern: str):
    files = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def guess_mime(p: Path) -> str:
    s = p.suffix.lower()
    if s == ".csv":
        return "text/csv"
    if s == ".json":
        return "application/json"
    if s in [".xlsx", ".xlsm"]:
        return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    if s == ".xls":
        return "application/vnd.ms-excel"
    return "application/octet-stream"


def render_artifacts_downloads(run_dir: Path, run_id: str):
    st.subheader("Artifacts (Downloads)")
    files = sorted([p for p in run_dir.glob("*") if p.is_file()], key=lambda x: x.name)

    if not files:
        st.info("No files were created in this run folder.")
        return

    for p in files:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.write(p.name)
        with col2:
            st.download_button(
                label="Download",
                data=p.read_bytes(),
                file_name=p.name,
                mime=guess_mime(p),
                key=f"dl_{run_id}_{p.name}",
                use_container_width=True,
            )


# =========================================================
# Session State Init
# =========================================================
if "analysis_ready" not in st.session_state:
    st.session_state.analysis_ready = False
    st.session_state.last_run_id = None
    st.session_state.last_run_dir = None
    st.session_state.summary = {}
    st.session_state.df_action = None
    st.session_state.df_recs = None
    st.session_state.df_enriched = None
    st.session_state.df_invalid = None
    st.session_state.run_log_text = ""
    st.session_state.last_cmd = ""
    st.session_state.last_returncode = None
    st.session_state.last_error = ""


# =========================================================
# Sidebar
# =========================================================
with st.sidebar:
    st.header("Settings")
    sheet_name = st.text_input("Sheet name (optional)", value="")
    allow_invalid = st.checkbox("Allow invalid rows (quarantine)", value=True)

    with st.expander("Thresholds", expanded=False):
        min_peer_requests = st.number_input("min_peer_requests", min_value=0, value=10000, step=1000)
        min_requests_eval = st.number_input("min_requests_eval", min_value=0, value=5000, step=500)
        min_responses_eval = st.number_input("min_responses_eval", min_value=0, value=200, step=50)
        max_ivt_rate = st.number_input(
            "max_ivt_rate",
            min_value=0.0,
            max_value=1.0,
            value=0.03,
            step=0.005,
            format="%.3f",
        )


# =========================================================
# Upload
# =========================================================
uploaded = st.file_uploader("Upload .xlsx / .xls / .csv", type=["xlsx", "xls", "csv"])

if uploaded is not None:
    # הצלחה תישאר ירוק - לפי הבקשה שלך 👌
    st.success(f"Loaded: {uploaded.name}")


# =========================================================
# Run
# =========================================================
run_clicked = st.button("Analyze", type="primary", disabled=(uploaded is None))

if run_clicked:
    if not APP_SCRIPT.exists():
        st.session_state.analysis_ready = False
        st.session_state.last_error = f"Analyzer script not found: {APP_SCRIPT}"
    else:
        # תיקייה קבועה לריצה (לא temp), כדי שתמיד נוכל לראות ולהוריד תוצרים
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = RUNS_ROOT / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        input_path = run_dir / uploaded.name
        with open(input_path, "wb") as f:
            f.write(uploaded.getbuffer())

        cmd = [
            sys.executable,
            str(APP_SCRIPT),
            "--input", str(input_path),
            "--output-dir", str(run_dir),
            "--min-peer-requests", str(int(min_peer_requests)),
            "--min-requests-eval", str(int(min_requests_eval)),
            "--min-responses-eval", str(int(min_responses_eval)),
            "--max-ivt-rate", str(float(max_ivt_rate)),
        ]

        if sheet_name.strip():
            cmd += ["--sheet", sheet_name.strip()]
        if allow_invalid:
            cmd += ["--allow-invalid"]

        with st.spinner("Running analysis..."):
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT),
            )

        # שמירת Run Log ל-session_state כדי שישרוד reruns (כמו לחיצה על Download)
        run_log_text = (
            f"CWD: {PROJECT_ROOT}\n"
            f"APP_SCRIPT: {APP_SCRIPT}\n"
            f"CMD: {' '.join(cmd)}\n\n"
            f"STDOUT:\n{proc.stdout or '<empty>'}\n\n"
            f"STDERR:\n{proc.stderr or '<empty>'}"
        )
        st.session_state.run_log_text = run_log_text
        st.session_state.last_cmd = " ".join(cmd)
        st.session_state.last_returncode = proc.returncode
        st.session_state.last_error = ""

        if proc.returncode != 0:
            st.session_state.analysis_ready = False
            if SHOW_RUN_LOG:
                st.session_state.last_error = "Analysis failed. Check Run Log below."
            else:
                st.session_state.last_error = (
                    "Analysis failed. Please verify the file structure/columns and try again. "
                    "If needed, use the sheet name setting for multi-sheet Excel files."
                )
        else:
            # מציאת קבצי פלט מהריצה הזו
            summary_path = latest_file(run_dir, "summary_*.json")
            action_path = latest_file(run_dir, "action_plan_*.csv") or (
                run_dir / "action_plan_latest.csv" if (run_dir / "action_plan_latest.csv").exists() else None
            )
            recs_path = latest_file(run_dir, "recommendations_*.csv")
            enriched_path = latest_file(run_dir, "enriched_kpis_*.csv")
            invalid_path = latest_file(run_dir, "invalid_rows_*.csv")

            summary = {}
            if summary_path and summary_path.exists():
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary = json.load(f)

            # טעינת טבלאות
            df_action = pd.read_csv(action_path) if action_path and Path(action_path).exists() else pd.DataFrame()
            df_recs = pd.read_csv(recs_path) if recs_path and Path(recs_path).exists() else pd.DataFrame()
            df_enriched = pd.read_csv(enriched_path) if enriched_path and Path(enriched_path).exists() else pd.DataFrame()
            df_invalid = pd.read_csv(invalid_path) if invalid_path and Path(invalid_path).exists() else pd.DataFrame()

            # שמירה ל-session_state -> הדשבורד ממשיך להופיע גם אחרי Download
            st.session_state.analysis_ready = True
            st.session_state.last_run_id = run_id
            st.session_state.last_run_dir = str(run_dir)
            st.session_state.summary = summary or {}
            st.session_state.df_action = df_action
            st.session_state.df_recs = df_recs
            st.session_state.df_enriched = df_enriched
            st.session_state.df_invalid = df_invalid


# =========================================================
# Persistent Render (survives reruns)
# Order requested:
# 1) Big metrics dashboard
# 2) Tabs (analysis tables)
# 3) Downloads
# 4) Run Log collapsed (bottom, debug-only)
# =========================================================
if st.session_state.last_error:
    st.error(st.session_state.last_error)

if st.session_state.analysis_ready and st.session_state.last_run_dir:
    run_id = st.session_state.last_run_id
    run_dir = Path(st.session_state.last_run_dir)
    summary = st.session_state.summary or {}

    # --- Big summary cards ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows after cleaning", int(summary.get("rows_after_cleaning", 0)))
    c2.metric("Recommendations", int(summary.get("recommendations_count", 0)))
    c3.metric("Invalid quarantined", int(summary.get("invalid_rows_quarantined", 0)))
    c4.metric("Action plan rows", int(summary.get("action_plan_count", 0)))

    # --- Global KPIs ---
    kpis = summary.get("kpis_global", {})
    if kpis:
        st.markdown("### Global KPIs")
        k1, k2, k3 = st.columns(3)
        k1.metric("Response Rate", f"{float(kpis.get('response_rate', 0)) * 100:.2f}%")
        k2.metric("Fill Rate", f"{float(kpis.get('fill_rate', 0)) * 100:.2f}%")
        k3.metric("RPM", f"{float(kpis.get('rpm', 0)):.4f}")

    # --- Tabs (הפילוח) ---
    tab1, tab2, tab3, tab4 = st.tabs(["Action Plan", "Recommendations", "Enriched KPIs", "Invalid Rows"])

    with tab1:
        df_action = st.session_state.df_action if st.session_state.df_action is not None else pd.DataFrame()
        if not df_action.empty:
            st.dataframe(df_action, use_container_width=True, height=500)
            st.download_button(
                "Download Action Plan CSV",
                df_action.to_csv(index=False).encode("utf-8"),
                file_name="action_plan.csv",
                mime="text/csv",
                key=f"dl_action_plan_table_{run_id}",
            )
        else:
            st.info("No action plan file found.")

    with tab2:
        df_recs = st.session_state.df_recs if st.session_state.df_recs is not None else pd.DataFrame()
        if not df_recs.empty:
            st.dataframe(df_recs, use_container_width=True, height=500)
            st.download_button(
                "Download Recommendations CSV",
                df_recs.to_csv(index=False).encode("utf-8"),
                file_name="recommendations.csv",
                mime="text/csv",
                key=f"dl_recommendations_table_{run_id}",
            )
        else:
            st.info("No recommendations file found.")

    with tab3:
        df_enriched = st.session_state.df_enriched if st.session_state.df_enriched is not None else pd.DataFrame()
        if not df_enriched.empty:
            st.dataframe(df_enriched.head(2000), use_container_width=True, height=500)
            st.download_button(
                "Download Enriched KPIs CSV",
                df_enriched.to_csv(index=False).encode("utf-8"),
                file_name="enriched_kpis.csv",
                mime="text/csv",
                key=f"dl_enriched_table_{run_id}",
            )
        else:
            st.info("No enriched KPI file found.")

    with tab4:
        df_invalid = st.session_state.df_invalid if st.session_state.df_invalid is not None else pd.DataFrame()
        if not df_invalid.empty:
            st.dataframe(df_invalid, use_container_width=True, height=350)
            st.download_button(
                "Download Invalid Rows CSV",
                df_invalid.to_csv(index=False).encode("utf-8"),
                file_name="invalid_rows.csv",
                mime="text/csv",
                key=f"dl_invalid_table_{run_id}",
            )
        else:
            st.info("No invalid rows file found (or strict mode passed clean).")

    # --- Downloads block (moved BELOW the analysis tables) ---
    st.markdown("---")
    render_artifacts_downloads(run_dir, run_id)

# Empty / initial state message
else:
    st.info("Upload a file and click Analyze to generate the dashboard.")

# --- Run Log (debug-only, hidden for regular users) ---
if SHOW_RUN_LOG and st.session_state.run_log_text:
    st.markdown("---")
    with st.expander("Run Log", expanded=False):
        st.code(st.session_state.run_log_text, language="bash")
