from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# =========================
# Config
# =========================

REQUIRED_CORE = ["requests", "responses", "impressions", "revenue"]

# At least one must exist
ENTITY_ANY_OF = [
    "publisher_id",
    "publisher_name",
    "advertiser_id",
    "advertiser_name",
]

SUPPLY_ANY_OF = [
    "bundle_id",
    "supplier_id",
    "site",
    "app",
]

ALIASES: Dict[str, str] = {
    # Core
    "requests": "requests",
    "req": "requests",
    "bid_requests": "requests",
    "bidrequest": "requests",
    "ad_requests": "requests",

    "responses": "responses",
    "resps": "responses",
    "resp": "responses",
    "bid_responses": "responses",
    "bidresponse": "responses",

    "impressions": "impressions",
    "imps": "impressions",
    "served_impressions": "impressions",

    "revenue": "revenue",
    "rev": "revenue",
    "publisher_revenue": "revenue",

    "bid_response": "responses",        # BID_RESPONSE
    "bid_responses": "responses",
    "gross_revenue": "revenue",        # GROSS_REVENUE
    "demand_partner_id": "advertiser_id",  # DEMAND_PARTNER_ID
    "supply_partner_id": "supplier_id",    # SUPPLY_PARTNER_ID
    "bundle": "bundle_id",

    # IVT (optional)
    "sivt": "sivt",
    "sivt_count": "sivt",
    "givt": "givt",
    "givt_count": "givt",

    # IDs / names
    "bundle_id": "bundle_id",
    "app_bundle": "bundle_id",
    "bundle": "bundle_id",
    "package_name": "bundle_id",

    "supplier_id": "supplier_id",
    "ssp_id": "supplier_id",
    "supply_partner_id": "supplier_id",
    "seller_id": "supplier_id",

    "advertiser_id": "advertiser_id",
    "buyer_id": "advertiser_id",
    "dsp_advertiser_id": "advertiser_id",

    "publisher_id": "publisher_id",
    "pub_id": "publisher_id",

    "publisher_name": "publisher_name",
    "pub_name": "publisher_name",

    "advertiser_name": "advertiser_name",
    "buyer_name": "advertiser_name",

    # Dimensions (optional)
    "date": "date",
    "day": "date",
    "campaign_id": "campaign_id",
    "campaign": "campaign_id",
    "site": "site",
    "app": "app",
    "format": "format",
    "geo": "geo",
    "country": "geo",
    "device": "device",
}


# =========================
# Helpers
# =========================

def norm_col_name(col: str) -> str:
    col = str(col).strip().lower()
    col = re.sub(r"[^\w]+", "_", col)  # non-word -> _
    col = re.sub(r"_+", "_", col).strip("_")
    return col


def coerce_numeric(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        s = series.astype(str).str.replace(",", "", regex=False)
        s = s.str.replace("$", "", regex=False)
        s = s.str.replace("€", "", regex=False)
        s = s.str.replace("%", "", regex=False)
        s = s.str.strip()
        return pd.to_numeric(s, errors="coerce")
    return pd.to_numeric(series, errors="coerce")


def safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    numer = pd.to_numeric(numer, errors="coerce")
    denom = pd.to_numeric(denom, errors="coerce")
    out = np.where((denom > 0) & np.isfinite(denom), numer / denom, np.nan)
    return pd.Series(out, index=numer.index)


def read_input(path: Path, sheet: Optional[str] = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()

    if suffix in [".xlsx", ".xlsm", ".xls"]:
        # פותחים פעם אחת בלבד
        xls = pd.ExcelFile(path, engine="openpyxl")
        sheet_names = xls.sheet_names

        # אם streamlit שולח "" במקום None
        if sheet is not None:
            sheet = str(sheet).strip()
            if sheet == "":
                sheet = None

        if sheet:
            if sheet not in sheet_names:
                raise ValueError(f"Sheet '{sheet}' not found. Available: {sheet_names}")
            chosen_sheet = sheet
        else:
            chosen_sheet = pick_best_sheet_xls(xls, path)
            print(f"[INFO] pick_best_sheet chose: {chosen_sheet}")

        print(f"[INFO] Available sheets: {sheet_names}")
        print(f"[INFO] Reading sheet: {chosen_sheet}")

        # חשוב: משתמשים ב-xls.parse כדי לא לטעון workbook מחדש
        return xls.parse(sheet_name=chosen_sheet)

    if suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported file type: {suffix}. Use .xlsx/.xls/.csv")





def normalize_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    # Step 1: normalize raw names
    raw_to_norm = {c: norm_col_name(c) for c in df.columns}
    df = df.rename(columns=raw_to_norm)

    # Step 2: map aliases -> canonical
    mapped_cols = []
    for c in df.columns:
        mapped_cols.append(ALIASES.get(c, c))
    df.columns = mapped_cols

    # Step 3: handle duplicate columns after mapping
    # e.g. req + bid_requests -> both become "requests"
    col_counts: Dict[str, int] = {}
    new_cols: List[str] = []
    for c in df.columns:
        if c not in col_counts:
            col_counts[c] = 0
            new_cols.append(c)
        else:
            col_counts[c] += 1
            new_cols.append(f"{c}__dup{col_counts[c]}")
    df.columns = new_cols

    # Merge dup columns into base column using first non-null
    bases = {c.split("__dup")[0] for c in df.columns}
    for b in bases:
        dup_cols = [c for c in df.columns if c == b or c.startswith(f"{b}__dup")]
        if len(dup_cols) > 1:
            df[b] = df[dup_cols].bfill(axis=1).iloc[:, 0]
            drop_cols = [c for c in dup_cols if c != b]
            df = df.drop(columns=drop_cols)

    return df, raw_to_norm


def validate_schema(df: pd.DataFrame) -> None:
    missing_core = [c for c in REQUIRED_CORE if c not in df.columns]
    if missing_core:
        raise ValueError(f"Missing required core columns: {missing_core}")

    if not any(c in df.columns for c in ENTITY_ANY_OF):
        raise ValueError(
            f"Missing entity identity fields. Need at least one of: {ENTITY_ANY_OF}"
        )

    if not any(c in df.columns for c in SUPPLY_ANY_OF):
        raise ValueError(
            f"Missing supply identity fields. Need at least one of: {SUPPLY_ANY_OF}"
        )


def clean_and_cast(df: pd.DataFrame) -> pd.DataFrame:
    # Parse date if exists
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Numeric cast for known numeric fields if present
    numeric_fields = [
        "requests", "responses", "impressions", "revenue", "sivt", "givt"
    ]
    for c in numeric_fields:
        if c in df.columns:
            df[c] = coerce_numeric(df[c])

    return df


def logic_checks(df: pd.DataFrame, allow_invalid: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # invalid masks
    cond_resp_gt_req = df["responses"] > df["requests"]
    cond_imps_gt_resp = df["impressions"] > df["responses"]
    cond_neg_core = (df[["requests", "responses", "impressions", "revenue"]] < 0).any(axis=1)

    invalid_mask = cond_resp_gt_req | cond_imps_gt_resp | cond_neg_core
    invalid_rows = df[invalid_mask].copy()

    if invalid_rows.empty:
        return df, invalid_rows

    if allow_invalid:
        # Drop invalid and continue
        return df.loc[~invalid_mask].copy(), invalid_rows

    # Strict mode: fail hard
    n_total = len(df)
    n_bad = len(invalid_rows)
    raise ValueError(
        f"Logic check failed in strict mode: {n_bad}/{n_total} invalid rows "
        f"(responses>requests OR impressions>responses OR negative core values)."
    )


def compute_kpis(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["response_rate"] = safe_div(out["responses"], out["requests"])
    out["fill_rate"] = safe_div(out["impressions"], out["requests"])
    out["imps_per_response"] = safe_div(out["impressions"], out["responses"])
    out["rpm"] = safe_div(out["revenue"] * 1000.0, out["impressions"])
    out["rpr"] = safe_div(out["revenue"], out["responses"])

    if "sivt" in out.columns:
        out["sivt_rate"] = safe_div(out["sivt"], out["impressions"])
    if "givt" in out.columns:
        out["givt_rate"] = safe_div(out["givt"], out["impressions"])
    if "sivt" in out.columns and "givt" in out.columns:
        out["ivt_total_rate"] = safe_div(out["sivt"] + out["givt"], out["impressions"])

    return out


def add_benchmarks(
    df: pd.DataFrame,
    min_peer_requests: int = 5000,
) -> pd.DataFrame:
    out = df.copy()

    # Global benchmark (ratio-of-sums)
    g_req = out["requests"].sum(skipna=True)
    g_resp = out["responses"].sum(skipna=True)
    g_imps = out["impressions"].sum(skipna=True)
    g_rev = out["revenue"].sum(skipna=True)

    global_response = (g_resp / g_req) if g_req > 0 else np.nan
    global_fill = (g_imps / g_req) if g_req > 0 else np.nan
    global_rpm = ((g_rev * 1000.0) / g_imps) if g_imps > 0 else np.nan

    out["benchmark_response_global"] = global_response
    out["benchmark_fill_global"] = global_fill
    out["benchmark_rpm_global"] = global_rpm

    # Peer key: advertiser_id + supplier_id if possible, else fallback
    peer_keys = [k for k in ["advertiser_id", "supplier_id"] if k in out.columns]
    if not peer_keys:
        peer_keys = [k for k in ["advertiser_name", "publisher_id", "bundle_id"] if k in out.columns][:1]

    if peer_keys:
        grp = out.groupby(peer_keys, dropna=False, as_index=False).agg(
            req_sum=("requests", "sum"),
            resp_sum=("responses", "sum"),
            imps_sum=("impressions", "sum"),
            rev_sum=("revenue", "sum"),
        )
        grp["benchmark_response_peer"] = np.where(grp["req_sum"] > 0, grp["resp_sum"] / grp["req_sum"], np.nan)
        grp["benchmark_fill_peer"] = np.where(grp["req_sum"] > 0, grp["imps_sum"] / grp["req_sum"], np.nan)
        grp["benchmark_rpm_peer"] = np.where(grp["imps_sum"] > 0, (grp["rev_sum"] * 1000.0) / grp["imps_sum"], np.nan)

        out = out.merge(
            grp[peer_keys + ["req_sum", "benchmark_response_peer", "benchmark_fill_peer", "benchmark_rpm_peer"]],
            on=peer_keys,
            how="left",
        )

        # Fallback: if peer volume too low -> global
        enough_peer = out["req_sum"] >= min_peer_requests
        out["benchmark_response"] = np.where(enough_peer, out["benchmark_response_peer"], out["benchmark_response_global"])
        out["benchmark_fill"] = np.where(enough_peer, out["benchmark_fill_peer"], out["benchmark_fill_global"])
        out["benchmark_rpm"] = np.where(enough_peer, out["benchmark_rpm_peer"], out["benchmark_rpm_global"])
    else:
        # no peer keys at all
        out["benchmark_response"] = out["benchmark_response_global"]
        out["benchmark_fill"] = out["benchmark_fill_global"]
        out["benchmark_rpm"] = out["benchmark_rpm_global"]
        out["req_sum"] = np.nan

    out["gap_response_vs_benchmark"] = out["response_rate"] - out["benchmark_response"]
    out["gap_fill_vs_benchmark"] = out["fill_rate"] - out["benchmark_fill"]
    out["gap_rpm_vs_benchmark"] = out["rpm"] - out["benchmark_rpm"]

    return out


def build_recommendations(
    df: pd.DataFrame,
    min_requests_eval: int = 1000,
    min_responses_eval: int = 50,
    max_ivt_rate: float = 0.03,
) -> pd.DataFrame:
    rows: List[Dict] = []

    for idx, r in df.iterrows():
        req = r.get("requests", np.nan)
        resp = r.get("responses", np.nan)
        imps = r.get("impressions", np.nan)
        rev = r.get("revenue", np.nan)

        if pd.isna(req) or pd.isna(resp) or req < min_requests_eval or resp < min_responses_eval:
            continue

        rr = r.get("response_rate", np.nan)
        fr = r.get("fill_rate", np.nan)
        rpm = r.get("rpm", np.nan)

        b_rr = r.get("benchmark_response", np.nan)
        b_fr = r.get("benchmark_fill", np.nan)
        b_rpm = r.get("benchmark_rpm", np.nan)

        # Basic impact estimates
        expected_imps = req * b_fr if pd.notna(b_fr) else np.nan
        lost_imps = max(0.0, expected_imps - imps) if pd.notna(expected_imps) and pd.notna(imps) else 0.0

        expected_rev_fill = (expected_imps * b_rpm / 1000.0) if pd.notna(expected_imps) and pd.notna(b_rpm) else np.nan
        lost_rev_fill = max(0.0, expected_rev_fill - rev) if pd.notna(expected_rev_fill) and pd.notna(rev) else 0.0

        expected_rev_rpm = (imps * b_rpm / 1000.0) if pd.notna(imps) and pd.notna(b_rpm) else np.nan
        lost_rev_rpm = max(0.0, expected_rev_rpm - rev) if pd.notna(expected_rev_rpm) and pd.notna(rev) else 0.0

        entity = {
            "date": str(r["date"].date()) if "date" in r and pd.notna(r["date"]) else None,
            "advertiser_id": r.get("advertiser_id"),
            "advertiser_name": r.get("advertiser_name"),
            "supplier_id": r.get("supplier_id"),
            "bundle_id": r.get("bundle_id"),
            "publisher_id": r.get("publisher_id"),
            "publisher_name": r.get("publisher_name"),
            "campaign_id": r.get("campaign_id"),
        }

        issue_types: List[str] = []
        actions: List[str] = []
        priority = 0.0

        # 1) Demand constraint
        if pd.notna(rr) and pd.notna(b_rr) and (rr < 0.8 * b_rr):
            issue_types.append("Demand Constraint")
            actions.append(
                "Response rate נמוך מול benchmark: לבדוק bid request quality, geo/device/format targeting, "
                "ו-QPS throttling בין SSP->DSP."
            )
            priority += float(lost_rev_fill) * 0.5 + 10

        # 2) Supply / delivery constraint
        if pd.notna(fr) and pd.notna(b_fr) and (fr < 0.8 * b_fr) and (pd.notna(rr) and pd.notna(b_rr) and rr >= 0.9 * b_rr):
            issue_types.append("Supply/Delivery Constraint")
            actions.append(
                "Fill נמוך למרות response סביר: לבדוק floor price, bundle/site quality, latency, "
                "win-rate post-response, ו-creative/render failures."
            )
            priority += float(lost_rev_fill) + 20

        # 3) Monetization problem
        if pd.notna(rpm) and pd.notna(b_rpm) and (rpm < 0.8 * b_rpm) and (pd.notna(fr) and pd.notna(b_fr) and fr >= 0.9 * b_fr):
            issue_types.append("Monetization Problem")
            actions.append(
                "RPM נמוך מול benchmark: לבחון pricing/floor segmentation, deal mix (open vs PMP), "
                "והתאמת traffic לביקוש איכותי יותר."
            )
            priority += float(lost_rev_rpm) + 15

        # 4) IVT risk
        ivt_total = r.get("ivt_total_rate", np.nan)
        if pd.notna(ivt_total) and ivt_total > max_ivt_rate:
            issue_types.append("IVT Risk")
            actions.append(
                "IVT גבוה: להחריג ספקים/באנדלים חשודים, להקשיח pre-bid filters, ולפתוח בדיקה מול anti-fraud provider."
            )
            priority += 25

        if not issue_types:
            continue

        rows.append(
            {
                **entity,
                "issue_types": " | ".join(issue_types),
                "priority_score": round(priority, 2),
                "requests": req,
                "responses": resp,
                "impressions": imps,
                "revenue": rev,
                "response_rate": rr,
                "fill_rate": fr,
                "rpm": rpm,
                "benchmark_response": b_rr,
                "benchmark_fill": b_fr,
                "benchmark_rpm": b_rpm,
                "estimated_lost_impressions": round(float(lost_imps), 2),
                "estimated_lost_revenue_fill_based": round(float(lost_rev_fill), 2),
                "estimated_lost_revenue_rpm_based": round(float(lost_rev_rpm), 2),
                "recommended_actions": " || ".join(actions),
            }
        )

    recs = pd.DataFrame(rows)
    if not recs.empty:
        recs = recs.sort_values(["priority_score"], ascending=False).reset_index(drop=True)
    return recs

def build_action_plan(recs: pd.DataFrame, top_n: Optional[int] = 50) -> pd.DataFrame:
    columns_out = [
        "priority",
        "advertiser_id",
        "supplier_id",
        "flags",
        "total_est_loss",
        "dominant_issue",
        "owner",
        "immediate_action_48h",
        "experiment_7d",
        "success_kpi",
    ]

    if recs is None or recs.empty:
        return pd.DataFrame(columns=columns_out)

    df = recs.copy()

    # Ensure loss columns exist
    for c in ["estimated_lost_revenue_fill_based", "estimated_lost_revenue_rpm_based"]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df["loss_total_row"] = df["estimated_lost_revenue_fill_based"] + df["estimated_lost_revenue_rpm_based"]

    # Group keys preference
    group_keys = [c for c in ["advertiser_id", "supplier_id"] if c in df.columns]
    if len(group_keys) < 2:
        group_keys = [c for c in ["advertiser_id", "supplier_id", "bundle_id", "publisher_id"] if c in df.columns][:2]
    if len(group_keys) == 0:
        df["__all__"] = "all"
        group_keys = ["__all__"]

    def dominant_issue(series: pd.Series) -> str:
        tokens: List[str] = []
        for s in series.dropna().astype(str):
            parts = [p.strip() for p in s.split("|")]
            tokens.extend([p for p in parts if p])
        if not tokens:
            return "Unknown"
        vc = pd.Series(tokens).value_counts()
        return str(vc.index[0])

    agg = (
        df.groupby(group_keys, dropna=False)
          .agg(
              flags=("loss_total_row", "count"),
              total_est_loss=("loss_total_row", "sum"),
              dominant_issue=("issue_types", dominant_issue) if "issue_types" in df.columns else ("loss_total_row", "count"),
          )
          .reset_index()
    )

    # If issue_types not exists, fix dominant_issue values
    if "issue_types" not in df.columns:
        agg["dominant_issue"] = "Unknown"

    # Priority buckets by ranking (robust גם כשיש מעט שורות)
    agg = agg.sort_values("total_est_loss", ascending=False).reset_index(drop=True)
    n = len(agg)
    if n == 0:
        return pd.DataFrame(columns=columns_out)

    p1_cut = max(1, int(np.ceil(n * 0.2)))   # top 20%
    p2_cut = max(p1_cut + 1, int(np.ceil(n * 0.5)))  # next 30%

    agg["priority"] = "P3"
    agg.loc[: p1_cut - 1, "priority"] = "P1"
    agg.loc[p1_cut : p2_cut - 1, "priority"] = "P2"

    def owner(issue: str) -> str:
        i = str(issue)
        if "IVT" in i:
            return "Fraud + AdOps"
        if "Supply/Delivery Constraint" in i:
            return "AdOps + Publisher Success"
        if "Demand Constraint" in i:
            return "AdOps + Demand"
        if "Monetization Problem" in i:
            return "Yield/RevOps"
        return "AdOps"

    def immediate_action(issue: str) -> str:
        i = str(issue)
        if "IVT" in i:
            return "להקשיח pre-bid filters, לחסום מקורות חשודים זמנית, ולפתוח audit."
        if "Supply/Delivery Constraint" in i:
            return "לבדוק floor/latency/render failures ולבודד bundles חלשים."
        if "Demand Constraint" in i:
            return "להרחיב demand coverage ולהרפות מגבלות טרגוט קשיחות."
        if "Monetization Problem" in i:
            return "להריץ A/B על floor (±10%) ולשפר deal mix (Open/PMP)."
        return "לעבור על funnel request→response→impression ולאתר root cause."

    def experiment_7d(issue: str) -> str:
        i = str(issue)
        if "IVT" in i:
            return "ניסוי שבועי: חסימה מדורגת + מדידת IVT/Revenue לפני-אחרי."
        if "Supply/Delivery Constraint" in i:
            return "ניסוי שבועי: timeout tuning + floor segmentation לפי geo/device/format."
        if "Demand Constraint" in i:
            return "ניסוי שבועי: הרחבת partners בסגמנטים חלשים ומדידת uplift."
        if "Monetization Problem" in i:
            return "ניסוי שבועי: 2–3 וריאציות pricing ובחירת policy מנצחת."
        return "ניסוי שבועי מבוקר על הפרמטר המשפיע ביותר."

    def success_kpi(issue: str) -> str:
        i = str(issue)
        if "IVT" in i:
            return "IVT rate ↓ לפחות 30% ללא ירידה ב-Revenue."
        if "Supply/Delivery Constraint" in i:
            return "Fill rate ↑ 15%+ ו-Imps/Response ↑ 10%+."
        if "Demand Constraint" in i:
            return "Response rate ↑ 15%+ ו-Fill rate ↑ 8%+."
        if "Monetization Problem" in i:
            return "RPM ↑ 10%+ תוך שמירה על Fill יציב."
        return "שיפור KPI ראשי מול benchmark."

    agg["owner"] = agg["dominant_issue"].apply(owner)
    agg["immediate_action_48h"] = agg["dominant_issue"].apply(immediate_action)
    agg["experiment_7d"] = agg["dominant_issue"].apply(experiment_7d)
    agg["success_kpi"] = agg["dominant_issue"].apply(success_kpi)

    # Guarantee columns
    if "advertiser_id" not in agg.columns:
        agg["advertiser_id"] = np.nan
    if "supplier_id" not in agg.columns:
        agg["supplier_id"] = np.nan

    out = agg[
        [
            "priority",
            "advertiser_id",
            "supplier_id",
            "flags",
            "total_est_loss",
            "dominant_issue",
            "owner",
            "immediate_action_48h",
            "experiment_7d",
            "success_kpi",
        ]
    ].copy()

    # Sort by priority then loss
    prio_order = {"P1": 0, "P2": 1, "P3": 2}
    out["_p"] = out["priority"].map(prio_order).fillna(9)
    out = out.sort_values(["_p", "total_est_loss"], ascending=[True, False]).drop(columns=["_p"]).reset_index(drop=True)

    if top_n is not None and top_n > 0:
        out = out.head(top_n).copy()

    return out

def summarize(df: pd.DataFrame, recs: pd.DataFrame, invalid_rows: pd.DataFrame) -> Dict:
    out = {
        "rows_after_cleaning": int(len(df)),
        "invalid_rows_quarantined": int(len(invalid_rows)),
        "recommendations_count": int(len(recs)),
        "totals": {
            "requests": float(df["requests"].sum(skipna=True)),
            "responses": float(df["responses"].sum(skipna=True)),
            "impressions": float(df["impressions"].sum(skipna=True)),
            "revenue": float(df["revenue"].sum(skipna=True)),
        },
        "kpis_global": {
            "response_rate": float(df["responses"].sum(skipna=True) / df["requests"].sum(skipna=True)) if df["requests"].sum(skipna=True) > 0 else None,
            "fill_rate": float(df["impressions"].sum(skipna=True) / df["requests"].sum(skipna=True)) if df["requests"].sum(skipna=True) > 0 else None,
            "rpm": float((df["revenue"].sum(skipna=True) * 1000.0) / df["impressions"].sum(skipna=True)) if df["impressions"].sum(skipna=True) > 0 else None,
        },
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="AdTech Excel Analyzer + Auto Recommendations")
    parser.add_argument("--input", required=True, help="Path to input file (.xlsx/.xls/.csv)")
    parser.add_argument("--sheet", default=None, help="Excel sheet name (optional)")
    parser.add_argument("--output-dir", default="outputs", help="Directory for output files")
    parser.add_argument("--allow-invalid", action="store_true", help="If set: drop invalid logic rows instead of failing")
    parser.add_argument("--min-peer-requests", type=int, default=5000, help="Min peer volume for benchmark fallback")
    parser.add_argument("--min-requests-eval", type=int, default=1000, help="Min requests for recommendations")
    parser.add_argument("--min-responses-eval", type=int, default=50, help="Min responses for recommendations")
    parser.add_argument("--max-ivt-rate", type=float, default=0.03, help="IVT threshold")
    parser.add_argument("--action-plan-top-n", type=int, default=50, help="Max rows in action plan")
    args = parser.parse_args()

    print(f"[INFO] input={args.input}")
    print(f"[INFO] sheet={args.sheet}")
    print(f"[INFO] output_dir={args.output_dir}")

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

def pick_best_sheet_xls(xls: pd.ExcelFile, path: Path) -> str:
    sheet_names = xls.sheet_names
    if not sheet_names:
        raise ValueError("Workbook has no sheets.")

    # 1) ניסיון מהיר: שמות נפוצים
    preferred = {
        "main", "main_dataset", "dataset", "data", "report", "raw", "export", "sheet1"
    }
    for s in sheet_names:
        if norm_col_name(s) in preferred:
            return s

    # 2) סריקה חכמה: לא לסרוק אינסוף שיטים
    # אם יש הרבה שיטים – סורקים רק את הראשונים כדי לא להיתקע
    max_scan = min(len(sheet_names), 8)

    best_sheet: Optional[str] = None
    best_score = -10**9
    errors: Dict[str, str] = {}

    for s in sheet_names[:max_scan]:
        try:
            # קוראים רק header (nrows=0) + מעט שורות כדי לוודא שיש דאטה
            hdr = xls.parse(sheet_name=s, nrows=0)
            mapped = [ALIASES.get(norm_col_name(c), norm_col_name(c)) for c in hdr.columns]

            core_hits = sum(c in mapped for c in REQUIRED_CORE)
            entity_hits = sum(c in mapped for c in ENTITY_ANY_OF)
            supply_hits = sum(c in mapped for c in SUPPLY_ANY_OF)

            sample = xls.parse(sheet_name=s, nrows=5)
            has_data_bonus = 1 if sample.dropna(how="all").shape[0] > 0 else 0

            score = core_hits * 100 + entity_hits * 10 + supply_hits * 10 + has_data_bonus

            if score > best_score:
                best_score = score
                best_sheet = s

        except Exception as e:
            errors[s] = str(e)

    # 3) fallback: אם לא מצאנו משהו חכם – קח את השיט הראשון
    if best_sheet is None:
        print(f"[WARN] Could not auto-detect best sheet, falling back to first. Errors: {errors}")
        return sheet_names[0]

    return best_sheet


def main() -> None:
    parser = argparse.ArgumentParser(description="AdTech Excel Analyzer + Auto Recommendations")
    parser.add_argument("--input", required=True, help="Path to input file (.xlsx/.xls/.csv)")
    parser.add_argument("--sheet", default=None, help="Excel sheet name (optional)")
    parser.add_argument("--output-dir", default="outputs", help="Directory for output files")
    parser.add_argument("--allow-invalid", action="store_true", help="If set: drop invalid logic rows instead of failing")
    parser.add_argument("--min-peer-requests", type=int, default=5000, help="Min peer volume for benchmark fallback")
    parser.add_argument("--min-requests-eval", type=int, default=1000, help="Min requests for recommendations")
    parser.add_argument("--min-responses-eval", type=int, default=50, help="Min responses for recommendations")
    parser.add_argument("--max-ivt-rate", type=float, default=0.03, help="IVT threshold")
    parser.add_argument("--action-plan-top-n", type=int, default=50, help="Max rows in action plan")
    args = parser.parse_args()

    print(f"[INFO] input={args.input}")
    print(f"[INFO] sheet={args.sheet}")
    print(f"[INFO] output_dir={args.output_dir}")

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1) Load
    print("[INFO] stage=read_input")
    df = read_input(input_path, sheet=args.sheet)

    print(f"[INFO] loaded rows={len(df)} cols={list(df.columns)}")
    if df is None or df.empty:
        raise ValueError(
            "No rows were loaded from the input file. "
            "Most common cause: wrong sheet name or empty sheet. "
            "Try leaving --sheet empty or choose a different sheet."
        )

    # 2) Normalize + validate
    print("[INFO] stage=normalize+validate")
    df, _raw_to_norm = normalize_columns(df)
    validate_schema(df)

    # 3) Clean + logic checks
    print("[INFO] stage=clean+logic")
    df = clean_and_cast(df)
    df, invalid_rows = logic_checks(df, allow_invalid=args.allow_invalid)

    # 4) KPIs + benchmarks
    print("[INFO] stage=kpis+benchmarks")
    df = compute_kpis(df)
    df = add_benchmarks(df, min_peer_requests=args.min_peer_requests)

    # 5) Recommendations + action plan
    print("[INFO] stage=recommendations+action_plan")
    recs = build_recommendations(
        df,
        min_requests_eval=args.min_requests_eval,
        min_responses_eval=args.min_responses_eval,
        max_ivt_rate=args.max_ivt_rate,
    )
    action_plan = build_action_plan(recs, top_n=args.action_plan_top_n)

    # 6) Save outputs
    print("[INFO] stage=save_outputs")
    enriched_path = output_dir / f"enriched_kpis_{ts}.csv"
    recs_path = output_dir / f"recommendations_{ts}.csv"
    invalid_path = output_dir / f"invalid_rows_{ts}.csv"
    summary_path = output_dir / f"summary_{ts}.json"
    action_plan_path = output_dir / f"action_plan_{ts}.csv"

    df.to_csv(enriched_path, index=False)
    recs.to_csv(recs_path, index=False)
    action_plan.to_csv(action_plan_path, index=False)


    if not invalid_rows.empty:
        invalid_rows.to_csv(invalid_path, index=False)

    summary = summarize(df, recs, invalid_rows)
    summary["action_plan_count"] = int(len(action_plan))
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 7) Console summary
    print("✅ Done")
    print(f"Input: {input_path}")
    print(f"Rows after cleaning: {len(df)}")
    print(f"Recommendations: {len(recs)}")
    print(f"Invalid quarantined: {len(invalid_rows)}")
    print(f"Action plan rows: {len(action_plan)}")
    print(f"Saved: {enriched_path}")
    print(f"Saved: {recs_path}")
    if not invalid_rows.empty:
        print(f"Saved: {invalid_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {action_plan_path}")
    print(f"Saved: {action_plan_latest_path}")


if __name__ == "__main__":
    main()
