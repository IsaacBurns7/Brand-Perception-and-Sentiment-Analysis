from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import altair as alt
import pandas as pd
import requests
import streamlit as st


API_BASE_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT_SECONDS = 2
SENTIMENT_DOMAIN = [-1, 1]
BRAND_NAME = "Brand Perception Dashboard"
CHART_HEIGHT = 248


def build_mock_payload() -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    days = [now - timedelta(days=offset) for offset in range(13, -1, -1)]
    avg_sentiments = [-0.18, -0.12, -0.08, -0.02, 0.03, 0.07, 0.11, 0.16, 0.12, 0.09, 0.14, 0.19, 0.17, 0.21]
    doc_counts = [8, 9, 7, 10, 12, 11, 14, 16, 13, 15, 14, 18, 17, 20]

    points: list[dict[str, Any]] = []
    for idx, day in enumerate(days):
        trailing = avg_sentiments[max(0, idx - 6) : idx + 1]
        count_window = doc_counts[max(0, idx - 6) : idx + 1]
        points.append(
            {
                "day": day.isoformat().replace("+00:00", "Z"),
                "doc_count": doc_counts[idx],
                "avg_sentiment": round(avg_sentiments[idx], 4),
                "sentiment_7d": round(sum(trailing) / len(trailing), 4),
                "count_7d": round(sum(count_window), 4),
            }
        )

    documents = [
        {
            "doc_id": "doc-001",
            "created_utc": (now - timedelta(hours=3)).isoformat().replace("+00:00", "Z"),
            "source": "Reuters",
            "topic": "Product Launch",
            "sentiment_label": "positive",
            "sentiment": 0.42,
        },
        {
            "doc_id": "doc-002",
            "created_utc": (now - timedelta(hours=7)).isoformat().replace("+00:00", "Z"),
            "source": "The Verge",
            "topic": "Customer Experience",
            "sentiment_label": "neutral",
            "sentiment": 0.03,
        },
        {
            "doc_id": "doc-003",
            "created_utc": (now - timedelta(hours=11)).isoformat().replace("+00:00", "Z"),
            "source": "Bloomberg",
            "topic": "Earnings",
            "sentiment_label": "negative",
            "sentiment": -0.27,
        },
        {
            "doc_id": "doc-004",
            "created_utc": (now - timedelta(hours=20)).isoformat().replace("+00:00", "Z"),
            "source": "TechCrunch",
            "topic": "AI Strategy",
            "sentiment_label": "positive",
            "sentiment": 0.31,
        },
        {
            "doc_id": "doc-005",
            "created_utc": (now - timedelta(days=1, hours=5)).isoformat().replace("+00:00", "Z"),
            "source": "CNBC",
            "topic": "Partnerships",
            "sentiment_label": "positive",
            "sentiment": 0.22,
        },
    ]

    return {
        "health": {
            "status": "degraded",
            "storage_backend": "mock",
            "processed_documents": 184,
        },
        "summary": {
            "document_count": 184,
            "source_count": 12,
            "topic_count": 8,
            "brand_count": 4,
            "average_sentiment": 0.128,
            "positive_share": 0.57,
            "negative_share": 0.18,
            "neutral_share": 0.25,
            "first_document_at": days[0].isoformat().replace("+00:00", "Z"),
            "last_document_at": days[-1].isoformat().replace("+00:00", "Z"),
        },
        "timeseries": {
            "rolling_window_days": 7,
            "points": points,
        },
        "sources": {
            "sources": [
                {"source": "Reuters", "document_count": 46, "row_count": 46, "average_sentiment": 0.19, "topic_count": 5},
                {"source": "Bloomberg", "document_count": 33, "row_count": 33, "average_sentiment": 0.06, "topic_count": 4},
                {"source": "The Verge", "document_count": 28, "row_count": 28, "average_sentiment": 0.14, "topic_count": 4},
                {"source": "TechCrunch", "document_count": 24, "row_count": 24, "average_sentiment": 0.21, "topic_count": 3},
                {"source": "CNBC", "document_count": 19, "row_count": 19, "average_sentiment": 0.02, "topic_count": 3},
            ]
        },
        "topics": {
            "topics": [
                {"topic": "AI Strategy", "document_count": 41, "row_count": 41, "average_sentiment": 0.17, "source_count": 4},
                {"topic": "Product Launch", "document_count": 38, "row_count": 38, "average_sentiment": 0.24, "source_count": 5},
                {"topic": "Customer Experience", "document_count": 29, "row_count": 29, "average_sentiment": 0.08, "source_count": 4},
                {"topic": "Earnings", "document_count": 26, "row_count": 26, "average_sentiment": -0.05, "source_count": 3},
                {"topic": "Partnerships", "document_count": 21, "row_count": 21, "average_sentiment": 0.15, "source_count": 3},
            ]
        },
        "aspects": {
            "aspects": [
                {"aspect": "pricing", "document_count": 36, "row_count": 43, "average_sentiment": -0.04, "topic_count": 4, "brand_count": 3},
                {"aspect": "product quality", "document_count": 32, "row_count": 37, "average_sentiment": 0.21, "topic_count": 5, "brand_count": 4},
                {"aspect": "customer support", "document_count": 24, "row_count": 29, "average_sentiment": 0.09, "topic_count": 3, "brand_count": 2},
                {"aspect": "innovation", "document_count": 19, "row_count": 23, "average_sentiment": 0.28, "topic_count": 4, "brand_count": 3},
            ]
        },
        "changepoints": {
            "rolling_window_days": 7,
            "penalty": 3.0,
            "changepoints": [
                {"day": days[4].isoformat().replace("+00:00", "Z"), "avg_sentiment": 0.03, "doc_count": 12},
                {"day": days[11].isoformat().replace("+00:00", "Z"), "avg_sentiment": 0.19, "doc_count": 18},
            ],
        },
        "recent_documents": documents,
    }


def safe_get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    response = requests.get(
        f"{API_BASE_URL}{path}",
        params=params,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=30, show_spinner=False)
def load_dashboard_data() -> dict[str, Any]:
    try:
        health = safe_get("/health")
        summary = safe_get("/analytics/summary")
        timeseries = safe_get("/analytics/timeseries")
        sources = safe_get("/analytics/sources")
        topics = safe_get("/analytics/topics")
        aspects = safe_get("/analytics/aspects")
        changepoints = safe_get("/analytics/changepoints")

        return {
            "mode": "live",
            "health": health,
            "summary": summary,
            "timeseries": timeseries,
            "sources": sources,
            "topics": topics,
            "aspects": aspects,
            "changepoints": changepoints,
            "recent_documents": [],
            "error": None,
        }
    except requests.RequestException as exc:
        mock = build_mock_payload()
        mock["mode"] = "mock"
        mock["error"] = str(exc)
        return mock


def prepare_timeseries_frame(points: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(points)
    if frame.empty:
        return frame
    frame["day"] = pd.to_datetime(frame["day"], utc=True)
    return frame.sort_values("day")


def prepare_breakdown_frame(items: list[dict[str, Any]], label_column: str) -> pd.DataFrame:
    frame = pd.DataFrame(items)
    if frame.empty:
        return frame
    return frame.sort_values("document_count", ascending=False).reset_index(drop=True).rename(columns={label_column: "label"})


def format_percentage(value: float) -> str:
    return f"{value:.0%}"


def format_sentiment(value: float) -> str:
    return f"{value:+.2f}"


def format_timestamp(value: str | None) -> str:
    if not value:
        return "n/a"
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        return "n/a"
    return timestamp.strftime("%b %d, %Y")


def render_intro(summary: dict[str, Any]) -> None:
    last_updated = format_timestamp(summary.get("last_document_at"))
    st.markdown(
        f"""
        <div class="hero-panel">
            <div>
                <p class="eyebrow">Executive View</p>
                <h1>{BRAND_NAME}</h1>
                <p class="hero-copy">
                    Track overall brand sentiment, source mix, topic coverage, and notable shifts in tone in one view.
                    This dashboard is designed for fast readouts in working sessions and presentations.
                </p>
            </div>
            <div class="hero-meta">
                <div class="hero-meta-label">Data through</div>
                <div class="hero-meta-value">{last_updated}</div>
                <div class="hero-meta-subtle">Aspect and topic are intentionally modeled as separate concepts.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_banner(data: dict[str, Any]) -> None:
    health = data["health"]
    if data["mode"] == "live":
        status_class = "status-live"
        title = "Live data connected"
        body = (
            f"Showing current API results from the `{health.get('storage_backend', 'unknown')}` backend. "
            f"{health.get('processed_documents', 0):,} processed documents are available."
        )
    else:
        status_class = "status-sample"
        title = "Sample data in use"
        body = (
            "The API could not be reached, so the dashboard is displaying curated sample data. "
            "Layouts, labels, and interactions remain presentation-ready while live model outputs are still limited."
        )

    st.markdown(
        f"""
        <div class="status-banner {status_class}">
            <div>
                <div class="status-title">{title}</div>
                <div class="status-body">{body}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if data["mode"] != "live" and data.get("error"):
        st.caption(f"Connection detail: {data['error']}")


def render_section_header(title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="section-heading">
            <h3>{title}</h3>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def is_aspect_data_ready(aspects: list[dict[str, Any]]) -> bool:
    if not aspects:
        return False
    labels = {str(item.get("aspect", "")).strip().lower() for item in aspects if item.get("aspect")}
    return bool(labels) and labels != {"general"}


def render_metric_cards(summary: dict[str, Any]) -> None:
    card_1, card_2, card_3, card_4 = st.columns(4)
    card_1.metric("Processed Documents", f"{summary.get('document_count', 0):,}")
    card_2.metric("Average Sentiment", format_sentiment(summary.get("average_sentiment", 0.0)))
    card_3.metric("Active Sources", f"{summary.get('source_count', 0):,}")
    card_4.metric("Tracked Topics", f"{summary.get('topic_count', 0):,}")


def render_sentiment_chart(timeseries_df: pd.DataFrame) -> None:
    if timeseries_df.empty:
        st.info("No time series data available.")
        return

    line_data = timeseries_df.melt(
        id_vars=["day", "doc_count"],
        value_vars=["avg_sentiment", "sentiment_7d"],
        var_name="series",
        value_name="sentiment",
    )
    line_data["series"] = line_data["series"].map(
        {
            "avg_sentiment": "Daily average",
            "sentiment_7d": "7-day rolling average",
        }
    )

    line = (
        alt.Chart(line_data)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("day:T", title="Date"),
            y=alt.Y("sentiment:Q", title="Sentiment", scale=alt.Scale(domain=SENTIMENT_DOMAIN)),
            color=alt.Color(
                "series:N",
                title="Series",
                scale=alt.Scale(range=["#1d4ed8", "#0f766e"]),
            ),
            tooltip=[
                alt.Tooltip("day:T", title="Date"),
                alt.Tooltip("series:N", title="Series"),
                alt.Tooltip("sentiment:Q", title="Sentiment", format=".3f"),
                alt.Tooltip("doc_count:Q", title="Documents"),
            ],
        )
        .properties(height=CHART_HEIGHT)
        .configure_axis(
            labelColor="#334155",
            titleColor="#0f172a",
            gridColor="#e2e8f0",
            domainColor="#cbd5e1",
            tickColor="#cbd5e1",
        )
        .configure_legend(
            titleColor="#0f172a",
            labelColor="#334155",
            orient="top",
            direction="horizontal",
        )
    )

    st.altair_chart(line, use_container_width=True)


def render_breakdown_chart(frame: pd.DataFrame, title: str, color: str) -> None:
    if frame.empty:
        st.info(f"No data available for {title.lower()}.")
        return

    chart = (
        alt.Chart(frame)
        .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6, color=color)
        .encode(
            x=alt.X("document_count:Q", title="Documents"),
            y=alt.Y("label:N", title=None, sort="-x"),
            tooltip=[
                alt.Tooltip("label:N", title="Name"),
                alt.Tooltip("document_count:Q", title="Documents"),
                alt.Tooltip("average_sentiment:Q", title="Avg sentiment", format=".3f"),
            ],
        )
        .properties(height=CHART_HEIGHT)
        .configure_axis(
            labelColor="#334155",
            titleColor="#0f172a",
            gridColor="#e2e8f0",
            domainColor="#cbd5e1",
            tickColor="#cbd5e1",
        )
    )

    st.altair_chart(chart, use_container_width=True)


def render_changepoints(changepoints: list[dict[str, Any]]) -> None:
    if not changepoints:
        st.info("No changepoints detected.")
        return

    frame = pd.DataFrame(changepoints)
    frame["day"] = pd.to_datetime(frame["day"], utc=True)
    frame["day"] = frame["day"].dt.strftime("%Y-%m-%d")
    frame = frame.rename(
        columns={
            "day": "Date",
            "avg_sentiment": "Average Sentiment",
            "doc_count": "Documents",
        }
    )
    st.dataframe(frame, use_container_width=True, hide_index=True)


def render_aspect_breakdown(aspect_df: pd.DataFrame, mode: str) -> None:
    if not is_aspect_data_ready(aspect_df.to_dict(orient="records")):
        status_copy = (
            "Live summary analytics are available, but aspect-level outputs are not yet rich enough to present."
            if mode == "live"
            else "Sample fallback is active. Aspect-level model outputs can be shown here as soon as they are available."
        )
        st.markdown(
            f"""
            <div class="placeholder-panel">
                <div class="placeholder-kicker">Aspect-ready</div>
                <div class="placeholder-title">Aspect Breakdown is reserved and ready for model output</div>
                <div class="placeholder-copy">
                    {status_copy}
                    This section will remain distinct from topic coverage so audiences can separate what coverage is about from
                    which product or service dimensions are being discussed.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    render_breakdown_chart(aspect_df, "Aspect Breakdown", "#7c3aed")


def render_coverage(summary: dict[str, Any]) -> None:
    coverage = pd.DataFrame(
        [
            {"Sentiment Mix": "Positive", "Share": summary.get("positive_share", 0.0)},
            {"Sentiment Mix": "Neutral", "Share": summary.get("neutral_share", 0.0)},
            {"Sentiment Mix": "Negative", "Share": summary.get("negative_share", 0.0)},
        ]
    )
    coverage["Share Label"] = coverage["Share"].map(format_percentage)
    chart = (
        alt.Chart(coverage)
        .mark_bar(cornerRadius=8)
        .encode(
            x=alt.X("Share:Q", title="Share", axis=alt.Axis(format="%")),
            y=alt.Y("Sentiment Mix:N", title=None, sort=["Positive", "Neutral", "Negative"]),
            color=alt.Color(
                "Sentiment Mix:N",
                scale=alt.Scale(
                    domain=["Positive", "Neutral", "Negative"],
                    range=["#059669", "#64748b", "#dc2626"],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("Sentiment Mix:N", title="Category"),
                alt.Tooltip("Share:Q", title="Share", format=".0%"),
            ],
        )
        .properties(height=180)
        .configure_axis(
            labelColor="#334155",
            titleColor="#0f172a",
            gridColor="#e2e8f0",
            domainColor="#cbd5e1",
            tickColor="#cbd5e1",
        )
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption(
        f"Coverage window: {format_timestamp(summary.get('first_document_at'))} to {format_timestamp(summary.get('last_document_at'))}"
    )


def render_recent_documents(recent_documents: list[dict[str, Any]], mode: str) -> None:
    if recent_documents:
        frame = pd.DataFrame(recent_documents)
        if "created_utc" in frame.columns:
            frame["created_utc"] = pd.to_datetime(frame["created_utc"], utc=True).dt.strftime("%Y-%m-%d %H:%M UTC")
        display_columns = [
            column
            for column in ["doc_id", "created_utc", "source", "topic", "sentiment_label", "sentiment"]
            if column in frame.columns
        ]
        st.dataframe(frame[display_columns], use_container_width=True, hide_index=True)
        return

    if mode == "live":
        st.markdown(
            """
            <div class="placeholder-panel">
                <div class="placeholder-kicker">Pipeline placeholder</div>
                <div class="placeholder-title">Recent document feed is not available from the live API yet</div>
                <div class="placeholder-copy">
                    Summary analytics are live, but document-level rows have not been exposed by the backend.
                    This section is intentionally reserved so the presentation layout stays stable when document previews are added.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Recent processed documents are unavailable.")


def main() -> None:
    st.set_page_config(
        page_title=BRAND_NAME,
        page_icon=":bar_chart:",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 1.5rem;
            max-width: 1280px;
        }
        [data-testid="column"] {
            gap: 0.85rem;
        }
        .stMainBlockContainer .element-container {
            margin-bottom: 0.35rem;
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(37, 99, 235, 0.10), transparent 28%),
                radial-gradient(circle at top right, rgba(5, 150, 105, 0.08), transparent 24%),
                linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
            color: #0f172a;
        }
        h1, h2, h3, p, div {
            color: #0f172a;
        }
        .stButton > button {
            min-height: 2.85rem;
            border-radius: 14px;
            border: 1px solid #cbd5e1;
            background: linear-gradient(180deg, #ffffff, #f8fafc);
            color: #0f172a;
            font-weight: 600;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
            transition: all 120ms ease;
        }
        .stButton > button:hover {
            border-color: #93c5fd;
            background: linear-gradient(180deg, #ffffff, #eff6ff);
            color: #1d4ed8;
        }
        .stButton > button:focus {
            box-shadow: 0 0 0 0.2rem rgba(59, 130, 246, 0.18);
            border-color: #60a5fa;
        }
        div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stMetric"]) {
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(248, 250, 252, 0.96));
            border: 1px solid #dbe4f0;
            border-radius: 18px;
            padding: 0.72rem 0.9rem;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
        }
        div[data-testid="stMetric"] {
            background: transparent;
            border: none;
            padding: 0.2rem 0.1rem;
            box-shadow: none;
        }
        div[data-testid="stMetricLabel"] label {
            color: #475569 !important;
            font-size: 0.82rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.02em;
            text-transform: uppercase;
        }
        div[data-testid="stMetricValue"] {
            color: #0f172a !important;
            font-size: 2rem !important;
            font-weight: 700 !important;
            line-height: 1.1;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid #dbe4f0;
            border-radius: 16px;
            overflow: hidden;
        }
        .hero-panel, .status-banner, .content-panel, .placeholder-panel {
            border: 1px solid #dbe4f0;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.94);
            box-shadow: 0 10px 26px rgba(15, 23, 42, 0.055);
        }
        .hero-panel {
            display: flex;
            justify-content: space-between;
            gap: 1.5rem;
            padding: 1.2rem 1.3rem;
            margin-bottom: 0.75rem;
        }
        .eyebrow {
            margin: 0 0 0.4rem 0;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #1d4ed8;
        }
        .hero-panel h1 {
            margin: 0;
            font-size: 2rem;
            line-height: 1.05;
        }
        .hero-copy {
            margin: 0.45rem 0 0 0;
            max-width: 48rem;
            color: #334155;
            font-size: 0.98rem;
        }
        .hero-meta {
            min-width: 220px;
            padding: 0.95rem 1rem;
            border-radius: 16px;
            background: linear-gradient(180deg, #eff6ff, #f8fafc);
            border: 1px solid #bfdbfe;
        }
        .hero-meta-label {
            color: #475569;
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .hero-meta-value {
            margin-top: 0.25rem;
            font-size: 1.25rem;
            font-weight: 700;
        }
        .hero-meta-subtle {
            margin-top: 0.5rem;
            color: #475569;
            font-size: 0.88rem;
            line-height: 1.4;
        }
        .status-banner {
            padding: 0.85rem 1rem;
            margin-bottom: 0.75rem;
        }
        .status-live {
            border-left: 5px solid #059669;
        }
        .status-sample {
            border-left: 5px solid #d97706;
            background: #fffaf0;
        }
        .status-title {
            font-size: 0.95rem;
            font-weight: 700;
        }
        .status-body {
            margin-top: 0.18rem;
            color: #334155;
            font-size: 0.94rem;
            line-height: 1.4;
        }
        .section-heading {
            margin: 0.05rem 0 0.45rem 0;
        }
        .section-heading h3 {
            margin: 0;
            font-size: 1.08rem;
        }
        .section-heading p {
            margin: 0.2rem 0 0 0;
            color: #475569;
            font-size: 0.9rem;
            line-height: 1.4;
        }
        .content-panel {
            padding: 0.9rem 0.95rem 0.55rem 0.95rem;
            margin-bottom: 0.7rem;
        }
        .placeholder-panel {
            padding: 1rem 1.05rem;
            background: linear-gradient(180deg, #fff, #f8fafc);
        }
        .placeholder-kicker {
            color: #1d4ed8;
            font-size: 0.74rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .placeholder-title {
            margin-top: 0.15rem;
            font-size: 0.98rem;
            font-weight: 700;
        }
        .placeholder-copy {
            margin-top: 0.3rem;
            color: #475569;
            line-height: 1.5;
        }
        .stCaption {
            color: #64748b !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    data = load_dashboard_data()
    summary = data["summary"]
    timeseries_df = prepare_timeseries_frame(data["timeseries"].get("points", []))
    sources_df = prepare_breakdown_frame(data["sources"].get("sources", []), "source")
    topics_df = prepare_breakdown_frame(data["topics"].get("topics", []), "topic")
    aspects_df = prepare_breakdown_frame(data.get("aspects", {}).get("aspects", []), "aspect")

    render_intro(summary)
    st.caption(f"Backend endpoint: `{API_BASE_URL}`")

    status_col, refresh_col = st.columns([5, 1])
    with status_col:
        render_status_banner(data)
    with refresh_col:
        if st.button("Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    render_metric_cards(summary)

    left_col, right_col = st.columns([1.6, 1])
    with left_col:
        st.markdown('<div class="content-panel">', unsafe_allow_html=True)
        render_section_header(
            "Sentiment Over Time",
            "Daily sentiment is shown alongside a rolling average to separate short-term swings from the broader trend.",
        )
        render_sentiment_chart(timeseries_df)
        st.markdown("</div>", unsafe_allow_html=True)
    with right_col:
        st.markdown('<div class="content-panel">', unsafe_allow_html=True)
        render_section_header(
            "Sentiment Mix",
            "A quick read on how much of the current coverage is positive, neutral, or negative.",
        )
        render_coverage(summary)
        st.markdown("</div>", unsafe_allow_html=True)

    source_col, topic_col = st.columns(2)
    with source_col:
        st.markdown('<div class="content-panel">', unsafe_allow_html=True)
        render_section_header(
            "Source Breakdown",
            "Compare which publishers are contributing the most coverage and how sentiment differs across outlets.",
        )
        render_breakdown_chart(sources_df, "Source Breakdown", "#2563eb")
        st.markdown("</div>", unsafe_allow_html=True)
    with topic_col:
        st.markdown('<div class="content-panel">', unsafe_allow_html=True)
        render_section_header(
            "Topic Breakdown",
            "Topics describe what coverage is about. They remain separate from aspect-level sentiment analysis.",
        )
        render_breakdown_chart(topics_df, "Topic Breakdown", "#059669")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="content-panel">', unsafe_allow_html=True)
    render_section_header(
        "Aspect Breakdown",
        "Aspects capture which product or service dimensions are being discussed. They remain separate from broader coverage topics.",
    )
    render_aspect_breakdown(aspects_df, data["mode"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="content-panel">', unsafe_allow_html=True)
    render_section_header(
        "Detected Changepoints",
        "Potential shifts in the sentiment trend that may warrant a closer look in the underlying coverage.",
    )
    render_changepoints(data["changepoints"].get("changepoints", []))
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="content-panel">', unsafe_allow_html=True)
    render_section_header(
        "Recent Processed Documents",
        "Use this section to review the latest items entering the pipeline when document-level rows are available.",
    )
    render_recent_documents(data.get("recent_documents", []), data["mode"])
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
