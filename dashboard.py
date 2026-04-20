from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import altair as alt
import pandas as pd
import requests
import streamlit as st


API_BASE_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT_SECONDS = 2


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
        changepoints = safe_get("/analytics/changepoints")

        return {
            "mode": "live",
            "health": health,
            "summary": summary,
            "timeseries": timeseries,
            "sources": sources,
            "topics": topics,
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


def render_metric_cards(summary: dict[str, Any]) -> None:
    card_1, card_2, card_3, card_4 = st.columns(4)
    card_1.metric("Documents", f"{summary.get('document_count', 0):,}")
    card_2.metric("Average Sentiment", f"{summary.get('average_sentiment', 0.0):.2f}")
    card_3.metric("Sources", f"{summary.get('source_count', 0):,}")
    card_4.metric("Topics", f"{summary.get('topic_count', 0):,}")


def render_sentiment_chart(timeseries_df: pd.DataFrame) -> None:
    st.subheader("Sentiment Over Time")
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
            y=alt.Y("sentiment:Q", title="Sentiment", scale=alt.Scale(domain=[-1, 1])),
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
        .properties(height=320)
    )

    st.altair_chart(line, use_container_width=True)


def render_breakdown_chart(frame: pd.DataFrame, title: str, color: str) -> None:
    st.subheader(title)
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
        .properties(height=300)
    )

    st.altair_chart(chart, use_container_width=True)


def render_changepoints(changepoints: list[dict[str, Any]]) -> None:
    st.subheader("Changepoints")
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


def render_recent_documents(recent_documents: list[dict[str, Any]], mode: str) -> None:
    st.subheader("Recent Processed Documents")
    if recent_documents:
        frame = pd.DataFrame(recent_documents)
        if "created_utc" in frame.columns:
            frame["created_utc"] = pd.to_datetime(frame["created_utc"], utc=True).dt.strftime("%Y-%m-%d %H:%M UTC")
        st.dataframe(frame, use_container_width=True, hide_index=True)
        return

    if mode == "live":
        st.info("The current backend API does not expose document-level rows, so recent processed documents cannot be shown from live data yet.")
    else:
        st.info("Recent processed documents are unavailable.")


def main() -> None:
    st.set_page_config(
        page_title="Brand Perception Dashboard",
        page_icon=":bar_chart:",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(59, 130, 246, 0.10), transparent 30%),
                radial-gradient(circle at top right, rgba(16, 185, 129, 0.10), transparent 28%),
                linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
        }
        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid rgba(148, 163, 184, 0.22);
            padding: 0.9rem 1rem;
            border-radius: 16px;
            box-shadow: 0 10px 35px rgba(15, 23, 42, 0.07);
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

    st.title("Brand Perception Dashboard")
    st.caption(f"Backend: `{API_BASE_URL}`")

    status_col, refresh_col = st.columns([5, 1])
    with status_col:
        if data["mode"] == "live":
            st.success(
                f"Live API connected. Backend: {data['health'].get('storage_backend', 'unknown')} | Documents: {data['health'].get('processed_documents', 0):,}"
            )
        else:
            st.warning("API unavailable. Rendering sample data so the dashboard stays usable.")
            if data.get("error"):
                st.caption(f"Connection error: {data['error']}")
    with refresh_col:
        if st.button("Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    render_metric_cards(summary)

    left_col, right_col = st.columns([1.6, 1])
    with left_col:
        render_sentiment_chart(timeseries_df)
    with right_col:
        st.subheader("Coverage")
        coverage = pd.DataFrame(
            [
                {"Metric": "Positive Share", "Value": summary.get("positive_share", 0.0)},
                {"Metric": "Neutral Share", "Value": summary.get("neutral_share", 0.0)},
                {"Metric": "Negative Share", "Value": summary.get("negative_share", 0.0)},
            ]
        )
        coverage["Value"] = coverage["Value"].map(lambda value: f"{value:.0%}")
        st.dataframe(coverage, use_container_width=True, hide_index=True)
        st.caption(
            f"Window: {summary.get('first_document_at', 'n/a')} to {summary.get('last_document_at', 'n/a')}"
        )

    source_col, topic_col = st.columns(2)
    with source_col:
        render_breakdown_chart(sources_df, "Source Breakdown", "#2563eb")
    with topic_col:
        render_breakdown_chart(topics_df, "Topic Breakdown", "#059669")

    render_changepoints(data["changepoints"].get("changepoints", []))
    render_recent_documents(data.get("recent_documents", []), data["mode"])


if __name__ == "__main__":
    main()
