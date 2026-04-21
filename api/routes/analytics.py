from __future__ import annotations

import os

from fastapi import APIRouter

from analytics.service import AnalyticsService


router = APIRouter()


def get_analytics_service() -> AnalyticsService:
    return AnalyticsService(
        backend=os.environ.get("BRAND_PERCEPTION_STORAGE_BACKEND", "duckdb"),
        path=os.environ.get("BRAND_PERCEPTION_STORAGE_PATH"),
        table_name=os.environ.get("BRAND_PERCEPTION_PROCESSED_TABLE", "processed_documents"),
        use_sample_data=os.environ.get("BRAND_PERCEPTION_USE_SAMPLE_DATA", "1") != "0",
        persist_sample=os.environ.get("BRAND_PERCEPTION_PERSIST_SAMPLE_DATA", "0") == "1",
    )


@router.get("/health")
def health() -> dict:
    service = get_analytics_service()
    metrics = service.summary_metrics()
    return {
        "status": "ok",
        "storage_backend": service.resolved_backend(),
        "processed_documents": metrics["document_count"],
    }


@router.get("/analytics/summary")
def analytics_summary() -> dict:
    service = get_analytics_service()
    return service.summary_metrics()


@router.get("/analytics/timeseries")
def analytics_timeseries(rolling_window_days: int = 7) -> dict:
    service = get_analytics_service()
    return {
        "rolling_window_days": rolling_window_days,
        "points": service.sentiment_over_time(rolling_window_days=rolling_window_days),
    }


@router.get("/analytics/sources")
def analytics_sources(limit: int = 10) -> dict:
    service = get_analytics_service()
    return {
        "limit": limit,
        "sources": service.source_breakdown(limit=limit),
    }


@router.get("/analytics/topics")
def analytics_topics(limit: int = 10) -> dict:
    service = get_analytics_service()
    return {
        "limit": limit,
        "topics": service.topic_breakdown(limit=limit),
    }


@router.get("/analytics/aspects")
def analytics_aspects(limit: int = 10) -> dict:
    service = get_analytics_service()
    return {
        "limit": limit,
        "aspects": service.aspect_breakdown(limit=limit),
    }


@router.get("/analytics/changepoints")
def analytics_changepoints(
    rolling_window_days: int = 7,
    penalty: float = 3.0,
) -> dict:
    service = get_analytics_service()
    return {
        "rolling_window_days": rolling_window_days,
        "penalty": penalty,
        "changepoints": service.changepoints(
            rolling_window_days=rolling_window_days,
            penalty=penalty,
        ),
    }
