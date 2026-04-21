from __future__ import annotations

from fastapi import FastAPI

from api.routes.analytics import router as analytics_router


def create_app() -> FastAPI:
    app = FastAPI(title="Brand Perception Analytics API", version="0.1.0")
    app.include_router(analytics_router)

    @app.get("/")
    def root() -> dict[str, object]:
        return {
            "service": "brand-perception-analytics",
            "endpoints": [
                "/health",
                "/analytics/summary",
            "/analytics/timeseries",
            "/analytics/sources",
            "/analytics/topics",
            "/analytics/aspects",
            "/analytics/changepoints",
        ],
    }

    return app


app = create_app()
