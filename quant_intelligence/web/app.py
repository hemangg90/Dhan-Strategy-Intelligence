"""Website for Nifty Intelligence: a read-only JSON API plus a single-page app.

Run from the project root:
    python -m uvicorn quant_intelligence.web.app:app --port 8000

There are deliberately no order endpoints: the website can only read research, risk and backtest results.
"""
from __future__ import annotations

from pathlib import Path

import plotly
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from quant_intelligence.database.db import init_db
from quant_intelligence.web import service

STATIC = Path(__file__).resolve().parent / "static"
# Plotly ships its browser bundle inside the python package, so the site works with no internet access.
PLOTLY_JS = Path(plotly.__file__).resolve().parent / "package_data" / "plotly.min.js"

app = FastAPI(title="Nifty Intelligence", docs_url=None, redoc_url=None)
init_db()


@app.on_event("startup")
def _warm() -> None:
    service.PIPELINE.warm()


def _output(instrument: str, timeframe: str, refresh: bool):
    try:
        return service.PIPELINE.get(instrument, timeframe, force=refresh)
    except Exception as exc:  # surfaced to the UI as a readable message, never a stack trace
        raise HTTPException(status_code=503, detail=f"Pipeline could not run: {exc}")


@app.get("/api/health")
def health():
    return {"ok": True, "mode": service.mode_info()}


@app.get("/api/overview")
def overview(instrument: str = "NIFTY", timeframe: str = "5min", refresh: bool = False):
    return JSONResponse(service.overview(_output(instrument, timeframe, refresh)))


@app.get("/api/strategies/{name}")
def strategy(name: str, instrument: str = "NIFTY", timeframe: str = "5min"):
    detail = service.strategy_detail(_output(instrument, timeframe, False), name)
    if detail is None:
        raise HTTPException(status_code=404, detail="Unknown strategy")
    return JSONResponse(detail)


@app.get("/api/backtests")
def backtests(limit: int = Query(200, ge=1, le=500)):
    return JSONResponse(service.backtest_runs(limit))


@app.get("/api/backtests/{run_id}")
def backtest(run_id: str):
    detail = service.backtest_detail(run_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Unknown run")
    return JSONResponse(detail)


@app.get("/api/risk")
def risk():
    return JSONResponse(service.risk_overview())


@app.get("/vendor/plotly.min.js")
def plotly_js():
    return FileResponse(PLOTLY_JS, media_type="application/javascript")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
