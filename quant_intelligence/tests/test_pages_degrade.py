"""Failure injection for the UI: when the research pipeline is down, every page must show a message, not a traceback.

A page that needs the pipeline may stop with an error; a page that does not need it must keep working. None may raise.
"""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
PAGES = [ROOT / "app.py", ROOT / "home_page.py"] + sorted((ROOT / "pages").glob("*.py"))
NEEDS_PIPELINE = {"home_page.py", "00_Decision_Center.py", "01_Market_State.py", "02_Strategy_Intelligence.py",
                  "05_Regime_Analysis.py", "06_Historical_Analogues.py"}


@pytest.fixture()
def pipeline_outage(monkeypatch):
    from quant_intelligence.data.data_manager import DataUnavailableError
    from quant_intelligence.research import shared_cache

    def down(*args, **kwargs):
        raise DataUnavailableError("simulated outage: no market data source is reachable")

    monkeypatch.setattr(shared_cache.PIPELINE, "get", down)


@pytest.mark.parametrize("page", PAGES, ids=[p.name for p in PAGES])
def test_a_page_never_raises_when_the_pipeline_is_down(page, pipeline_outage):
    at = AppTest.from_file(str(page), default_timeout=90)
    at.run()
    assert not at.exception, f"{page.name} raised: {[str(e.value)[:200] for e in at.exception]}"
    if page.name in NEEDS_PIPELINE:
        shown = " ".join(str(e.value) for e in at.error)
        assert "simulated outage" in shown or "could not run" in shown.lower() or "not available" in shown.lower(), (
            f"{page.name} did not tell the user why there is no data: {shown[:200]!r}")


# ------------------------------------------------------------------ stored reports outlive code versions
def test_a_report_the_current_code_generates_is_complete_and_older_shapes_are_not():
    import datetime as dt

    from quant_intelligence.reports import eod

    built = eod.build_report(dt.date(2026, 1, 5), "ALL")
    assert eod.is_complete(built["content_json"])
    assert not eod.is_complete({"summary": {"today": {"trades": 3, "net": 1500.0, "win_rate": 66.0}, "cumulative": {"net": 4000.0}}})
    assert not eod.is_complete({"a": 1}) and not eod.is_complete(None) and not eod.is_complete("garbage")


def test_the_daily_report_page_rebuilds_instead_of_crashing_on_an_incomplete_stored_report():
    from quant_intelligence.reports import eod
    from quant_intelligence.utils.timeutil import now_ist

    minimal = {"content_json": {"summary": {"today": {"trades": 3, "net": 1500.0, "win_rate": 66.0}, "cumulative": {"net": 4000.0}}},
               "content_markdown": "# an older report"}
    eod.save_report(now_ist().date(), "ALL", minimal)  # the day the page opens on
    at = AppTest.from_file(str(ROOT / "pages" / "15_Daily_Report.py"), default_timeout=90)
    at.run()
    assert not at.exception, [str(e.value)[:200] for e in at.exception]
    assert any("older version" in str(w.value) for w in at.warning), "the user must be told why the report was rebuilt"
