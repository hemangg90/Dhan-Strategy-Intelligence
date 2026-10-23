"""ANALOGUE_METHOD=context: the live pipeline gives every historical observation its market-structure state and compares it."""
import datetime as dt

from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.research.pipeline import run_pipeline


def _run(monkeypatch, tmp_path, method):
    from quant_intelligence.data import data_manager as dm_module
    from quant_intelligence.data.data_manager import DataManager
    from quant_intelligence.data_adapters.synthetic import SyntheticAdapter

    monkeypatch.setattr(dm_module, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(DataManager, "_fetch", lambda self, instrument, timeframe, start, end, prefer: (
        "test", SyntheticAdapter().get_ohlcv(instrument, timeframe, start, end)))
    original = SETTINGS.analogue_method
    object.__setattr__(SETTINGS, "analogue_method", method)
    try:
        end = dt.datetime(2024, 6, 28, 15, 30)
        return run_pipeline("NIFTY", "5min", end - dt.timedelta(days=60), end)
    finally:
        object.__setattr__(SETTINGS, "analogue_method", original)


def test_the_context_method_compares_structure_features_and_names_the_options_ones_it_could_not_use(monkeypatch, tmp_path):
    output = _run(monkeypatch, tmp_path, "context")
    used = [si.analogues for si in output.strategy_intel if len(si.analogues) and si.analogues.attrs.get("method") == "context"]
    assert used, "no strategy produced analogues under the context method"
    attrs = used[0].attrs
    assert "structure_trend_code" in attrs["features_used"]
    assert {"iv_rank", "pcr_oi"} <= set(attrs["features_unavailable"])  # no historical option chains: reported, not invented
    # every analogue carries the three separate times
    frame = used[0]
    assert {"feature_timestamp", "event_timestamp", "outcome_available_at"} <= set(frame.columns)
    assert (frame["outcome_available_at"] >= frame["event_timestamp"]).all()


def test_the_default_method_is_unchanged_and_uses_no_structure_features(monkeypatch, tmp_path):
    output = _run(monkeypatch, tmp_path, "legacy")
    for si in output.strategy_intel:
        if len(si.analogues):
            assert "structure_trend_code" not in (si.analogues.attrs.get("features_used") or [])
