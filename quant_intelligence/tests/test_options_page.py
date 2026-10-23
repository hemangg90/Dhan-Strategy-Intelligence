"""The Options Intelligence page renders every tab from a seeded chain: chain, Greeks, IV, OI, liquidity, selector, risk.

Runs the real page headlessly with a chain placed in the session (no broker call) and the pipeline fed by synthetic candles.
"""
import datetime as dt
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from quant_intelligence.tests.test_option_pipeline import _chain

PAGE = Path(__file__).resolve().parent.parent / "pages" / "21_Options_Intelligence.py"
NOW = dt.datetime(2030, 1, 7, 10, 0)


@pytest.fixture
def seeded(monkeypatch, tmp_path):
    from quant_intelligence.brokers.dhan_api_client import DhanApiClient
    from quant_intelligence.data import data_manager as dm_module
    from quant_intelligence.data.data_manager import DataManager
    from quant_intelligence.data_adapters.synthetic import SyntheticAdapter
    from quant_intelligence.options import option_selector

    monkeypatch.setattr(DhanApiClient, "is_configured", lambda self: True)
    monkeypatch.setattr(dm_module, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(DataManager, "_fetch", lambda self, instrument, timeframe, start, end, prefer: (
        "test", SyntheticAdapter().get_ohlcv(instrument, timeframe, start, end)))
    monkeypatch.setattr("quant_intelligence.utils.timeutil.now_ist", lambda: NOW)
    monkeypatch.setattr("quant_intelligence.risk.risk_engine.now_ist", lambda: NOW)
    monkeypatch.setattr(option_selector, "get_underlying_info", lambda u: {"lot_size": 75})
    monkeypatch.setattr("quant_intelligence.data_adapters.dhan_instrument_master.list_fno_stock_symbols", lambda: [])
    chain = _chain(24000.0, NOW)
    return chain


def test_every_tab_renders_without_an_exception(seeded):
    at = AppTest.from_file(str(PAGE), default_timeout=240)
    at.session_state["option_chain"] = seeded
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert len(at.tabs) == 7 and len(at.metric) >= 10
    labels = {m.label for m in at.metric}
    assert {"Spot", "ATM strike", "ATM IV", "IV rank / percentile", "PCR (OI)", "Trading days left"} <= labels
    text = " ".join(str(m.value) for m in at.markdown) + " ".join(c.value for c in at.caption)
    assert "Option data" in text and "MEASURED" in text and "CALCULATED" in text
    iv = next(m for m in at.metric if m.label == "IV rank / percentile")
    assert iv.value == "unavailable"  # no recorded history: said so, not invented


def test_an_unusable_chain_is_refused_with_the_reason(seeded):
    stale = _chain(24000.0, NOW - dt.timedelta(minutes=30))
    at = AppTest.from_file(str(PAGE), default_timeout=240)
    at.session_state["option_chain"] = stale
    at.run()
    assert not at.exception
    assert any("old" in str(m.value) for m in at.markdown) and len(at.tabs) == 0
