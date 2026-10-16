"""Underlying stop/target -> option premium stop/target, against hand arithmetic."""
import pandas as pd
import pytest

from quant_intelligence.options import premium_model
from quant_intelligence.options.option_selector import OptionContract
from quant_intelligence.options.premium_model import PremiumSizingError, translate_setup
from quant_intelligence.strategies.base_strategy import Setup


def _contract(transaction="BUY", premium=100.0, delta=0.4):
    return OptionContract("1", "NIFTY-X", "NIFTY", 22500.0, "CE", "2026-10-29", 75, transaction, premium, delta, 15.0)


def _setup(entry=22000.0, stop=21950.0, target=22100.0):
    return Setup(pd.Timestamp("2026-10-16 10:00"), "LONG", entry, stop, target)


def test_a_bought_option_scales_stop_and_target_by_delta():
    p = translate_setup(_setup(), _contract("BUY", 100.0, 0.4))
    assert p.entry_price == 100.0
    assert p.stop_price == pytest.approx(100.0 - 0.4 * 50.0)  # underlying stop is 50 points away
    assert p.target_price == pytest.approx(100.0 + 0.4 * 100.0)  # underlying target is 100 points away
    assert p.delta_assumed is False


def test_a_negative_put_delta_is_used_by_magnitude():
    assert translate_setup(_setup(), _contract("BUY", 100.0, -0.4)).stop_price == pytest.approx(80.0)


def test_a_bought_stop_is_floored_at_one_tick_never_below_zero():
    p = translate_setup(_setup(stop=21000.0), _contract("BUY", 10.0, 0.5))  # 1000 points x 0.5 = 500 >> premium
    assert p.stop_price == pytest.approx(premium_model.MIN_PREMIUM_TICK)


def test_a_written_option_has_a_stop_above_entry_and_a_target_below():
    p = translate_setup(_setup(), _contract("SELL", 100.0, 0.4))
    assert p.stop_price == pytest.approx(120.0) and p.target_price == pytest.approx(60.0)
    assert p.stop_price > p.entry_price > p.target_price


def test_an_unknown_transaction_is_refused():
    with pytest.raises(PremiumSizingError):
        translate_setup(_setup(), _contract("HOLD"))


def test_a_missing_delta_is_assumed_flagged_and_logged_never_silent(monkeypatch):
    logged = []
    monkeypatch.setattr(premium_model, "log_event", lambda component, message, **kw: logged.append((kw.get("level"), message)))
    p = translate_setup(_setup(), _contract("BUY", 100.0, None))
    assert p.delta_assumed is True
    assert p.stop_price == pytest.approx(100.0 - premium_model.DEFAULT_DELTA_WHEN_MISSING * 50.0)
    assert logged and logged[0][0] == "WARNING" and "assuming" in logged[0][1]
