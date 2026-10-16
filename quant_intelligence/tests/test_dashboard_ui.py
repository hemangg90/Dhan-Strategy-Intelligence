from quant_intelligence.ui.dashboard import _dd_text, _exp_cell


def test_expectancy_bar_shows_sign_and_clips():
    pos, neg = _exp_cell(0.25), _exp_cell(-0.4)
    assert "+0.25R" in pos and "var(--qi-good)" in pos
    assert "-0.40R" in neg and "var(--qi-critical)" in neg
    assert "width:50.0%" in _exp_cell(7.0)  # clipped at +1R


def test_missing_values_render_as_na_not_zero():
    assert "n/a" in _exp_cell(None)
    assert _dd_text(None) == "n/a" and _dd_text(-3.04) == "-3.0R"
