"""Tests for timesfm_kronos_compare — горизонты и устойчивость fetch.

Запуск (системным Python, где установлен pytest):
    python -m pytest test_timesfm_kronos_compare.py -q
"""

import numpy as np
import pandas as pd

import timesfm_kronos_compare as m


# ── Общие фикстуры/хелперы ─────────────────────────────────────

CANDLE_COLUMNS = ["OPEN", "CLOSE", "HIGH", "LOW", "VOLUME", "BEGIN"]


def _candles_payload(rows):
    return {"candles": {"columns": CANDLE_COLUMNS, "data": rows}}


class _FakeResponse:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}

    def json(self):
        return self._payload


# ── Хелпер: прогон main() с моками на границах внешних вызовов ─

def _capture_main_horizons(monkeypatch):
    """Прогнать main() и вернуть (kronos_horizons, timesfm_horizons).

    Моки стоят на границах внешних вызовов (load_*, fetch, predict-функции,
    plt.savefig), а не внутри модуля, поэтому тест ловит именно то значение
    горизонта, которое реально уходит в каждую модель.
    """
    n = 300
    fake_df = pd.DataFrame({
        "begin": pd.date_range("2024-01-01", periods=n, freq="h"),
        "close": np.linspace(100.0, 110.0, n),
    })

    kronos_horizons = []
    timesfm_horizons = []

    monkeypatch.setattr(m, "load_kronos", lambda: object())
    monkeypatch.setattr(m, "load_timesfm", lambda: object())
    monkeypatch.setattr(
        m, "fetch_moex_candles",
        lambda ticker, days=60: fake_df.copy(),
    )

    def fake_kronos_predict(predictor, df_full, pred_len):
        kronos_horizons.append(pred_len)
        return np.full(pred_len, 105.0)

    def fake_timesfm_forecast(model, values, horizon):
        timesfm_horizons.append(horizon)
        return np.full(horizon, 105.0)

    monkeypatch.setattr(m, "kronos_predict", fake_kronos_predict)
    monkeypatch.setattr(m, "timesfm_forecast", fake_timesfm_forecast)
    monkeypatch.setattr(m.plt, "savefig", lambda *a, **k: None)

    m.main()
    return kronos_horizons, timesfm_horizons


# ── Тест 1: TimesFM и Kronos получают одинаковый горизонт ──────

def test_horizons_match_between_models(monkeypatch):
    kronos_horizons, timesfm_horizons = _capture_main_horizons(monkeypatch)

    # len(TICKERS) == 3 (SBERP, GAZP, LKOH) — тот же список, что в main().
    expected = [30, 60, 90] * 3

    assert timesfm_horizons == expected
    assert kronos_horizons == expected
    assert kronos_horizons == timesfm_horizons


# ── Тест 1b: явная защита от регрессии `horizon * 24` для Kronos ─

def test_kronos_horizon_is_not_multiplied_by_24(monkeypatch):
    kronos_horizons, timesfm_horizons = _capture_main_horizons(monkeypatch)

    # Старая логика `min(horizon * 24, 500)` отдавала Kronos 500/500/500
    # при TimesFM 30/60/90. Если её вернут, этот тест обязан упасть.
    assert kronos_horizons == [30, 60, 90] * 3
    assert max(kronos_horizons) == 90
    assert all(k == t for k, t in zip(kronos_horizons, timesfm_horizons))


# ── Тест 2: ретрай на 429 ──────────────────────────────────────

def test_fetch_retries_on_429(monkeypatch):
    calls = {"n": 0}
    payload = _candles_payload([[1, 2, 3, 4, 5, "2024-01-01 10:00:00"]])

    def fake_get(url, params=None, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeResponse(429)
        return _FakeResponse(200, payload)

    monkeypatch.setattr(m.requests, "get", fake_get)
    monkeypatch.setattr(m.time, "sleep", lambda *a, **k: None)

    df = m.fetch_moex_candles("SBERP", days=1)

    assert calls["n"] >= 2
    assert df is not None
    assert len(df) == 1


# ── Тест 3: ретраи исчерпаны → None ровно за `retries` попыток ─

def test_fetch_retries_exhausted_returns_none(monkeypatch):
    calls = {"n": 0}

    def fake_get(url, params=None, timeout=None):
        calls["n"] += 1
        return _FakeResponse(500)

    monkeypatch.setattr(m.requests, "get", fake_get)
    monkeypatch.setattr(m.time, "sleep", lambda *a, **k: None)

    df = m.fetch_moex_candles("SBERP", days=1, retries=3)

    assert calls["n"] == 3
    assert df is None


# ── Тест 4: пустой candles.data → None без исключения ──────────

def test_fetch_skips_empty_data(monkeypatch):
    monkeypatch.setattr(
        m.requests, "get",
        lambda url, params=None, timeout=None: _FakeResponse(200, _candles_payload([])),
    )
    monkeypatch.setattr(m.time, "sleep", lambda *a, **k: None)

    assert m.fetch_moex_candles("SBERP", days=1) is None
