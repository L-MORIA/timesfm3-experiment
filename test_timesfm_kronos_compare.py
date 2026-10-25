"""Tests for timesfm_kronos_compare — горизонты, fetch и сигналы.

Запуск (системным Python, где установлен pytest):
    python -m pytest test_timesfm_kronos_compare.py -q
"""

import datetime

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


# ── Тест 5: compute_signal — порог ±2% и знак изменения ────────

def test_compute_signal_thresholds():
    last = 100.0

    # Ровно на пороге +2% → ПОКУПКА, чуть ниже — ДЕРЖАТЬ.
    assert m.compute_signal(102.0, last)[0] == "ПОКУПКА"
    assert m.compute_signal(101.9, last)[0] == "ДЕРЖАТЬ"

    # Ровно на пороге -2% → ПРОДАЖА, чуть выше — ДЕРЖАТЬ.
    assert m.compute_signal(98.0, last)[0] == "ПРОДАЖА"
    assert m.compute_signal(98.1, last)[0] == "ДЕРЖАТЬ"

    # В коридоре — ДЕРЖАТЬ.
    assert m.compute_signal(100.0, last)[0] == "ДЕРЖАТЬ"


def test_compute_signal_returns_change_percent():
    signal, chg = m.compute_signal(110.0, 100.0)

    assert signal == "ПОКУПКА"
    assert chg == 10.0


# ── Тест 6: compare_signals — три ветки согласия ──────────────

def test_compare_signals_agreement():
    # Одинаковые сигналы → СОВПАДЕНИЕ / ВЫСОКАЯ (независимо от chg).
    assert m.compare_signals("ПОКУПКА", 5.0, "ПОКУПКА", 0.1) == (
        "СОВПАДЕНИЕ", "ВЫСОКАЯ",
    )


def test_compare_signals_neutral_when_both_small():
    # Разные сигналы, но оба изменения < 0.5% → НЕЙТРАЛЬНО / СРЕДНЯЯ.
    assert m.compare_signals("ПОКУПКА", 0.4, "ДЕРЖАТЬ", -0.3) == (
        "НЕЙТРАЛЬНО", "СРЕДНЯЯ",
    )


def test_compare_signals_conflict():
    # Разные сигналы и заметное расхождение → РАСХОЖДЕНИЕ / НИЗКАЯ.
    assert m.compare_signals("ПОКУПКА", 3.0, "ПРОДАЖА", -2.5) == (
        "РАСХОЖДЕНИЕ", "НИЗКАЯ",
    )

    # Один chg большой, другой маленький — всё равно РАСХОЖДЕНИЕ
    # (условие НЕЙТРАЛЬНО требует оба < 0.5%).
    assert m.compare_signals("ПОКУПКА", 3.0, "ДЕРЖАТЬ", 0.1)[0] == "РАСХОЖДЕНИЕ"


# ── Тест 7: ERROR-сигналы не образуют «СОВПАДЕНИЕ» ─────────────

def test_compare_signals_error_is_not_agreement():
    # Отказ обеих моделей — не «согласие»: раньше возвращалось
    # ("СОВПАДЕНИЕ", "ВЫСОКАЯ"), что ложно завышало уверенность.
    assert m.compare_signals("ERROR", 0, "ERROR", 0) == ("ОШИБКА", "НЕТ")
    assert m.compare_signals("ERROR", 0, "ПОКУПКА", 5.0) == ("ОШИБКА", "НЕТ")
    assert m.compare_signals("ПОКУПКА", 5.0, "ERROR", 0) == ("ОШИБКА", "НЕТ")


# ── Тест 8: обе модели упали → в выводе нет ложного «СОВПАДЕНИЕ» ─

def test_main_error_path_does_not_report_agreement(monkeypatch, capsys):
    _run_main_with_mocks(
        monkeypatch,
        kronos_predict=lambda predictor, df_full, pred_len: None,
        timesfm_forecast=lambda model, values, horizon: (_ for _ in ()).throw(
            RuntimeError("model down")),
    )
    out = capsys.readouterr().out

    assert "ОШИБКА" in out
    # Старое поведение печатало "СОВПАДЕНИЕ / ВЫСОКАЯ" на ERROR-сигналах.
    assert "СОВПАДЕНИЕ" not in out
    assert "ВЫСОКАЯ" not in out


# ── Тест 9: main() пропускает тикер с нулевой последней ценой ──

def test_main_skips_ticker_with_zero_last_price(monkeypatch):
    kronos_horizons = []

    def fake_kronos_predict(predictor, df_full, pred_len):
        kronos_horizons.append(pred_len)
        return np.full(pred_len, 105.0)

    _run_main_with_mocks(
        monkeypatch,
        fetch=lambda ticker, days=60: (
            _fake_df(300, np.zeros(300))
            if ticker in ("SBERP", "GAZP")
            else _fake_df(300, np.linspace(100.0, 110.0, 300))
        ),
        kronos_predict=fake_kronos_predict,
    )

    # SBERP/GAZP с нулевой ценой пропущены -> остались 3 горизонта LKOH.
    assert kronos_horizons == [30, 60, 90]


# ── Тест 10: смешанные дни — сбой/пусто/данные → сортировка ────

def test_fetch_mixed_days(monkeypatch):
    today = datetime.date.today()
    day_strs = [(today - datetime.timedelta(days=i)).strftime("%Y-%m-%d")
                for i in range(3)]

    def fake_get(url, params=None, timeout=None):
        frm = params["from"]
        if frm == day_strs[0]:
            return _FakeResponse(500)  # ретраи исчерпаны — день пропущен
        if frm == day_strs[1]:
            return _FakeResponse(200, _candles_payload([]))  # без торгов
        # последний день: две свечи, намеренно в обратном порядке
        return _FakeResponse(200, _candles_payload([
            [6, 7, 8, 9, 5, "2024-01-02 12:00:00"],
            [1, 2, 3, 4, 5, "2024-01-02 10:00:00"],
        ]))

    monkeypatch.setattr(m.requests, "get", fake_get)
    monkeypatch.setattr(m.time, "sleep", lambda *a, **k: None)

    df = m.fetch_moex_candles("SBERP", days=3, retries=2)

    assert df is not None
    assert len(df) == 2
    assert df["begin"].is_monotonic_increasing


# ── Тест 11: дубликаты begin между днями схлопываются ──────────

def test_fetch_dedups_duplicate_begins(monkeypatch):
    payload = _candles_payload([[1, 2, 3, 4, 5, "2024-01-01 10:00:00"]])
    monkeypatch.setattr(
        m.requests, "get",
        lambda url, params=None, timeout=None: _FakeResponse(200, payload),
    )
    monkeypatch.setattr(m.time, "sleep", lambda *a, **k: None)

    df = m.fetch_moex_candles("SBERP", days=2)

    assert df is not None
    assert len(df) == 1


# ── Хелпер: прогон main() с настраиваемыми моками ──────────────

def _fake_df(n, close):
    return pd.DataFrame({
        "begin": pd.date_range("2024-01-01", periods=n, freq="h"),
        "close": close,
    })


def _run_main_with_mocks(monkeypatch, fetch=None, kronos_predict=None,
                         timesfm_forecast=None):
    good_df = _fake_df(300, np.linspace(100.0, 110.0, 300))

    monkeypatch.setattr(m, "load_kronos", lambda: object())
    monkeypatch.setattr(m, "load_timesfm", lambda: object())
    monkeypatch.setattr(
        m, "fetch_moex_candles",
        fetch if fetch is not None else (lambda ticker, days=60: good_df.copy()),
    )
    monkeypatch.setattr(
        m, "kronos_predict",
        kronos_predict if kronos_predict is not None
        else (lambda predictor, df_full, pred_len: np.full(pred_len, 105.0)),
    )
    monkeypatch.setattr(
        m, "timesfm_forecast",
        timesfm_forecast if timesfm_forecast is not None
        else (lambda model, values, horizon: np.full(horizon, 105.0)),
    )
    monkeypatch.setattr(m.plt, "savefig", lambda *a, **k: None)

    m.main()
