"""TimesFM 2.5 vs TimesFM-3 — паритет на реальных MOEX-данных (ЭКСПЕРИМЕНТ).

Методология 1-в-1 как в рабочем timesfm_kronos_compare.py:
  fetch_moex_candles(ticker, days=60) -> hourly close -> values_np float32
  -> forecast(horizon) -> preds[-1] vs last_price -> compute_signal (±2%)
Только вместо Kronos вторым плечом идёт TimesFM-3 в univariate-режиме.

Переиспользует fetch/compute/compare/load из референса (import, не копипаст).
Лог: logs/parity_YYYYMMDD_HHMM.log
"""
import os
import sys
import time
import datetime

# Force CPU: Blackwell sm_120 не поддерживается сборками torch
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
from timesfm_kronos_compare import (  # noqa: E402
    fetch_moex_candles,
    compute_signal,
    compare_signals,
    load_timesfm,
    timesfm_forecast,
)


def load_timesfm3():
    """Load TimesFM-3 evaluator (CPU)."""
    from timesfm3 import TimesFM3Evaluator, ModelConfig

    print("[TimesFM-3] Loading model...")
    config = ModelConfig(
        checkpoint_path="google/timesfm-3.0-pytorch",
        per_core_batch_size=8,
        device="cpu",
    )
    forecaster = TimesFM3Evaluator(config)
    print("[TimesFM-3] Ready (device=cpu)!")
    return forecaster


def timesfm3_forecast_univariate(forecaster, values, horizon):
    """Univariate forecast via TimesFM-3. Returns array of close prices."""
    outputs = list(
        forecaster.predict_batch(
            [values],
            horizon=horizon,
            return_quantiles=False,
            use_symmetric_averaging=False,
        )
    )
    return np.asarray(outputs[0].forecast, dtype=np.float64)


def main():
    TICKERS = ["SBERP", "GAZP", "LKOH"]
    HORIZONS = [30, 60, 90]

    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join(
        "logs", datetime.datetime.now().strftime("parity_%Y%m%d_%H%M.log")
    )

    def emit(msg=""):
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(msg + "\n")

    emit("=" * 80)
    emit("  TIMESFM 2.5 vs TIMESFM-3 (univariate) — PARITY")
    emit("  " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M MSK"))
    emit("=" * 80)

    t0 = time.time()
    emit("\n[INIT] Loading TimesFM 2.5...")
    m25 = load_timesfm()
    t25_load = time.time() - t0

    t0 = time.time()
    emit("\n[INIT] Loading TimesFM-3...")
    m3 = load_timesfm3()
    t3_load = time.time() - t0
    emit(f"\n[INIT] load times: 2.5={t25_load:.1f}s  3.0={t3_load:.1f}s")

    all_results = []
    for ticker in TICKERS:
        emit(f"\n{'=' * 80}\n[{ticker}] Fetching MOEX ISS data (60 days)...")
        df = fetch_moex_candles(ticker, days=60)
        if df is None or len(df) < 100:
            emit(f"  ERROR: not enough data for {ticker}")
            continue
        values_np = df["close"].values.astype(np.float32)
        last_price = float(df["close"].iloc[-1])
        emit(f"  Got {len(df)} hourly candles, last close: {last_price:.2f} RUB")

        for horizon in HORIZONS:
            emit(f"\n  --- Horizon: {horizon} ---")
            try:
                t0 = time.time()
                p25 = timesfm_forecast(m25, values_np, horizon)[-1]
                t25 = time.time() - t0
                s25, c25 = compute_signal(p25, last_price)
                emit(f"  [2.5] {horizon}: {p25:.2f} ({c25:+.2f}%) -> [{s25}] ({t25:.1f}s)")
            except Exception as e:
                s25, c25, p25, t25 = "ERROR", 0, last_price, 0
                emit(f"  [2.5] ERROR: {e}")
            try:
                t0 = time.time()
                p3 = timesfm3_forecast_univariate(m3, values_np, horizon)[-1]
                t3 = time.time() - t0
                s3, c3 = compute_signal(p3, last_price)
                emit(f"  [3.0] {horizon}: {p3:.2f} ({c3:+.2f}%) -> [{s3}] ({t3:.1f}s)")
            except Exception as e:
                s3, c3, p3, t3 = "ERROR", 0, last_price, 0
                emit(f"  [3.0] ERROR: {e}")

            agree, conf = compare_signals(s25, c25, s3, c3)
            emit(f"  => {agree} (уверенность: {conf})")
            all_results.append(
                dict(ticker=ticker, horizon=horizon, last=last_price,
                     p25=p25, s25=s25, c25=c25, t25=t25,
                     p3=p3, s3=s3, c3=c3, t3=t3,
                     agree=agree, conf=conf)
            )

    emit(f"\n{'=' * 80}\nPARITY TABLE (2.5 vs 3.0 univariate)")
    emit("=" * 80)
    for h in HORIZONS:
        emit(f"\n--- Горизонт {h} ---")
        emit(f"{'Тикер':<7} {'Цена':>9} | {'2.5 сигнал':<9} {'изм%':>7} {'t,c':>5} | "
             f"{'3.0 сигнал':<9} {'изм%':>7} {'t,c':>5} | Итог")
        for r in [x for x in all_results if x["horizon"] == h]:
            emit(f"{r['ticker']:<7} {r['last']:>9.2f} | {r['s25']:<9} "
                 f"{r['c25']:>+7.2f} {r['t25']:>5.1f} | {r['s3']:<9} "
                 f"{r['c3']:>+7.2f} {r['t3']:>5.1f} | {r['agree']}")
    emit(f"\nLog saved: {log_path}\nDONE")


if __name__ == "__main__":
    main()
