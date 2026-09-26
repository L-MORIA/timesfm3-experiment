"""Backtest 2.5 vs 3.0-uni vs 3.0-multi на истории MOEX (ЭКСПЕРИМЕНТ).

Схема: 120 дней часовых свечей -> состыковка 3 тикеров по begin ->
срезы «как будто сейчас» (cutoffs) -> прогноз горизонтов -> сверка
с известной реальностью. Метрики: endpoint-ошибка (п.п.), MAPE пути,
hit-rate сигнала (то же правило ±2%). Единица горизонта = шаг свечи
(та же методология, что в прод-скрипте, где шаги названы «днями»).

Плечи: T25 (TimesFM 2.5), T3U (TimesFM-3 univariate), T3M (TimesFM-3
multivariate: корзина + объёмы как past-covariate, без утечек —
ковариаты тоже режутся по cutoff).

Лог: logs/backtest_YYYYMMDD_HHMM.log
"""
import os
import sys
import time
import datetime

# Force CPU: Blackwell sm_120 не поддерживается сборками torch
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from timesfm_kronos_compare import (  # noqa: E402
    fetch_moex_candles,
    compute_signal,
    load_timesfm,
    timesfm_forecast,
)
from timesfm3 import TimesFM3Evaluator, ModelConfig  # noqa: E402

TICKERS = ["SBERP", "GAZP", "LKOH"]
DAYS = 120
# (cutoff_back_steps, horizon): будущее cutoff+h обязано лежать в истории
PLAN = [(120, 30), (90, 30), (60, 30),
        (120, 60), (90, 60), (60, 60),
        (150, 90), (120, 90)]


def load_timesfm3():
    print("[TimesFM-3] Loading model...", flush=True)
    m = TimesFM3Evaluator(ModelConfig(
        checkpoint_path="google/timesfm-3.0-pytorch",
        per_core_batch_size=8,
        device="cpu",
    ))
    print("[TimesFM-3] Ready (device=cpu)!", flush=True)
    return m


def mape(pred, real):
    pred = np.asarray(pred, dtype=np.float64)
    real = np.asarray(real, dtype=np.float64)
    mask = real != 0
    return float(np.mean(np.abs((real[mask] - pred[mask]) / real[mask])) * 100)


def main():
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join(
        "logs", datetime.datetime.now().strftime("backtest_%Y%m%d_%H%M.log")
    )

    def emit(msg=""):
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(msg + "\n")

    emit("=" * 80)
    emit("  BACKTEST: 2.5 vs 3.0-uni vs 3.0-multi (MOEX history)")
    emit("  " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M MSK"))
    emit("=" * 80)

    frames = {}
    for t in TICKERS:
        emit(f"\n[{t}] Fetching {DAYS}d hourly...")
        df = fetch_moex_candles(t, days=DAYS)
        if df is None or len(df) < 300:
            emit(f"  ERROR: not enough data for {t}")
            return
        frames[t] = df[["begin", "close", "volume"]].rename(
            columns={"close": f"close_{t}", "volume": f"vol_{t}"})
        emit(f"  Got {len(df)} candles")
    joint = frames[TICKERS[0]]
    for t in TICKERS[1:]:
        joint = pd.merge(joint, frames[t], on="begin", how="inner")
    joint = joint.sort_values("begin").reset_index(drop=True)
    T = len(joint)
    emit(f"\n[JOIN] aligned rows: {T}")
    if T < 400:
        emit("  ERROR: join too short for backtest plan")
        return

    closes = {t: joint[f"close_{t}"].values.astype(np.float64) for t in TICKERS}
    vols = {t: joint[f"vol_{t}"].fillna(0).values.astype(np.float64)
            for t in TICKERS}

    t0 = time.time()
    emit("\n[INIT] Loading TimesFM 2.5...")
    m25 = load_timesfm()
    emit("\n[INIT] Loading TimesFM-3...")
    m3 = load_timesfm3()
    emit(f"[INIT] loads done in {time.time() - t0:.1f}s")

    recs = []  # список dict-записей: model, ticker, horizon, end_err, mape, hit
    for back, h in PLAN:
        c = T - back  # cutoff index: «сейчас»
        if c < 200 or c + h > T:
            continue
        cdate = joint["begin"].iloc[c]
        emit(f"\n--- cutoff T-{back} ({cdate}), horizon {h} ---")
        tgt = np.stack([closes[t][:c] for t in TICKERS]).astype(np.float32)
        vcov = np.stack([vols[t][:c] for t in TICKERS]).astype(np.float32)
        # multivariate: один вызов
        try:
            out_m = list(m3.predict_batch(
                contexts=[tgt], horizon=h,
                past_only_covariates=[vcov],
                return_quantiles=False,
                use_symmetric_averaging=False))[0]
            pm = np.asarray(out_m.forecast, dtype=np.float64)
        except Exception as e:
            emit(f"  [T3M] ERROR: {e}")
            pm = None
        for i, t in enumerate(TICKERS):
            ctx = tgt[i]
            last = float(ctx[-1])
            real_path = closes[t][c:c + h]
            real_end = float(real_path[-1])
            real_chg = (real_end - last) / last * 100
            s_real, _ = compute_signal(real_end, last)
            row = dict(ticker=t, horizon=h, back=back)
            # 2.5
            try:
                p = float(timesfm_forecast(m25, ctx.astype(np.float32), h)[-1])
                chg = (p - last) / last * 100
                s, _ = compute_signal(p, last)
                recs.append((dict(row, model="T25",
                                  end_err=abs(chg - real_chg),
                                  hit=(s == s_real))))
            except Exception as e:
                emit(f"  [{t}] T25 ERROR: {e}")
            # 3.0 uni
            try:
                pu = list(m3.predict_batch(
                    [ctx], horizon=h, return_quantiles=False,
                    use_symmetric_averaging=False))[0]
                fpu = np.asarray(pu.forecast, dtype=np.float64)
                p = float(fpu[-1])
                chg = (p - last) / last * 100
                s, _ = compute_signal(p, last)
                recs.append((dict(row, model="T3U",
                                  end_err=abs(chg - real_chg),
                                  mape=mape(fpu, real_path),
                                  hit=(s == s_real))))
            except Exception as e:
                emit(f"  [{t}] T3U ERROR: {e}")
            # 3.0 multi
            if pm is not None:
                fpm = pm[i]
                p = float(fpm[-1])
                chg = (p - last) / last * 100
                s, _ = compute_signal(p, last)
                recs.append((dict(row, model="T3M",
                                  end_err=abs(chg - real_chg),
                                  mape=mape(fpm, real_path),
                                  hit=(s == s_real))))
            emit(f"  {t}: last={last:.2f} real_end={real_end:.2f} "
                 f"({real_chg:+.2f}%) [{s_real}]")

    emit(f"\n{'=' * 80}\nBACKTEST RESULTS (mean over cutoffs x tickers)")
    emit("=" * 80)
    emit(f"{'Модель':<6} {'Горизонт':>8} | {'N':>3} | "
         f"{'Ср.ошибка endpoint, п.п.':>24} | {'Ср.MAPE пути,%':>14} | {'Hit-rate':>8}")
    for model in ["T25", "T3U", "T3M"]:
        for h in [30, 60, 90]:
            sel = [r for r in recs if r["model"] == model and r["horizon"] == h]
            if not sel:
                continue
            n = len(sel)
            ee = sum(r["end_err"] for r in sel) / n
            mm = [r["mape"] for r in sel if "mape" in r]
            ms = f"{sum(mm) / len(mm):.2f}" if mm else "n/a (2.5)"
            hr = sum(1 for r in sel if r["hit"]) / n * 100
            emit(f"{model:<6} {h:>8} | {n:>3} | {ee:>24.2f} | {ms:>14} | {hr:>7.1f}%")
    emit(f"\nLog saved: {log_path}\nDONE")


if __name__ == "__main__":
    main()
