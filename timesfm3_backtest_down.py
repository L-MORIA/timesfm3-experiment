"""Backtest на ПАДАЮЩЕМ окне: 2.5 vs 3.0-uni vs 3.0-multi (ЭКСПЕРИМЕНТ).

Фаза 1: 220 дней часовых свечей -> стыковка -> скан cutoffs, выбор
якоря с самым отрицательным средним 60-шаговым форвард-доходом.
Фаза 2: те же три плеча и метрики, что в timesfm3_backtest.py
(endpoint-ошибка, MAPE пути, hit-rate сигнала ±2%).

Лог: logs/backtest_down_YYYYMMDD_HHMM.log
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
from timesfm3_backtest import load_timesfm3, mape  # noqa: E402

TICKERS = ["SBERP", "GAZP", "LKOH"]
DAYS = 220
H_SCAN = 60  # горизонт для поиска падающего окна


def main():
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join(
        "logs", datetime.datetime.now().strftime("backtest_down_%Y%m%d_%H%M.log")
    )

    def emit(msg=""):
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(msg + "\n")

    emit("=" * 80)
    emit("  BACKTEST-DOWN: 2.5 vs 3.0-uni vs 3.0-multi (falling window)")
    emit("  " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M MSK"))
    emit("=" * 80)

    frames = {}
    for t in TICKERS:
        emit(f"\n[{t}] Fetching {DAYS}d hourly...")
        df = fetch_moex_candles(t, days=DAYS)
        if df is None or len(df) < 500:
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
    if T < 700:
        emit("  ERROR: join too short")
        return

    closes = {t: joint[f"close_{t}"].values.astype(np.float64) for t in TICKERS}
    vols = {t: joint[f"vol_{t}"].fillna(0).values.astype(np.float64)
            for t in TICKERS}

    # Фаза 1: скан падающего окна (средний форвард-доход корзины за H_SCAN)
    best, best_c = 1e9, None
    cands = range(200, T - H_SCAN - 100, 10)
    for c in cands:
        chgs = [((closes[t][c + H_SCAN] / closes[t][c]) - 1) * 100
                for t in TICKERS]
        m = sum(chgs) / len(chgs)
        if m < best:
            best, best_c = m, c
    anchor = best_c
    emit(f"\n[SCAN] anchor cutoff idx={anchor} ({joint['begin'].iloc[anchor]}), "
         f"mean fwd{H_SCAN}={best:+.2f}%")
    # Срезы вокруг якоря (смещения в шагах от anchor); в индексы ниже
    plan_idx = []
    for back, h in [(T - (anchor + d), h) for d, h in
                    [(-120, 30), (-90, 30), (-60, 30), (-120, 60),
                     (-90, 60), (-60, 60), (-150, 90), (-120, 90)]]:
        c = T - back
        if c >= 200 and c + h <= T:
            plan_idx.append((c, h))
    emit(f"[PLAN] {len(plan_idx)} срезов: " +
         ", ".join(f"T-{T - c}/h{h}" for c, h in plan_idx))

    t0 = time.time()
    emit("\n[INIT] Loading TimesFM 2.5...")
    m25 = load_timesfm()
    emit("\n[INIT] Loading TimesFM-3...")
    m3 = load_timesfm3()
    emit(f"[INIT] loads done in {time.time() - t0:.1f}s")

    recs = []
    for c, h in plan_idx:
        cdate = joint["begin"].iloc[c]
        emit(f"\n--- cutoff idx={c} ({cdate}), horizon {h} ---")
        tgt = np.stack([closes[t][:c] for t in TICKERS]).astype(np.float32)
        vcov = np.stack([vols[t][:c] for t in TICKERS]).astype(np.float32)
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
            row = dict(ticker=t, horizon=h)
            try:
                p = float(timesfm_forecast(m25, ctx.astype(np.float32), h)[-1])
                chg = (p - last) / last * 100
                s, _ = compute_signal(p, last)
                recs.append((dict(row, model="T25",
                                  end_err=abs(chg - real_chg),
                                  hit=(s == s_real))))
            except Exception as e:
                emit(f"  [{t}] T25 ERROR: {e}")
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

    emit(f"\n{'=' * 80}\nBACKTEST-DOWN RESULTS")
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
