"""TimesFM-3 multivariate vs univariate на MOEX-корзине (ЭКСПЕРИМЕНТ).

Дизайн:
  target (multi):      close SBERP + GAZP + LKOH, состыкованные по begin (inner join)
  past-only covariate: volume тех же тикеров (3 канала) — известны только исторически
  past-future:         нет (v1; календарь сессии — follow-up, см. вывод)
  baseline:            TimesFM-3 univariate на каждом тикере отдельно (те же ряды)

Сигналы тем же compute_signal (±2%). Лог: logs/multi_YYYYMMDD_HHMM.log
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
    compare_signals,
)
from timesfm3 import TimesFM3Evaluator, ModelConfig  # noqa: E402

TICKERS = ["SBERP", "GAZP", "LKOH"]
HORIZONS = [30, 60, 90]


def load_timesfm3():
    print("[TimesFM-3] Loading model...", flush=True)
    config = ModelConfig(
        checkpoint_path="google/timesfm-3.0-pytorch",
        per_core_batch_size=8,
        device="cpu",
    )
    m = TimesFM3Evaluator(config)
    print("[TimesFM-3] Ready (device=cpu)!", flush=True)
    return m


def main():
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join(
        "logs", datetime.datetime.now().strftime("multi_%Y%m%d_%H%M.log")
    )

    def emit(msg=""):
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(msg + "\n")

    emit("=" * 80)
    emit("  TIMESFM-3 MULTIVARIATE (basket+volume) vs UNIVARIATE — MOEX")
    emit("  " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M MSK"))
    emit("=" * 80)

    # 1. Fetch + inner-join по begin
    frames = {}
    for t in TICKERS:
        emit(f"\n[{t}] Fetching MOEX ISS data (60 days)...")
        df = fetch_moex_candles(t, days=60)
        if df is None or len(df) < 100:
            emit(f"  ERROR: not enough data for {t}")
            return
        frames[t] = df[["begin", "close", "volume"]].rename(
            columns={"close": f"close_{t}", "volume": f"vol_{t}"}
        )
        emit(f"  Got {len(df)} hourly candles")
    joint = frames[TICKERS[0]]
    for t in TICKERS[1:]:
        joint = pd.merge(joint, frames[t], on="begin", how="inner")
    joint = joint.sort_values("begin").reset_index(drop=True)
    emit(f"\n[JOIN] aligned rows: {len(joint)} "
         f"({joint['begin'].iloc[0]} — {joint['begin'].iloc[-1]})")
    if len(joint) < 100:
        emit("  ERROR: join too short")
        return

    target = np.stack(
        [joint[f"close_{t}"].values for t in TICKERS]
    ).astype(np.float32)  # (3, T)
    vol_cov = np.stack(
        [joint[f"vol_{t}"].fillna(0).values for t in TICKERS]
    ).astype(np.float32)  # (3, T) past-only
    lasts = {t: float(joint[f"close_{t}"].iloc[-1]) for t in TICKERS}
    emit(f"[DATA] target{target.shape} vol_cov{vol_cov.shape}")

    # 2. Model
    t0 = time.time()
    m3 = load_timesfm3()
    emit(f"[INIT] load: {time.time() - t0:.1f}s")

    # 3. Horizons
    all_results = []
    for h in HORIZONS:
        emit(f"\n{'=' * 80}\n--- Horizon: {h} ---")
        # multivariate: один вызов на все 3 цели
        t0 = time.time()
        out = list(
            m3.predict_batch(
                contexts=[target],
                horizon=h,
                past_only_covariates=[vol_cov],
                return_quantiles=False,
                use_symmetric_averaging=False,
            )
        )[0]
        t_multi = time.time() - t0
        multi_preds = np.asarray(out.forecast, dtype=np.float64)  # (3, h)
        emit(f"  [multi] joint forecast{multi_preds.shape} ({t_multi:.1f}s)")
        for i, t in enumerate(TICKERS):
            s_m, c_m = compute_signal(float(multi_preds[i, -1]), lasts[t])
            # univariate baseline на том же ряде
            t0 = time.time()
            out_u = list(
                m3.predict_batch(
                    [target[i]],
                    horizon=h,
                    return_quantiles=False,
                    use_symmetric_averaging=False,
                )
            )[0]
            t_uni = time.time() - t0
            p_u = float(np.asarray(out_u.forecast, dtype=np.float64)[-1])
            s_u, c_u = compute_signal(p_u, lasts[t])
            agree, conf = compare_signals(s_u, c_u, s_m, c_m)
            emit(f"  {t}: last={lasts[t]:.2f} | uni {p_u:.2f} ({c_u:+.2f}%) [{s_u}] "
                 f"({t_uni:.1f}s) | multi {float(multi_preds[i, -1]):.2f} "
                 f"({c_m:+.2f}%) [{s_m}] => {agree}/{conf}")
            all_results.append(dict(ticker=t, horizon=h, last=lasts[t],
                                    pu=p_u, su=s_u, cu=c_u,
                                    pm=float(multi_preds[i, -1]), sm=s_m, cm=c_m,
                                    agree=agree, conf=conf))

    emit(f"\n{'=' * 80}\nSUMMARY (uni vs multi)")
    emit("=" * 80)
    for h in HORIZONS:
        emit(f"\n--- Горизонт {h} ---")
        for r in [x for x in all_results if x["horizon"] == h]:
            emit(f"{r['ticker']:<6} last={r['last']:.2f} | "
                 f"uni [{r['su']}] {r['cu']:+.2f}% | "
                 f"multi [{r['sm']}] {r['cm']:+.2f}% => {r['agree']}")
    n_same = sum(1 for r in all_results if r["agree"] == "СОВПАДЕНИЕ")
    emit(f"\nСовпадений uni/multi: {n_same}/{len(all_results)}")
    emit(f"\nLog saved: {log_path}\nDONE")


if __name__ == "__main__":
    main()
