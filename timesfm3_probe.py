"""TimesFM-3 smoke probe — univariate parity scaffold (ЭКСПЕРИМЕНТ, F:/timesfm3-experiment).

Ничего общего с рабочим C:/Users/User/timesfm-kronos-comparison не имеет:
свой venv, свой скрипт, синтетические данные (без MOEX/HF-зависимостей кроме весов).

Шаг 1: синтетика -> TimesFM-3 univariate -> shapes + timing.
Шаг 2 (позже): те же тикеры/горизонты, что в timesfm_kronos_compare.py.
"""
import os
import time

# Force CPU: RTX 5060 Ti (Blackwell sm_120) не поддерживается сборками torch
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np

t0 = time.time()
from timesfm3 import TimesFM3Evaluator, ModelConfig
t_import = time.time() - t0
print(f"[probe] import timesfm3: {t_import:.1f}s", flush=True)

t0 = time.time()
config = ModelConfig(
    checkpoint_path="google/timesfm-3.0-pytorch",
    per_core_batch_size=8,
    device="cpu",
)
forecaster = TimesFM3Evaluator(config)
t_load = time.time() - t0
print(f"[probe] load 3.0-pytorch (CPU): {t_load:.1f}s", flush=True)

# Два ряда разной длины — как в официальном примере
ts1 = np.linspace(0, 1, 100).astype(np.float32)
ts2 = np.sin(np.linspace(0, 24, 72)).astype(np.float32)

t0 = time.time()
outputs = list(
    forecaster.predict_batch(
        [ts1, ts2],
        horizon=12,
        return_quantiles=True,
        use_symmetric_averaging=False,
    )
)
t_infer = time.time() - t0

for i, out in enumerate(outputs):
    print(
        f"[probe] series{i}: forecast{out.forecast.shape} "
        f"quantiles{out.quantiles.shape}",
        flush=True,
    )
print(f"[probe] infer (2 series x h=12): {t_infer:.1f}s", flush=True)
print("[probe] OK", flush=True)
