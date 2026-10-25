# TimesFM 2.5 + Kronos Mini — Сравнение сигналов MOEX

[![tests](https://github.com/L-MORIA/timesfm3-experiment/actions/workflows/tests.yml/badge.svg)](https://github.com/L-MORIA/timesfm3-experiment/actions/workflows/tests.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

AI-ансамбль из двух моделей для прогнозирования цен акций MOEX (Сбербанк преф, Газпром, Лукойл).

## Суть проекта

Проект отвечает на один вопрос: **согласуются ли сигналы универсальной
foundation-модели и узкоспециализированной модели на одном и том же рынке**.

Сравниваются два принципиально разных подхода:

- **TimesFM 2.5** — большая foundation-модель Google (200M), обученная на
  разнородных временных рядах и не видевшая MOEX;
- **Kronos Mini** — компактная (4M) авторегрессионная модель, обученная
  специально на биржевых свечах.

Обе модели получают одну и ту же историю часовых свечей с MOEX ISS и строят
точечный прогноз на горизонт 30/60/90 часов. Из прогноза выводится сигнал
**ПОКУПКА / ПРОДАЖА / ДЕРЖАТЬ** (порог ±2%), после чего сигналы сопоставляются.

Практическая ценность — **гибридный ансамбль**: когда обе модели независимо
приходят к одному выводу (CONCORDANT), уверенность в сигнале выше; расхождение
(CONFLICT) — повод воздержаться от сделки. Побочно проверяется гипотеза, что
тяжёлая foundation-модель без доменного обучения даёт на MOEX сигналы,
сопоставимые со специализированной моделью.

Дочерние скрипты `timesfm3_*.py` расширяют эксперимент: паритет
TimesFM 2.5 ↔ TimesFM-3 и backtest прибыльности сигналов на истории.

## Что делает

Загружает данные с MOEX ISS API → запускает **TimesFM 2.5** (Google Foundation Model) и **Kronos Mini** (4M params) → сравнивает сигналы BUY/SELL/HOLD на горизонтах 30/60/90 часов (часовые свечи MOEX).

## Быстрый старт — выбор варианта

### Вариант А: CUDA (GPU с sm_80+ — Ampere RTX 30xx / A100 / H100 / Hopper)
```bash
# Создать venv (Python 3.12; НЕ использовать Hermes global venv)
uv venv .venv --python 3.12

# Установить зависимости с CUDA (PyTorch cu121)
uv pip install -r requirements-cuda.txt

# Запустить сравнение (автоматически использует GPU)
python timesfm_kronos_compare.py
```

### Вариант Б: CPU-only (любой CPU, GPU без CUDA-поддержки, Blackwell RTX 50xx sm_120)
```bash
# Создать venv (Python 3.12; НЕ использовать Hermes global venv)
uv venv .venv --python 3.12

# Установить зависимости CPU-only (PyTorch CPU)
uv pip install -r requirements-cpu.txt

# Запустить сравнение (принудительно CPU)
python timesfm_kronos_compare.py
```

> **Важно:** пин `cu121` в `requirements-cuda.txt` **не содержит** ядер для Blackwell
> (sm_120, RTX 50xx) — на таких GPU CUDA-вариант из этого файла упадёт с
> «no kernel image is available for execution on the device». Стабильный PyTorch 2.7+
> из индекса `cu128` sm_120 поддерживает — при желании соберите свой стек с
> `--index-url https://download.pytorch.org/whl/cu128`. Иначе используй CPU-вариант.
> Если у тебя Ampere (RTX 30xx) или новее с поддержкой — CUDA-вариант даст ускорение в 5-10x.

> **Принудительный CPU:** в `load_kronos()` стоит `device="cpu"`, а во всех скриптах
> в шапке выставляется `os.environ["CUDA_VISIBLE_DEVICES"] = "-1"` (это принуждает
> к CPU и TimesFM). Для CUDA-запуска поменяй `device` на `"cuda"` в `load_kronos()`
> и убери переопределение `CUDA_VISIBLE_DEVICES`.

## Результаты

- **stdout**: таблица сравнения + интерпретация сигналов
- **График**: `timesfm_kronos_comparison.png`
- **Лог**: `logs/comparison_YYYYMMDD.log` (автоматически)

## Архитектура

```
timesfm3-experiment/
├── timesfm_kronos_compare.py       # основной скрипт: TimesFM 2.5 + Kronos Mini
├── timesfm3_parity.py              # эксперимент: 2.5 vs 3.0 univariate
├── timesfm3_multi.py               # эксперимент: 3.0 multivariate vs univariate
├── timesfm3_backtest.py            # эксперимент: backtest 2.5/3.0 на истории
├── timesfm3_backtest_down.py       # эксперимент: backtest на падающем окне
├── timesfm3_probe.py               # эксперимент: smoke-проба TimesFM-3
├── requirements.txt                # CPU-only стек (дубль requirements-cpu.txt, для CI-кэша)
├── requirements-cpu.txt            # CPU-only стек (torch CPU)
├── requirements-cuda.txt           # CUDA стек (torch cu121)
├── test_timesfm_kronos_compare.py  # pytest-тесты (горизонты, fetch, сигналы)
├── pyproject.toml                  # конфигурация ruff для CI
├── .github/workflows/tests.yml     # CI: pytest + ruff на push/PR
├── logs/                           # результаты (автоматически, в .gitignore)
│   └── comparison_YYYYMMDD.log
├── LICENSE
├── .gitignore
└── README.md
```

## Разработка

```bash
# Тесты (нужен pytest в окружении)
python -m pytest -q

# Линтер (набор правил — в pyproject.toml)
ruff check .
```

Обе проверки гоняются автоматически в GitHub Actions на каждый push и PR.

> Kronos Mini загружается из соседнего проекта `~/kronos-signal/`
> (модели лежат в `~/kronos-signal/models/Kronos-mini/` и
> `~/kronos-signal/models/Kronos-Tokenizer-2k/`). См. `load_kronos()`.

## Модели

| Модель | Размер | Тип | Горизонт |
|--------|--------|-----|----------|
| **TimesFM 2.5** | 200M params | Foundation Model (Google) | до 128 шагов (часов) |
| **Kronos Mini** | 4M params | Autoregressive (MOEX) | до ~20 дней |

> **Ограничение горизонта:** `HORIZONS = [30, 60, 90]` (в часах) защищён в `main()`:
> `max(HORIZONS) > 128` → `raise ValueError` (guard переживает запуск с `python -O`).
> Чтобы выйти за 128 часов, пересоберите TimesFM с большим `max_horizon` в `load_timesfm()`.

## Сигналы

- **CONCORDANT**: обе модели согласны → HIGH confidence
- **NEUTRAL-MATCH**: обе близки к текущей цене → MEDIUM confidence
- **CONFLICT**: модели расходятся → LOW confidence, ждать подтверждения
- **ОШИБКА**: прогноз одной из моделей не удался (ERROR) → согласие не
  засчитывается, в логе «НЕТ ДАННЫХ»

## Производительность

Внутри `main()` результаты 30-часового прогноза (история + TimesFM-forecast)
кэшируются в `chart_data[ticker]`, поэтому секция графиков **не** повторяет
запрос к MOEX ISS (~60 HTTP-вызовов на тикер) и **не** гоняет TimesFM
повторно. Это сокращает время работы примерно вдвое.

## Зависимости

- Python 3.12+
- `timesfm` (Google)
- `torch` — выбери вариант: CPU-only (`requirements-cpu.txt`) или CUDA (`requirements-cuda.txt`)
- `einops`, `requests`, `numpy`, `pandas`, `matplotlib`
- Kronos Mini — локально из `~/kronos-signal/models/`
- MOEX ISS API (бесплатно, без ключа)

## Лицензия

[MIT](LICENSE) © 2026 L-MORIA