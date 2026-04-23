# Стресс-тестирование GARCH на синтетических данных от GAN

Папка содержит исследование для аспирантской конференции.

## Структура

```
notebooks/research/
├── src/                      # переиспользуемый код
│   ├── data.py               # загрузка S&P 500 через yfinance, train/test split
│   ├── garch_eval.py         # GARCH(1,1)-t fit, fixed-parameter walk-forward, метрики, VaR, Kupiec PoF
│   ├── timegan.py            # упрощённая PyTorch-реализация TimeGAN (Yoon et al., 2019)
│   ├── ctgan_wrapper.py      # CTGAN на оконных данных (используется как контрастный пример)
│   ├── stylized.py           # стилизованные факты финансовых рядов
│   └── plots.py              # все графики для отчёта
├── run_all.py                # один скрипт-оркестратор, считает 4 ветки эксперимента
├── artifacts/                # результаты прогона run_all.py
│   ├── synth_timegan.csv, synth_ctgan.csv
│   ├── garch_params_*.json   # параметры GARCH по веткам
│   ├── forecasts_B*.csv      # прогнозы σ̂ и VaR на test
│   ├── metrics_summary.csv   # итоговая таблица метрик
│   ├── stylized_facts.csv
│   ├── timegan_history.json, timegan_model.pt
│   └── figures/*.png         # 8 графиков для доклада
├── 00_overview.ipynb           # данные и постановка
├── 01_baseline_garch.ipynb     # B1: real-only GARCH
├── 02_synthetic_generation.ipynb   # TimeGAN vs CTGAN, стилизованные факты
├── 03_tstr_evaluation.ipynb    # B2-TimeGAN, B2-CTGAN — TSTR
├── 04_augmentation.ipynb       # B3 — augmentation
└── 05_results_summary.ipynb    # сводные таблицы и графики для доклада
```

## Воспроизведение с нуля

```powershell
# 1) окружение (один раз)
python -m venv .venv
.\.venv\Scripts\pip install --upgrade pip wheel setuptools
.\.venv\Scripts\pip install numpy pandas scipy matplotlib seaborn scikit-learn `
                            arch yfinance statsmodels jupyter ipykernel notebook tqdm
.\.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\pip install ctgan
.\.venv\Scripts\python -m ipykernel install --user --name evst-research --display-name "Python (EVST research)"

# 2) полный прогон (загрузка данных + B1 + TimeGAN + CTGAN + B3) — ~10 мин на RTX 5070 Ti
.\.venv\Scripts\python notebooks\research\run_all.py

# 3) графики
.\.venv\Scripts\python notebooks\research\src\plots.py

# 4) ноутбуки
.\.venv\Scripts\python -m jupyter lab notebooks\research
```

## Ветки эксперимента

| Ветка | Train для GARCH | GAN-источник | Что исследует |
|---|---|---|---|
| **B1** | real 2010–2019 | — | baseline |
| **B2-TimeGAN** | synth(2515) от TimeGAN(real train) | TimeGAN | TSTR (Q1) |
| **B2-CTGAN** | synth(2515) от CTGAN(real train) | CTGAN | контраст к B2 |
| **B3** | real train + synth(2515) | TimeGAN | Augmentation (Q2) |

Во всех ветках: GARCH(1,1)-t, параметры обучаются один раз и **фиксируются**, walk-forward один шаг вперёд на real test 2020–2024 (1258 дней).

## Краткие выводы

- **Q1 (TSTR):** даже специализированный TimeGAN при чистой замене реальных данных синтетикой ухудшает прогноз GARCH (RMSE ↑, β→0, нарушения VaR(1%) в 3.5 раза выше нормы). CTGAN — катастрофа.
- **Q2 (Augmentation):** добавление TimeGAN-синтетики к реальному train **умеренно улучшает** прогноз волатильности (RMSE: 0.869 vs 0.887, MZ-наклон ближе к 1, нарушений VaR(5%) меньше).
- **Главный практический итог:** синтетика от GAN полезна как дополнение, но не как замена в задаче моделирования волатильности.
