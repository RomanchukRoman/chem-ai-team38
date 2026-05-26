# ChemAI: Predict the Cure — Team 38

Решение хакатона МИФИ по предсказанию `IC50`, `CC50` и `SI` для химических соединений
против вируса гриппа. Метрика соревнования — RMSE, усреднённый по трём таргетам.

Ссылка на соревнование:
<https://www.kaggle.com/competitions/chem-ai-predict-the-cure>

## Структура репозитория

```
chem-ai-team38/
├── pyproject.toml          # зависимости (uv)
├── README.md
├── data/
│   └── raw/                # train.csv, test.csv, sample_submission.csv (gitignored)
├── notebooks/              # исследовательский EDA-ноутбук
├── scripts/
│   └── run_full_hpo.sh     # запускает Optuna для всех (model, target)
├── src/
│   ├── seeds.py            # фиксация всех random-состояний
│   ├── data.py             # загрузка и дедупликация трейна
│   ├── features.py         # leak-safe препроцессинг + PCA/KMeans
│   ├── models.py           # CatBoost + LightGBM + XGBoost ensemble
│   ├── tuning.py           # Optuna HPO per (model, target)
│   ├── train.py            # CLI: обучение + сохранение артефактов
│   └── predict.py          # CLI: сборка submission.csv из артефактов
├── artifacts/              # OOF/test предсказания + cv_report.json (gitignored)
└── submissions/            # итоговые submission CSV
```

## Установка

Используется [`uv`](https://docs.astral.sh/uv/) и Python 3.14+.

```bash
cd chem-ai-team38
uv sync
```

## Подготовка данных

Данные закрыты лабораторией и доступны по ссылке из условия хакатона
(Google Drive). Положи скачанные `train.csv` и `test.csv` в `data/raw/`:

```bash
mkdir -p data/raw
cp ~/Downloads/train.csv  data/raw/
cp ~/Downloads/test.csv   data/raw/
```

## Воспроизведение результата

```bash
# (опционально) полный поиск гиперпараметров через Optuna — ~1 час
bash scripts/run_full_hpo.sh

# обучение (3 seeds × 5 folds × {CatBoost, LightGBM, XGBoost} × 3 таргета)
# автоматически подхватывает best_params, если они сохранены
uv run python -m src.train --n-seeds 3 --n-splits 5

# сборка submission на основе кэшированных предсказаний
uv run python -m src.predict --si blend --out submissions/submission.csv
```

### Гиперпараметрический поиск

`src.tuning` запускает Optuna-исследование по одной комбинации (модель, таргет):

```bash
uv run python -m src.tuning --model catboost --target "CC50, mM" --n-trials 50
uv run python -m src.tuning --model lightgbm --target "IC50, mM" --timeout 1200
```

Внутри одного trial — `KFold(3)`, log1p-таргет, early-stopping. Лучшие
параметры сохраняются в `artifacts/best_params/{model}_{target_slug}.json`
и автоматически подгружаются `make_catboost`/`make_lightgbm`/`make_xgboost`
при следующем `src.train`.

`predict.py --si` принимает три стратегии для предсказания SI:

| Стратегия | Что делает |
|---|---|
| `ratio` | `SI = CC50_pred / IC50_pred` — арифметическое тождество |
| `model` | Отдельная модель, обученная на `log1p(SI)` |
| `blend` | Линейная смесь `α * model + (1−α) * ratio` с `α`, подобранным на OOF |

Все три варианта сохраняются в `submissions/`. Лучший по `cv_report.json`
заливается на Kaggle.

Быстрая проверка пайплайна (1 seed × 3 folds, только CatBoost):

```bash
uv run python -m src.train --quick
```

## Подход

### Данные

* Трейн: 751 строка × 214 колонок, после дедупликации — **628 уникальных
  молекул** (одни и те же молекулы измеряли в нескольких сериях; для них
  таргеты агрегируются медианой).
* Тест: 250 строк × 211 колонок.
* Признаки: 210 числовых RDKit-дескрипторов (включая `FpDensityMorgan*`,
  топологические `Chi*`, `Kappa*`, поверхностные `SMR_VSA*`/`SlogP_VSA*` и
  бинарные `fr_*` функциональных групп). SMILES в данных нет — морган-фингерпринты
  пересчитать невозможно.
* Все три таргета сильно скошены вправо (`IC50` skew=3.79, `CC50`=2.06,
  `SI`=15.63). Работаем в `log1p`-пространстве с обратным `expm1` при инференсе.
* **`SI = CC50/IC50` с точностью до double-precision** — это арифметическое
  тождество, а не учебная цель. В пайплайне предсказание SI собирается двумя
  способами и берётся то, что лучше на OOF.

### Препроцессинг (leak-safe)

Всё, что имеет состояние, лежит в одном `sklearn.Pipeline`, который фитится
**внутри каждого CV-фолда**:

1. `SimpleImputer(strategy='median')` — пропуски в 12 колонках.
2. `VarianceThreshold(0)` — выбрасывает константные признаки (~18 штук).
3. `StandardScaler` — нормализация.
4. `CorrelationFilter(threshold=0.98)` — удаляет один признак из каждой пары
   с `|corr| > 0.98` (~37 штук).
5. `UnsupervisedFeatureExpander` — конкатенирует к признакам:
   * 40 главных компонент PCA;
   * метку из 12 кластеров KMeans;
   * 12 расстояний до центров кластеров.

После пайплайна имеем ≈226 признаков.

### Модели

Для каждого таргета обучается ансамбль из трёх градиентных бустингов на
одинаковых признаках:

| Модель | Гиперы (основные) |
|---|---|
| CatBoost  | iter=4000, lr=0.02, depth=5, l2=5 |
| LightGBM  | n=4000, lr=0.02, num_leaves=31, λ=5 |
| XGBoost   | n=4000, lr=0.02, depth=5, λ=5 |

`KFold(n_splits=5, shuffle=True)` × 3 seeds × ранний стоп на 300 итераций.
Предсказания усредняются по фолдам/seed-ам внутри каждой модели, затем
три модели усредняются простым средним.

### Воспроизводимость

* `src/seeds.py::set_seed` фиксирует `PYTHONHASHSEED`, `random`, `numpy`.
* Все модели принимают `random_seed` (или `random_state`); seed-ы детерминированы.
* `n_init=20` для KMeans — детерминированный фит при фиксированном seed.
* Препроцессор фитится только на train-части CV-фолда, без утечки.
* Все артефакты (OOF и test предсказания, `cv_report.json`) сохраняются
  в `artifacts/`. `submission.csv` собирается из них без повторного обучения.

## Что было изменено по сравнению с исходным Colab-ноутбуком

* Перенесли пайплайн из двух Jupyter-ноутбуков в модули `src/*.py`.
* Заменили `!wget` к Google Drive на чтение из `data/raw/`.
* `imputer`, `scaler`, `VarianceThreshold`, корреляционный фильтр, `PCA`
  и `KMeans` теперь фитятся внутри CV-фолда (раньше — на полном трейне до фолда).
* Добавили LightGBM и XGBoost к CatBoost — ансамбль из трёх моделей.
* `SI` собирается тремя способами с подбором `α` по OOF.
* Удалили зависимость от Colab (`from google.colab import files`).
* Зафиксировали все seed-ы.
