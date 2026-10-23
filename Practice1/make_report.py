import json
import pathlib

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent


def table(df):
    z = df.copy()
    for c in z:
        z[c] = z[c].map(
            lambda v: (
                "—"
                if pd.isna(v)
                else (f"{v:.6f}" if isinstance(v, (float, np.floating)) else str(v))
            )
        )
    return (
        "| "
        + " | ".join(z.columns)
        + " |\n| "
        + " | ".join(["---"] * len(z.columns))
        + " |\n"
        + "\n".join("| " + " | ".join(r) + " |" for r in z.to_numpy())
    )


def main():
    r = ROOT / "results"

    def load(n):
        return json.loads((r / n).read_text(encoding="utf8"))

    audit = load("data_audit.json")
    frozen = load("frozen_selection.json")
    ci = load("uncertainty.json")
    runtime = load("runtime.json")
    trials = pd.DataFrame(load("validation_trials.json"))
    test = pd.read_csv(r / "test_metrics.csv")
    q = pd.read_pickle(ROOT / "data/queries.pkl")
    e = pd.read_pickle(ROOT / "data/events.pkl")
    manual = test[test.model == "manual"].iloc[0]
    auto = test[test.model == "automatic"].iloc[0]
    aimp = pd.read_csv(r / "automatic_importance.csv")
    testcut = q.loc[q.split == "test", "cutoff"].iloc[0]
    future = e[(e.date >= testcut) & (e.date < testcut + pd.Timedelta(days=7))]
    testqueries = q[q.split == "test"]
    cold = audit["snapshots"][-1]["excluded_cold_positives"]
    lo, hi = ci["paired_user_bootstrap_95"]
    delta = ci["delta_ndcg"]
    conclusion = (
        "Положительный прирост NDCG@10 подтверждён парным интервалом в пределах этого тестового протокола."
        if lo > 0
        else "Автоматический набор уступил ручному по NDCG@10; весь парный интервал ниже нуля."
        if hi < 0
        else "Парный интервал включает ноль: уверенного улучшения NDCG@10 на этом тесте не установлено."
    )

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    names = ["Популярность 30 дн.", "Ручной CatBoost", "Автопоиск + CatBoost"]
    ordered = test.set_index("model").loc[["popularity30", "manual", "automatic"]]
    for ax, col, title in zip(axs, ["ndcg10", "recall10"], ["NDCG@10", "Recall@10"]):
        vals = ordered[col].to_numpy()
        bars = ax.barh(names, vals, color=["#a6adb4", "#2e6485", "#cd8a36"])
        ax.set_xlim(0, min(1, max(vals) * 1.28))
        ax.set_title(title)
        ax.set_xlabel("Среднее по запросам")
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)
        for b, v in zip(bars, vals):
            ax.text(v + 0.007, b.get_y() + b.get_height() / 2, f"{v:.4f}", va="center")
    fig.suptitle(
        f"KION · sampled reranking · test 16–22 августа 2021 · {len(testqueries.groupby('user_id'))} пользователей",
        fontsize=12,
    )
    fig.savefig(r / "comparison.png", dpi=160)
    plt.close(fig)
    snapshots = pd.DataFrame(audit["snapshots"])[
        ["cutoff", "split", "history_rows", "queries", "positives", "excluded_cold_positives"]
    ]
    grid = trials[trials.name.str.contains("_grid_")][
        ["name", "columns", "ndcg10", "recall10", "logloss", "fit_seconds"]
    ]
    ab = trials[
        trials.name.isin(
            [
                "manual_grid_0",
                "all_generated",
                "manual_without_indirect",
                "manual_without_direct",
                "automatic_only",
            ]
        )
    ][["name", "columns", "ndcg10", "recall10", "logloss"]]
    ref = float(trials.loc[trials.name == "manual_grid_0", "ndcg10"].iloc[0])
    val_gain = float(
        trials.loc[trials.name.str.startswith("auto_grid_"), "ndcg10"].max()
        - trials.loc[trials.name.str.startswith("manual_grid_"), "ndcg10"].max()
    )
    ablation_deltas = "; ".join(
        f"`{row['name']}`: {row['ndcg10'] - ref:+.6f}"
        for _, row in ab.iterrows()
        if row["name"] != "manual_grid_0"
    )
    av = ab.set_index("name").ndcg10
    selection_gap = float(
        trials.loc[trials.name == "auto_grid_0", "ndcg10"].iloc[0] - av["all_generated"]
    )

    from experiment import metrics

    pred = pd.read_csv(r / "test_predictions.csv.gz", float_precision="round_trip")
    history = e[e.date < testcut].groupby("user_id").size()
    n = testqueries.user_id.map(history).fillna(0).to_numpy()
    groups = []
    for name, mask in [
        ("Нет прошлых записей", n == 0),
        ("1–4 прошлых записи", (n >= 1) & (n <= 4)),
        ("5+ прошлых записей", n >= 5),
    ]:
        if not mask.any():
            continue
        for model in ("manual", "automatic"):
            met = metrics(testqueries[mask], pred.loc[mask, model].to_numpy())
            groups.append(
                {
                    "история": name,
                    "модель": model,
                    "запросы": met["queries"],
                    "NDCG@10": met["ndcg10"],
                    "Recall@10": met["recall10"],
                }
            )
    subgroups = pd.DataFrame(groups)
    subgroups.to_csv(r / "test_subgroups.csv", index=False)
    budgets = []
    for label, mask in [
        ("Ручная сетка", trials.name.str.startswith("manual_grid_")),
        ("Поиск пакетов", trials.name.str.startswith("search_")),
        ("Автоматическая сетка", trials.name.str.startswith("auto_grid_")),
        ("Диагностические абляции", ~trials.name.str.contains("_grid_|search_")),
    ]:
        budgets.append(
            {
                "этап": label,
                "число fit": int(mask.sum()),
                "fit, секунд": trials.loc[mask, "fit_seconds"].sum(),
            }
        )
    report = f"""# Автопоиск признаков KION

## Методы: от исходной реализации к текущей

**Изначально.** Проверяли автоматизацию ручных агрегатов при одном алгоритме ранжирования — CatBoostClassifier. Baseline: 34 признака активности пользователя, популярности объекта, прямой связи user–item и интереса к жанру/типу. Автопоиск перебирал пакеты окон 2/5/10/21/60 дней и затухания с half-life 3/10/21, до двух жадных добавлений по validation NDCG@10. Из 89 доступных признаков выбраны окна 2 и 21 день, итог — 50. На исходном test NDCG@10 0.360780 → 0.358450, интервал разности включает ноль: улучшение не подтверждено. Автопоиск реализован обычным алгоритмом, без ИИ-агента/генерации SQL.

**Что проверили затем.** Спектральные признаки графа; ItemKNN (сходство объектов по совместным просмотрам); расширенный набор `rich`; CatBoostRanker с YetiRank/NDCG@10. В одинаковом скрининге validation NDCG@10: baseline 0.361023, граф 0.368486, ItemKNN 0.382607, rich 0.384640, ranker 0.354487. Выиграл rich: 34 ручных + 6 графовых сигналов SVD32 + 3 ItemKNN + дополнительные окна 2/14, затухание 3/14 и доли интереса по всем жанрам; всего 73 признака. Это фиксированный расширенный набор, а не результат прежнего жадного автопоиска. EASE рассмотрен, но не запускался из-за плотной матрицы O(I²); GNN/трансформер отложены из-за бюджета и ограничений last-watch KION, их качество не измерялось.

**К чему пришли.** Грамматика окон сама по себе не обеспечила прирост на следующей неделе; сочетание агрегатов и сигналов сходства оказалось полезнее. После фиксации конфигураций сравнение повторили с переобучением на другой когорте из 5000 пользователей: NDCG@10 0.353255 → 0.372940, семейно скорректированный интервал Δ [0.008137; 0.031558]. На уже просмотренном старом test выполнена дополнительная проверка отсутствия регрессии: NDCG@10 0.360780 → 0.378699, Recall@10 0.526525 → 0.545733. Новая когорта относится к тому же периоду; перенос на другие периоды/промышленный retrieval не доказан.

**Что используется сейчас.** `predict.py` по умолчанию загружает **CatBoostClassifier с 73 признаками rich**, `results/upgrades/active.cbm`; depth=7, iterations=320, learning_rate=0.05, seed=2026. Источник конфигурации — [active_model.json](active_model.json). Для пары user–item признаки вычисляются из истории до cutoff, затем объекты сортируются по оценке. Это пакетный sampled reranking, не онлайн-сервис и не оценка CTR. Исходные manual/automatic и их таблицы ниже сохранены для воспроизводимости; они не описывают активную версию. Полные сравнения, условия внедрения и команды — [IMPROVEMENTS.md](IMPROVEMENTS.md).

**Исходный эксперимент:** CatBoost, 34 ручных → 50 отобранных признаков. Test NDCG@10 **{manual.ndcg10:.6f} → {auto.ndcg10:.6f}**, Δ **{delta:+.6f} ({delta / manual.ndcg10 * 100:+.2f}%)**, парный 95% CI **[{lo:+.6f}; {hi:+.6f}]**. {conclusion}

![Test](results/comparison.png)

## Данные и протокол

Прогноз просмотров следующей недели. [KION](https://github.com/irsafilo/KION_DATASET), commit `{audit["commit"]}`: полный `data_original.zip` — **{audit["source_rows"]:,} строк / {audit["source_users"]:,} пользователей**, {audit["min_date"]}–{audit["max_date"]} (конечный год карточки ошибочен); сокращённый `interactions.csv.zip` — 1 594 787 строк. [Yambda](https://huggingface.co/datasets/yandex/yambda) рассмотрен, не использован.

`sort(unique user_id)` → `default_rng(42)` → **{audit["sampled_users"]:,} пользователей**, без возвращения/фильтра активности; все **{audit["sampled_events"]:,} строк**, включая однократных. Дубликаты источника/выборки: {audit["duplicate_rows"]}/{audit["sample_duplicate_rows"]}, повторы user–item: {int(e.duplicated(["user_id", "item_id"]).sum())}; user/item/date без пропусков, {audit["input_nulls"]["watched_pct"]} пропусков watched_pct → 0.

KION хранит **последний просмотр пары**, условно `event_type=watch`: без показов/кликов/сырых повторов. Отсев будущих опубликованных строк проверен, но last-watch может скрывать ранние действия: точное прошлое сервиса не восстановить. Первый жанр, тип и возрастной рейтинг считаются постоянными (версий нет); демография/тексты не используются.

{table(snapshots)}

История `date<t`, label `[t,t+7 дней)`, окно `[t−w,t)`; дневная точность, без внутридневного порядка. Восемь train-недель → validation 9–15 августа → test 16–22 августа 2021; прошлые недели доступны поздним срезам, внутри недели признаки фиксированы.

Запрос=user×неделя с ≥1 положительным из прошлого каталога когорты. Кандидаты: все такие положительные + **99 уникальных неотмеченных** (50 равномерно, 49 по прошлой популярности^0.75). Label=1 — наличие строки, включая 0% watched_pct/total_dur; 0 — отсутствие строки, не отказ после показа. Просмотренное допускается в отрицательных: при уникальных парах прямая связь фильтрует его, что ограничивает перенос на повторное потребление.

**Sampled reranking:** будущие положительные включены принудительно, retrieval нет; оценка условна на активных запросах/сэмплированном каталоге. У методов одинаковы кандидаты/метки/история, метки и отклики других кандидатов вне признаков. Test: **{testqueries.user_id.nunique()} запросов / {len(testqueries):,} пар / {int(testqueries.label.sum()):,} положительных**; {cold} будущих пар ({cold / len(future) * 100:.2f}%) исключены как новые объекты. Cold user сохранены; неизвестная история — нули, давность 9999.

Общий train-бюджет: случайные **{runtime["training_queries"]} полных запросов / {runtime["training_rows"]:,} строк**, история не обрезана. Выбор только по validation → fit с добавлением всей validation, известной до test.

## Признаки и поиск

Manual: CatBoostClassifier/Logloss, без числовых ID; **34 признака**. Для user/item/user–item/user–genre/user–content_type — counts all/7/30 и давность; категории item, доли жанра/типа, тренды item, прошлый watched_pct user/item/user–genre. Сглаживание: знаменатель `count+1`, prior=0, не CTR. Точные формулы/исходный протокол: [DESIGN.md](DESIGN.md), [features.py](features.py). Ансамблей, retrieval и текстовых эмбеддингов нет.

Грамматика «связь×окно/затухание×агрегат»: окна **2/5/10/21/60** (counts/доли/тренды), half-life **3/10/21** (`sum(2**(-(t-date)/h))`), всего **89 признаков**. ≤2 жадных добавлений пакета, принять Δvalidation NDCG@10>0.00001; скрининг первой конфигурацией, затем общая сетка. Выбраны **{", ".join(frozen["selected_bundles"]) or "ни одного"} → {len(frozen["auto_columns"])} признаков**. События/связи/операции/target задаёт человек; возможен пропуск совместно полезных пакетов, оптимальность не доказана.

Общая сетка `(depth,iterations,lr)`: **(5,180,0.08), (6,260,0.06), (7,320,0.05)**; seed42, L2=5, threads6, без early stopping; выбор по validation NDCG@10:

{table(grid)}

## Результаты и анализ

{table(test[["model", "features", "logloss", "roc_auc", "ndcg10", "recall10", "fit_seconds", "predict_seconds"]])}

LogLoss/AUC — по строкам; NDCG/Recall@10 — равновесное среднее запросов. `DCG=sum(label/log2(rank+1))`, IDCG учитывает все положительные, ограничен top10; Recall=hits@10/**все положительные**. Равные score → меньший item_id. Популярность некалибрована (LogLoss=—); вероятности/LogLoss относятся к sampled-label, не CTR. Код: [experiment.py](experiment.py).

CI: 2000 парных bootstrap-повторов по {ci["users"]} пользователям, seed2026; условен на модели/неделе, без вариативности периодов/seed/поиска. Эквивалентность и неухудшение не доказаны (граница не задана). Лучший validation Δ **{val_gain:+.6f}** не перенёсся на test; шум/смена спроса — неразличимые здесь гипотезы.

Абляции: validation, первая конфигурация, без отдельного подбора; контроль `manual_grid_0`:

{table(ab)}

`without_indirect` удаляет user–genre/type, `without_direct` — user–item, `automatic_only` оставляет генерацию+3 категории. ΔNDCG: {ablation_deltas}. Косвенные связи отражают интересы, прямые — фильтр просмотренного. Отбор против всех 89: **{selection_gap:+.6f}**; больше признаков не гарантирует прироста. Затухание не выбрано; half-life вне сетки не проверены. Вклады условны на бюджете, не причинны/универсальны.

Top10 importance автоматической модели (коррелирующие окна делят importance; это не прирост метрики):

{table(aimp.head(10))}

Полная траектория: [validation_trials.json](results/validation_trials.json). Описательные test-срезы, вне подбора:

{table(subgroups)}

Различия групп включают активность/состав объектов, не доказывают эффект длины истории; малые группы нестабильны.

## Затраты и границы

Wall-clock без сети/установки: подготовка **{audit["feature_seconds"]:.1f} с**, {len(trials)} validation-fit **{runtime["selection_fit_seconds"]:.1f} с**, весь experiment (поиск/финальные fit/прогнозы/метрики/bootstrap) **{runtime["total_experiment_seconds"]:.1f} с**; финальные fit/predict выше, пик памяти не измерен.

{table(pd.DataFrame(budgets))}

Среда: {runtime["platform"]}, Python {runtime["python"].split()[0]}, CatBoost {runtime["catboost"]}, NumPy {runtime["numpy"]}, pandas {runtime["pandas"]}, sklearn {runtime["sklearn"]}; 6 потоков/8 логических CPU. Общая сетка не уравнивает бюджеты: дополнительные fit поиска требуют амортизации. Подготовка ≈O(S·W·(E+Q)), матрица O(Q_total·F), без нагрузочной проверки; миллиардам событий нужны инкрементальные/потоковые окна и as-of joins. Популярность оценена по когорте.

Один период/seed, малая когорта/сетка, два шага поиска; CTR, удовлетворённость, полный каталог, KION SOTA и промышленный масштаб не проверены. Продолжение: rolling holdout/новые seed → полный каталог/фиксированный retrieval → поиск связей с авторами/всеми жанрами. Увиденный test не использовать для подбора.

## Запуск и проверки

Python 3.11, из проекта:

```powershell
python -m venv .venv
.venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
python run_all.py
```

Linux/macOS: `source .venv/bin/activate`. `run_all.py`: test_protocol → `prepare.py --users 10000 --seed 42` → validate → experiment → verify_results → make_report → test_upgrades → train_active. Архивы в data позволяют работать без сети, иначе скачивается фиксированный commit. Prepare восстанавливает удалённые матрицы/queries; сохранённый events.pkl позволяет сразу использовать predict. Experiment перезаписывает исходные результаты; платформы/версии могут дать малые численные отличия.

Проверены обучение/test, 2 исходных unit-теста, все метки/кандидаты/поля полной истории/SHA-256, counts одного кандидата каждого из 10 срезов: [protocol_validation.json](results/protocol_validation.json), [test_protocol.py](test_protocol.py). Независимые метрики из прогнозов: [result_validation.json](results/result_validation.json). Новые проверки/активная модель описаны в IMPROVEMENTS.

Код: prepare — данные/кандидаты; features — point-in-time грамматика; experiment — поиск/модели/метрики; validate/test — аудит; verify_results — независимые метрики; make_report — README/график. `results/`: таблицы, параметры/поиск, модели, importance, прогнозы, bootstrap; `data/source.json`/`results/data_audit.json`: источник/commit/хеши/объёмы. Поставка сохраняет исходные ZIP, выбранную историю/ID и результаты; пересчитываемые матрицы, queries и Python-кэши удалены.
"""

    if (ROOT / "IMPROVEMENTS.md").exists():
        header, _, body = report.partition("\n\n")
        report = (
            header + "\n\n[Текущая модель и проверенные улучшения](IMPROVEMENTS.md).\n\n" + body
        )
    (ROOT / "README.md").write_text(report, encoding="utf8")
    print("README.md and results/comparison.png created")


if __name__ == "__main__":
    main()
