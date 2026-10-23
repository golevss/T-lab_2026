import json
import pathlib

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent


def table(df):
    def f(v):
        return (
            "—" if pd.isna(v) else (f"{v:.6f}" if isinstance(v, (float, np.floating)) else str(v))
        )

    return (
        "| "
        + " | ".join(df.columns)
        + " |\n| "
        + " | ".join(["---"] * len(df.columns))
        + " |\n"
        + "\n".join("| " + " | ".join(f(v) for v in row) + " |" for row in df.to_numpy())
    )


def main():
    r = ROOT / "results"

    def load(n):
        return json.loads((r / n).read_text())

    a = load("data_audit.json")
    frozen = load("frozen_selection.json")
    intervals = load("intervals.json")
    runtime = load("runtime.json")
    test = pd.read_csv(r / "test.csv")
    trials = pd.DataFrame(load("validation.json"))
    checks = load("validation_checks.json")
    graph_seconds = sum(g["seconds"] for g in a["graphs"])
    base = float(test.loc[test.model == "manual", "ndcg10"].iloc[0])
    hybrid = float(test.loc[test.model == "hybrid", "ndcg10"].iloc[0])
    ci = pd.DataFrame(
        [
            dict(model=k, delta=v["delta"], lo95=v["ci95"][0], hi95=v["ci95"][1])
            for k, v in intervals.items()
        ]
    )
    verdict = []
    for name, v in intervals.items():
        verdict.append(
            f"{name}: "
            + (
                "положительный интервал разности NDCG@10"
                if v["ci95"][0] > 0
                else "отрицательный интервал разности NDCG@10"
                if v["ci95"][1] < 0
                else "интервал разности NDCG@10 включает ноль"
            )
        )
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(9, 4.6), layout="constrained")
    vals = test.ndcg10.to_numpy()
    ax.barh(test.model, vals, color=["#396582", "#bd8b37", "#6c7c54", "#a0a4a8"])
    ax.invert_yaxis()
    ax.set_xlim(0, vals.max() * 1.23)
    ax.set_xlabel("NDCG@10 · среднее по запросам")
    ax.set_title("KION · новая когорта · test 16–22 августа 2021")
    for j, v in enumerate(vals):
        ax.text(v + 0.006, j, f"{v:.4f}", va="center")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(r / "comparison.png", dpi=160)
    plt.close(fig)
    snapshots = pd.DataFrame(a["snapshots"])
    ab = trials[trials.name.str.contains("graph16|no_categories")][
        ["name", "features", "ndcg10", "recall10", "logloss"]
    ]
    q = pd.read_pickle(ROOT / "data/queries.pkl")
    qt = q[q.split == "test"]
    e = pd.read_pickle(ROOT / "data/events.pkl")
    pred = pd.read_csv(r / "predictions.csv.gz", float_precision="round_trip")
    from metrics import metrics

    known = set(e[e.date < qt.cutoff.iloc[0]].user_id)
    warm = qt.user_id.isin(known).to_numpy()
    slices = []
    for group, mask in [("cold", ~warm), ("warm", warm)]:
        for model in ["manual", "representation", "hybrid"]:
            met = metrics(qt[mask], pred.loc[mask, model].to_numpy())
            slices.append(
                dict(
                    group=group,
                    model=model,
                    queries=met["queries"],
                    ndcg10=met["ndcg10"],
                    recall10=met["recall10"],
                )
            )
    slices = pd.DataFrame(slices)
    slices.to_csv(r / "test_slices.csv", index=False)
    category_deltas = {
        mode: float(
            trials.loc[trials.name == f"{mode}_graph16", "ndcg10"].iloc[0]
            - trials.loc[trials.name == f"{mode}_no_categories", "ndcg10"].iloc[0]
        )
        for mode in ["representation", "hybrid"]
    }
    importance = pd.read_csv(r / "hybrid_importance.csv").head(8)
    verification_note = ""
    if (ROOT / "verification/reproduction.json").exists():
        verification_note = "Полный повтор run_all: выбор идентичен, validation/test-метрики и все test-прогнозы совпали при atol=1e-12. Исходные замеры сохранены; [итог](verification/reproduction.json) и лог в [verification/](verification/). Дубли результатов удалены после сравнения."
    text = f"""# Спектральные признаки KION

## Методы: от исходной реализации к текущей

**Изначально.** Проверяли замену ручных агрегатов обучаемыми графовыми представлениями на отдельной когорте 5000 пользователей. Для каждого временного среза строили граф user–item с категориями жанра/типа; нормированную матрицу разлагали TruncatedSVD, проверяли размерности 16/32. Из item-эмбеддингов и истории пользователя получали 6 сигналов сходства, давности и холодного старта. Сравнивали один CatBoostClassifier на 34 ручных признаках (`manual`), только 6 графовых (`representation`) и их объединении из 40 (`hybrid`); дополнительно — популярность за 30 дней. Это спектральная модель с ручной компрессией истории, без GNN.

**Что показал исходный опыт.** Выбрана размерность SVD32. Test NDCG@10: manual 0.351639, representation 0.158649, hybrid 0.368044; прирост hybrid к manual +4.67%, парный 95% интервал Δ [0.007829; 0.024674]. Граф дополнил агрегаты, но не заменил их. Проверяли граф без категорий и cold/warm-срезы: категории полезны на validation; у пользователей без истории графовые сигналы одинаковы всем кандидатам, поэтому representation фактически сортирует их по item_id. Результаты не обобщаются на все графовые модели.

**Какие улучшения рассматривали.** К hybrid добавляли ItemKNN и набор rich из 73 признаков (граф32, ItemKNN, окна 2/14, затухание 3/14, все жанры); проверяли YetiRank/NDCG@10. Скрининг validation NDCG@10: baseline 0.360996, ItemKNN 0.357231, rich 0.371358, ranker 0.354294. После фиксации и переобучения на новой когорте rich дал 0.366861 → 0.377569, но семейно скорректированный интервал Δ [−0.000599; 0.021470] включает ноль: условие внедрения не выполнено. EASE рассмотрен без запуска из-за O(I²); GNN/трансформер отложены из-за бюджета и ограничений last-watch, их качество не измерялось.

**К чему пришли и что используется сейчас.** Основная модель остаётся **CatBoostClassifier hybrid: 34 ручных + 6 графовых сигналов SVD32**, `results/hybrid.cbm`; depth=7, iterations=260, learning_rate=0.05, seed=2026. `predict.py` по умолчанию использует её согласно [active_model.json](active_model.json); неподтверждённый rich не включён. В отдельном `--mode representation` внедрено переключение: пользователю без прошлой истории — прежний hybrid, остальным — исходная representation. На новой когорте NDCG@10 этого режима 0.166274 → 0.287421, семейный интервал Δ [0.100054; 0.143214]; на старом test 0.158649 → 0.282810. Это улучшение отдельного режима, не прирост основного hybrid. `--mode representation_raw` сохраняет исходную графовую модель без переключения.

Признаки и граф используют только историю до cutoff, далее ранжируются заданные кандидаты. Это пакетный sampled reranking; новая когорта проверяет другой состав пользователей того же периода, не перенос на новый период/сервис. Таблицы ниже относятся к исходному эксперименту; детали улучшений/проверок — [IMPROVEMENTS.md](IMPROVEMENTS.md). Метрики двух заданий напрямую несопоставимы из-за разных исходных когорт.

**Исходный эксперимент:** {a["users"]} пользователей / {a["events"]} строк, без пересечения с 10 000 пользователями первой работы. CatBoost: manual34 / representation6 / hybrid40.

{table(test)}

ΔNDCG@10 к manual: парный percentile bootstrap по пользователям, **95%, 2000 повторов, seed2026**:

{table(ci)}

Гибрид **{(hybrid / base - 1) * 100:+.2f}%**: граф дополняет, но не заменяет агрегаты. {"; ".join(verdict)}. Интервалы условны на периоде/моделях/seed, без вариативности поиска/когорт/недель; обобщение не гарантировано, неухудшение без заданной границы не доказано.

![Test](results/comparison.png)

## Данные и протокол

[KION](https://github.com/irsafilo/KION_DATASET), commit `{a["commit"]}`: полный `data_original.zip`, **{a["source_rows"]} строк / {a["source_users"]} пользователей**, март–август 2021; сокращённый interactions.zip не используется. SHA-256 `{a["source_sha256"]}`. Исключены `excluded_previous_users.npy` → sort(ID) → `default_rng(2026)` → 5000 без возвращения/фильтра активности. Все исходные поля истории независимо сверены, пересечение=0; с первой работой общие период/каталог, независимость только пользователей.

Last-watch пары (`event_type=watch`), повторов {a["duplicate_pairs"]}; без показов/кликов/сырой последовательности. Отсев будущих опубликованных строк проверен, но последние даты скрывают ранние действия: точное прошлое сервиса не восстановить. Метаданные постоянны без версий: первый жанр/тип — графу, плюс возрастной рейтинг — manual/hybrid; демография/тексты не используются.

{table(snapshots)}

История `date<t`, label `[t,t+7 дней)`, окно `[t−w,t)`, дневная точность без внутридневного порядка. Шесть train-недель → validation 9–15 августа → test 16–22 августа 2021; прошлые недели доступны поздним срезам, внутри недели признаки фиксированы.

Запрос=user×неделя с ≥1 положительным из прошлого каталога когорты. Все такие положительные + **99 уникальных неотмеченных** (50 равномерно, 49 по прошлой популярности^0.75). Label=1 — строка, включая 0% watched_pct/total_dur; 0 — нет строки, не отказ. Cold item исключены (таблица), cold user сохранены. Просмотренное допускается в отрицательных: при уникальных парах прямая связь фильтрует его, не моделирует повторы.

**Sampled reranking:** положительные включены принудительно, запросы активные; абсолютные метрики не описывают базу/каталог/сложный retrieval. У методов одинаковы кандидаты/метки/история, labels вне признаков. Все train-запросы → выбор только на validation → fit train+validation, известной до test.

## Признаки и подбор

Manual: CatBoostClassifier/Logloss, без числовых ID, **34 признака**. Категории item; counts all/7/30 и давность user/item/user–item/user–genre/user–type (9999 без истории); доли `count_relation/(count_user+1)`, тренды `count_window/(count_all+1)`, watched_pct user/item/user–genre `sum(fillna(pct,0))/(count+1)`. Prior=0, не CTR: [features.py](features.py).

Разреженная A каждого среза: строки user/категории, столбцы прошлых item; user–item вес1, item–genre/type по одному ребру веса0.5. Нормировка взвешенными степенями `D_row^(-1/2) A D_col^(-1/2)`. [TruncatedSVD](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html): randomized, n_iter7, seed2026, d=16/32; L2-нормированные строки V — item-эмбеддинги. Обучается спектр без labels, не GNN.

L2-профили — суммы всей истории, равномерная и `2^(-age_days/7)`. **6 сигналов:** косинусы с обоими профилями; max-косинус с последними3 item (равные даты → item_id); `log(1+sum(weights))`; евклидово расстояние профилей; cold-индикатор (остальные cold-сигналы=0). Масса/cold, связи, нормировка, half-life7 и последние3 заданы вручную. Сходства/расстояния инвариантны общему ортогональному повороту; координаты разных срезов не сравниваются. Fit строго прошлого, одна доступная история всем. graph16/32 с категориями; no_categories — диагностические16 без них. Representation=6, hybrid=6+34, общий CatBoost.

Сетка `(depth,iterations,lr)`: **(5,180,0.08), (7,260,0.05)**; L2=5, threads6, seed2026, без early stopping. Manual: обе; representation/hybrid: graph16/32 при первой, победитель validation NDCG@10 также при второй. no_categories вне выбора; бюджет альтернатив больше. Зафиксировано: `{json.dumps(frozen, ensure_ascii=False)}`.

{table(trials[["name", "features", "ndcg10", "recall10", "logloss", "fit_seconds"]])}

Абляция категорий: d16, первая конфигурация:

{table(ab)}

Δvalidation NDCG@10 категорий **{category_deltas["representation"]:+.6f}/{category_deltas["hybrid"]:+.6f}** (representation/hybrid): меняется весь спектр, не один жанр/тип; вклад условен на seed/бюджете, не причинен. Компрессия может терять counts/временные масштабы; короткая история/популярность/геометрия — гипотезы, один holdout их не различает.

## Диагностика и метрики

Test-срезы, вне подбора:

{table(slices)}

Cold: графовые сигналы одинаковы кандидатам → item_id tie-break; manual/hybrid сохраняют популярность/категории item. Warm-компрессия также может терять статистику; срезы не причинны, провал не обобщается на все графовые модели.

Top8 importance hybrid (корреляции делят importance, она не равна приросту качества; полные `*_importance.csv`):

{table(importance)}

LogLoss/AUC по строкам, NDCG/Recall@10 — равновесное среднее запросов. `DCG=sum(label/log2(rank+1))`, IDCG по всем положительным, ограничен top10; Recall=hits@10/**все положительные**. Равные score → меньший item_id. Вероятности/LogLoss относятся к sampled-label, не CTR; популярность некалибрована (LogLoss=—).

## Затраты и границы

Wall-clock без сети/установки: подготовка **{a["prepare_seconds"]:.1f} с**, включая 24 графа/признака (3×8, **{graph_seconds:.1f} с**); experiment **{runtime["total_seconds"]:.1f} с**, validation-fit **{runtime["validation_fit_seconds"]:.1f} с**; финальные fit/predict выше. Графовые fit/predict входят в подготовку, не CatBoost; пик памяти не измерен. {runtime["platform"]}, Python3.11, `requirements.txt`, threads6.

Память A: O(nnz(A)), эмбеддинги O((U+I)·d); SVD≈O(nnz(A)·d·итерации), без ортогонализации/oversampling; пары O(Q·d) плюс профили/последние3. Нагрузочных проверок нет; миллиардам событий нужны инкрементальные профили/потоковые окна/распределённый fit/кэш.

Один seed/период, малая когорта/сетка/d, фиксированная компрессия, общая эпоха первой работы; ограничения last-watch/метаданных/sampled описаны выше. CTR, удовлетворённость, полный retrieval и KION SOTA не проверены. Продолжение: rolling holdout/новые seed → полный каталог/фиксированный retrieval → GNN/обучаемая история с тем же протоколом. Увиденный test не использовать для подбора.

## Запуск и проверки

```powershell
python -m venv .venv
.venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
python run_all.py
```

Linux/macOS: `source .venv/bin/activate`. run_all: test_protocol → prepare → validate → experiment → validate → report → test_upgrades → train_active. Архив в data позволяет работать без сети, иначе скачивается фиксированный commit; exclusion-ID сохранены. Prepare восстанавливает удалённые матрицы/queries; events.pkl оставлен для predict. Experiment перезаписывает исходные результаты.

Проверены manual/graph future-invariance (с/без категорий, cold user), полная история/непересечение когорт, все метки/кандидаты, независимые test NDCG/Recall из прогнозов: [validation_checks.json](results/validation_checks.json), `{json.dumps(checks)}`.

{verification_note}

Код: prepare — данные/выборка/срезы; features — manual; graph — спектр/компрессия; experiment — подбор/модели; metrics — fit/метрики; validate/test — аудит; report — README/график. results: подбор/параметры/test/прогнозы/интервалы/модели/importance/аудит/время. Сохранены исходный ZIP, история/ID и результаты; матрицы, queries, Python-кэши и дубли повторного прогона удалены.
"""

    if (ROOT / "IMPROVEMENTS.md").exists():
        header, _, body = text.partition("\n\n")
        text = header + "\n\n[Текущая модель и проверенные улучшения](IMPROVEMENTS.md).\n\n" + body
    (ROOT / "README.md").write_text(text, encoding="utf8")
    print("Report and figure created")


if __name__ == "__main__":
    main()
