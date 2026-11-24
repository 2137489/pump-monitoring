# pump-monitoring

**Статистический мониторинг насоса с частотным приводом по остаткам регрессии**
*Regime-normalized residuals versus raw-signal thresholds for statistical monitoring of variable-speed centrifugal pumps*

Авторы / Authors: И.С. Гатауллин, В.Ю. Арьков (I. S. Gataullin, V. Yu. Arkov), Уфимский университет науки и технологий (Ufa University of Science and Technology).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/2137489/pump-monitoring/blob/main/pump_monitoring.ipynb)

## О проекте

Открытая модель центробежного насоса с частотным приводом (законы подобия, характеристики насоса и сети, КПД, мощность, вибрация), три постепенно развивающиеся неисправности (износ рабочего колеса, засор, деградация подшипника) и сравнение пяти детекторов аномалий методом Монте-Карло (1600 прогонов) при одинаковой откалиброванной вероятности ложной тревоги 10 % за 20 суток:

- по исходным сигналам: 3σ, границы Тьюки, квантили KDE;
- по остаткам регрессии на параметры режима: 3σ и EWMA.

Главный результат: правила по исходным сигналам практически не видят износ и засор, а EWMA по остаткам регрессии обнаруживает их с медианной задержкой 3,1 и 1,1 сут.

## Запуск

Проще всего открыть `pump_monitoring.ipynb` в Google Colab (кнопка выше): блокнот сам создаёт все файлы и воспроизводит все таблицы и рисунки.

Локально:

```bash
pip install -r requirements.txt
cd src
python run_experiments.py results.json        # Монте-Карло, около 3 минут
python make_figures.py fig results.json       # рисунки (PDF, англ.) и fig/summary.json
python make_figures_ru.py fig_ru results.json # рисунки (PNG, рус.)
mkdir -p tables && python make_tables.py fig/summary.json tables   # таблицы LaTeX
```

Начальное значение генератора случайных чисел — 2026. С версиями библиотек из статьи (NumPy 2.5, SciPy 1.18, scikit-learn 1.9) все числа воспроизводятся точно. В других окружениях, например в Google Colab, результаты детекторов по остаткам могут немного отличаться (пороги примерно на 1 %, задержки до 0,4 сут) из-за различий в вычислениях с плавающей точкой; выводы при этом не меняются.

## Файлы

| Файл | Назначение |
|---|---|
| `src/pumpsim.py` | модель насоса, режим работы, шум датчиков, неисправности, пять детекторов |
| `src/run_experiments.py` | калибровка порогов и оценка детекторов (Монте-Карло) |
| `src/make_figures.py`, `src/make_figures_ru.py` | рисунки и сводные числа (англ. / рус.) |
| `src/make_tables.py` | таблицы статьи |
| `pump_monitoring.ipynb` | блокнот Google Colab — всё исследование целиком |
| `paper/preprint_en.pdf` | препринт статьи на английском языке |

## English

Open simulator of a variable-speed centrifugal pump with three gradually developing faults and a Monte Carlo comparison (1,600 runs) of five anomaly detectors at a common calibrated false-alarm probability of 10% per 20 days. Raw-signal rules (3σ, Tukey fences, KDE quantiles) are effectively blind to impeller wear and clogging; an EWMA of regression residuals on the operating regime detects them with median delays of 3.1 and 1.1 days. Run the Colab notebook or the commands above (Python 3, seed 2026). Numbers are exactly reproducible with the library versions listed in the paper; in other environments (e.g. Google Colab) the residual-detector results may differ slightly because of floating-point differences, without changing the conclusions.

## Лицензия / License

MIT — см. `LICENSE`.
