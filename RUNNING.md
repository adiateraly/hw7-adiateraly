# Как запускать

## Установка
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -c "import gymnasium; print(gymnasium.__version__)"   # запишите версию в SUBMISSION.md
```

## Порядок коммитов (по заданию)
1. Сначала закоммитить только `paper/` (фото бумажных расчётов), и запушить.
2. Потом код (`blackjack_rl/`, `tests/`, `requirements.txt`) и результаты в `outputs/`.

## Команды (запускать из корня репозитория)
| Часть | Команда | Результат |
|---|---|---|
| C: replay бумажных расчётов | `python -m blackjack_rl.paper_replay` | `outputs/paper-replay/` |
| D: fixed policy, 3 метода | `python -m blackjack_rl.fixed_policy` | `outputs/fixed-policy/` |
| D: Q-learning + оценка | `python -m blackjack_rl.q_experiment` | `outputs/q-learning/` |
| Проверка | `python -m pytest -q` | |

Файлы `paper/paper_results.json` — это значения с ваших фото, перепечатанные вручную;
replay сравнивает с ними результат программы. Проверьте, что они совпадают с листами.
