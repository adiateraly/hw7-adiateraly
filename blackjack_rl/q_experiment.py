"""Part D (2): Q-learning (epsilon 0.2 and 0) vs the hit-below-20 baseline.

Run:  python -m blackjack_rl.q_experiment
Output: outputs/q-learning/
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .common import ACTION_NAME, ALPHA, GAMMA, HIT, STICK, Transition, END, baseline_action, write_versions
from .envs import make_env, play_game
from .learners import QLearning
from .tables import DEALER, SUMS, grid_csv_rows, grid_md

ROOT = Path(__file__).resolve().parent.parent
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8985"
SEED_COLOR = {11: BLUE, 22: ORANGE, 33: AQUA}


def eps_name(e) -> str:
    return f"eps_{e:g}"


def train(seed: int, epsilon: float, n_games: int):
    env = make_env()
    rng = np.random.default_rng(seed)          # action generator (exploration + tie-breaking)
    agent = QLearning(ALPHA, GAMMA)
    rewards, cutoffs = [], []
    for g in range(1, n_games + 1):
        steps, cut = play_game(env, lambda s: agent.act(s, epsilon, rng),
                               first_seed=seed if g == 1 else None)
        for st in steps:   # online updates happen as the transitions occur
            nxt = END if st["terminal"] else st["next_obs"]
            agent.update(Transition(st["state"], st["action"], st["reward"], nxt))
        if cut:
            cutoffs.append(g)          # not a win/loss/draw; excluded from reward statistics
            rewards.append(None)
        else:
            rewards.append(steps[-1]["reward"])
    env.close()
    return agent, rewards, cutoffs


def evaluate(choose, eval_seed: int, n_games: int):
    """Fresh env seeded at the first reset; no learning, no exploration."""
    env = make_env()
    w = l = d = cut = 0
    total = 0.0
    for g in range(1, n_games + 1):
        steps, c = play_game(env, choose, first_seed=eval_seed if g == 1 else None)
        if c:
            cut += 1
            continue
        r = steps[-1]["reward"]
        total += r
        w += r > 0
        l += r < 0
        d += r == 0
    env.close()
    n = w + l + d
    return {"wins": int(w), "losses": int(l), "draws": int(d), "cutoffs": cut,
            "mean_reward": total / n if n else float("nan")}


def block_means(rewards, size=1000):
    out = []
    for i in range(0, len(rewards), size):
        blk = [r for r in rewards[i:i + size] if r is not None]
        out.append(float(np.mean(blk)) if blk else float("nan"))
    return out


def policy_cell(agent):
    def cell(s, d, a):
        key = (s, d, a)
        if key not in agent.Q:
            return None
        best = agent.best_actions(key)
        return "tie" if len(best) > 1 else ACTION_NAME[best[0]]
    return cell


def save_run(agent, rewards, cutoffs, seed, eps, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    cell = policy_cell(agent)
    with open(out / "q_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["usable_ace", "player_sum", "dealer_showing", "Q_stick", "Q_hit",
                    "visits_stick", "visits_hit", "greedy"])
        for (s, d, a) in sorted(agent.Q, key=lambda k: (k[2], k[0], k[1])):
            q, n = agent.Q[(s, d, a)], agent.N[(s, d, a)]
            w.writerow([a, s, d, repr(q[STICK]), repr(q[HIT]), n[STICK], n[HIT], cell(s, d, a)])
    for ace in (0, 1):
        with open(out / f"greedy_policy_ace{ace}.csv", "w", newline="") as f:
            csv.writer(f).writerows(grid_csv_rows(cell, ace))
        with open(out / f"visit_counts_ace{ace}.csv", "w", newline="") as f:
            csv.writer(f).writerows(grid_csv_rows(
                lambda s, d, a: (None if (s, d, a) not in agent.N else
                                 f"{agent.N[(s, d, a)][STICK]}/{agent.N[(s, d, a)][HIT]}"), ace))
    md = [f"# Seed {seed}, epsilon {eps:g}\n",
          grid_md("Greedy action", cell),
          "Visits are shown as stick/hit.\n",
          grid_md("Visits stick/hit", lambda s, d, a: (None if (s, d, a) not in agent.N else
                                                       f"{agent.N[(s, d, a)][STICK]}/{agent.N[(s, d, a)][HIT]}"))]
    (out / "tables.md").write_text("\n".join(md))
    with open(out / "training_games.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["game", "reward"])
        for i, r in enumerate(rewards, 1):
            w.writerow([i, "cutoff" if r is None else r])
    blocks = block_means(rewards)
    with open(out / "training_blocks.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["block_end_game", "mean_reward"])
        for i, b in enumerate(blocks, 1):
            w.writerow([i * 1000, b])
    (out / "training_cutoffs.json").write_text(json.dumps(cutoffs) + "\n")
    return blocks


def plot_policy(agent, seed, eps, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = {"stick": BLUE, "hit": ORANGE, "tie": "#d9d8d3", None: "#ffffff"}
    letters = {"stick": "S", "hit": "H", "tie": "=", None: ""}
    cell = policy_cell(agent)
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
    for ax, ace in zip(axes, (0, 1)):
        for yi, s in enumerate(SUMS):
            for xi, d in enumerate(DEALER):
                c = cell(s, d, ace)
                ax.add_patch(plt.Rectangle((xi, yi), 1, 1, facecolor=cols[c], edgecolor="white", lw=1.5))
                ax.text(xi + .5, yi + .5, letters[c], ha="center", va="center", fontsize=8,
                        color="white" if c in ("stick", "hit") else "#3a3936")
        ax.set_xlim(0, len(DEALER)); ax.set_ylim(0, len(SUMS))
        ax.set_xticks([i + .5 for i in range(len(DEALER))], [str(d) for d in DEALER])
        ax.set_yticks([i + .5 for i in range(len(SUMS))], [str(s) for s in SUMS])
        ax.set_xlabel("dealer showing (1 = ace)")
        ax.set_title("usable ace: " + ("yes" if ace else "no"), fontsize=11)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_ylabel("player sum")
    fig.suptitle(f"Greedy policy after training - seed {seed}, epsilon {eps:g}\n"
                 "S = stick, H = hit, = tie, blank = never visited", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_training(blocks, evals, path: Path, epsilons):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, len(epsilons), figsize=(5.6 * len(epsilons), 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, eps in zip(axes, epsilons):
        nb = len(next(iter(blocks.values())))
        for idx, seed in enumerate(sorted({s for s, _ in blocks})):
            col = SEED_COLOR.get(seed, GRAY)
            b = blocks[(seed, eps)]
            ax.plot(range(1, nb + 1), b, color=col, lw=2)
            ev = evals[(seed, eps)]
            ax.plot([nb + 2.5 + idx], [ev["learned"]["mean_reward"]], "o", color=col, ms=8, mec="white", mew=1.5)
            ax.plot([nb + 7 + idx], [ev["baseline"]["mean_reward"]], "s", color=col, ms=8, mfc="white", mew=2)
        ax.axvspan(nb + 1.2, nb + 10.8, color="#f0efec", zorder=0)
        ax.text(nb + 6, 0.97, "evaluation", ha="center", va="top", fontsize=8,
                color="#52514e", transform=ax.get_xaxis_transform())
        ax.set_xticks([0, 10, 20, 30, 40, 50], ["0", "10k", "20k", "30k", "40k", "50k"])
        ax.set_xlabel("training games (1,000-game blocks)")
        ax.set_title(f"epsilon = {eps:g}", fontsize=11)
        ax.grid(axis="y", color="#e6e5e1", lw=.8)
        ax.axhline(0, color="#bbbab4", lw=.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("mean reward per game")
    handles = [Line2D([0], [0], color=SEED_COLOR[s], lw=2, label=f"seed {s}") for s in sorted(SEED_COLOR)
               if any(k[0] == s for k in blocks)]
    handles += [Line2D([0], [0], color=GRAY, lw=2, label="training (exploring/learning)"),
                Line2D([0], [0], marker="o", color=GRAY, lw=0, label="evaluation: frozen Q (greedy)"),
                Line2D([0], [0], marker="s", color=GRAY, mfc="white", lw=0, label="evaluation: hit-below-20")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=50_000)
    ap.add_argument("--eval-games", type=int, default=10_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    ap.add_argument("--epsilons", type=float, nargs="+", default=[0.2, 0.0])
    ap.add_argument("--out", default=str(ROOT / "outputs" / "q-learning"))
    a = ap.parse_args()
    out = Path(a.out)
    print(write_versions(out / "versions.json"))

    blocks, evals, rows = {}, {}, []
    for seed in a.seeds:
        for eps in a.epsilons:
            agent, rewards, cutoffs = train(seed, eps, a.games)
            run_dir = out / f"seed_{seed}" / eps_name(eps)
            blocks[(seed, eps)] = save_run(agent, rewards, cutoffs, seed, eps, run_dir)
            plot_policy(agent, seed, eps, run_dir / "greedy_policy.png")
            eval_seed = 1000 + seed
            frozen = QLearning(ALPHA, GAMMA)
            frozen.Q = {k: list(v) for k, v in agent.Q.items()}       # frozen copy
            erng = np.random.default_rng(eval_seed)                  # only for random tie-breaking
            learned = evaluate(lambda s: frozen.act(s, 0.0, erng), eval_seed, a.eval_games)
            baseline = evaluate(lambda s: baseline_action(s[0]), eval_seed, a.eval_games)
            evals[(seed, eps)] = {"learned": learned, "baseline": baseline}
            (run_dir / "evaluation.json").write_text(json.dumps(
                {"train_seed": seed, "epsilon": eps, "eval_seed": eval_seed, "train_cutoffs": cutoffs,
                 "learned": learned, "baseline": baseline}, indent=2) + "\n")
            rows.append([seed, f"Q-learning, eps={eps:g}", eval_seed, *learned.values()])
            print(f"seed {seed} eps {eps:g}: Q mean {learned['mean_reward']:+.4f} "
                  f"vs baseline {baseline['mean_reward']:+.4f}  (train cutoffs {len(cutoffs)})")
        b = evals[(seed, a.epsilons[0])]["baseline"]
        rows.append([seed, "baseline hit-below-20", 1000 + seed, *b.values()])
    rows.sort(key=lambda r: (r[0], 0 if r[1].startswith("baseline") else 1, r[1]))
    head = ["training_seed", "policy", "evaluation_seed", "wins", "losses", "draws", "cutoffs", "mean_reward"]
    with open(out / "evaluation_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(head)
        w.writerows(rows)
    with open(out / "evaluation_summary.md", "w") as f:
        f.write("| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n")
        for r in rows:
            f.write("| " + " | ".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in r) + " |\n")
    plot_training(blocks, evals, out / "training_curves.png", a.epsilons)
    print("done ->", out)


if __name__ == "__main__":
    main()
