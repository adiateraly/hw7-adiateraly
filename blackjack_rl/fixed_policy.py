"""Part D (1): evaluate the fixed hit-below-20 policy with three methods on the SAME saved games.

Run:  python -m blackjack_rl.fixed_policy
Output: outputs/fixed-policy/
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from .common import END, Transition, baseline_action, write_versions
from .envs import make_env, play_game
from .learners import ModelBased, MonteCarlo, TD0
from .tables import grid_md

ROOT = Path(__file__).resolve().parent.parent
LOG_HEADER = ["seed", "game", "step", "obs_sum", "obs_dealer", "obs_ace", "action", "reward",
              "next_sum", "next_dealer", "next_ace", "terminal", "cutoff"]


def collect(seed: int, n_games: int, path: Path) -> None:
    """Play n_games with the baseline policy and write every step to a CSV log."""
    env = make_env()
    policy = lambda s: baseline_action(s[0])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(LOG_HEADER)
        for g in range(1, n_games + 1):
            steps, cutoff = play_game(env, policy, first_seed=seed if g == 1 else None)
            for i, st in enumerate(steps, 1):
                s, n = st["state"], st["next_obs"]
                w.writerow([seed, g, i, *s, st["action"], st["reward"], *n,
                            int(st["terminal"]), int(cutoff and i == len(steps))])
    env.close()


def load_episodes(path: Path):
    """Read the saved log. Terminal transitions become END. Cut-off games are returned separately."""
    games, cut = defaultdict(list), set()
    with open(path) as f:
        for r in csv.DictReader(f):
            g = int(r["game"])
            s = (int(r["obs_sum"]), int(r["obs_dealer"]), int(r["obs_ace"]))
            nxt = END if r["terminal"] == "1" else (int(r["next_sum"]), int(r["next_dealer"]), int(r["next_ace"]))
            games[g].append(Transition(s, int(r["action"]), float(r["reward"]), nxt))
            if r["cutoff"] == "1":
                cut.add(g)
    episodes = [games[g] for g in sorted(games) if g not in cut]
    return episodes, sorted(cut)


def run_seed(seed: int, n_games: int, checkpoints, out: Path, tol: float, max_sweeps: int):
    sd = out / f"seed_{seed}"
    sd.mkdir(parents=True, exist_ok=True)
    log = sd / "games.csv"
    collect(seed, n_games, log)
    episodes, cutoffs = load_episodes(log)
    policy = lambda s: baseline_action(s[0])

    mb, mc, td = ModelBased(), MonteCarlo(), TD0()
    visits = defaultdict(int)
    est_rows, conv_rows, md = [], [], []
    snaps = {}
    cp = set(checkpoints)
    for i, ep in enumerate(episodes, 1):
        for t in ep:
            mb.observe(t)
            td.update(t)
            visits[t.state] += 1
        mc.observe_episode(ep)
        if i in cp:
            res = mb.evaluate(policy, tol=tol, max_sweeps=max_sweeps)
            missing_succ = sorted({s2 for (s, a, s2) in mb.n_sas if s2 != END and s2 in res.unobserved})
            conv_rows.append([seed, i, res.sweeps, res.converged, res.last_delta, len(res.values),
                              len(res.affected), len(missing_succ)])
            snaps[i] = {"model-based": dict(res.values),
                        "monte-carlo": {s: mc.value(s) for s in mc.counts},
                        "td0": {s: td.value(s) for s in td.counts},
                        "visits": dict(visits), "affected": res.affected}
            for method, vals in snaps[i].items():
                if method in ("visits", "affected"):
                    continue
                for s, v in sorted(vals.items()):
                    est_rows.append([seed, i, method, s[2], s[0], s[1], repr(v), visits[s],
                                     int(method == "model-based" and s in res.affected)])
            md.append(f"## After {i} games (seed {seed})\n")
            md.append(f"Model-based: {res.sweeps} sweeps, converged={res.converged}, last change {res.last_delta:.2e}.\n")
            md.append("Cells: estimate (visits). `-` = never visited, no estimate. "
                      "Unobserved pairs are left empty, not filled with zero.\n")
            for method in ("model-based", "monte-carlo", "td0"):
                vals = snaps[i][method]
                md.append(grid_md(method, lambda s, d, a, vals=vals: (
                    f"{vals[(s, d, a)]:+.3f} ({visits[(s, d, a)]})" if (s, d, a) in vals else None)))
    with open(sd / "estimates.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "games", "method", "usable_ace", "player_sum", "dealer_showing",
                    "value", "visits", "affected_by_missing_successor"])
        w.writerows(est_rows)
    (sd / "tables.md").write_text("\n".join(md))
    return snaps, conv_rows, cutoffs


def compare_states(seed, snaps, final, n_each=5):
    """Frequently vs rarely visited states: estimates of all three methods at every checkpoint."""
    v = snaps[final]["visits"]
    order = sorted(v, key=lambda s: (-v[s], s))
    chosen = [("frequent", s) for s in order[:n_each]]
    rare = sorted(v, key=lambda s: (v[s], s))[:n_each]
    chosen += [("rare", s) for s in rare]
    rows = []
    for kind, s in chosen:
        for cp in sorted(snaps):
            row = [seed, kind, s[2], s[0], s[1], cp, snaps[cp]["visits"].get(s, 0)]
            for m in ("model-based", "monte-carlo", "td0"):
                x = snaps[cp][m].get(s)
                row.append("" if x is None else repr(x))
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=10_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    ap.add_argument("--checkpoints", type=int, nargs="+", default=[100, 1000, 10_000])
    ap.add_argument("--out", default=str(ROOT / "outputs" / "fixed-policy"))
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--max-sweeps", type=int, default=10_000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(write_versions(out / "versions.json"))

    conv, cmp_rows, cut_report = [], [], {}
    for seed in a.seeds:
        snaps, conv_rows, cutoffs = run_seed(seed, a.games, [c for c in a.checkpoints if c <= a.games],
                                             out, a.tol, a.max_sweeps)
        conv += conv_rows
        cut_report[seed] = cutoffs
        cmp_rows += compare_states(seed, snaps, max(snaps))
        for r in conv_rows:
            print(f"seed {r[0]} @ {r[1]:>6} games: model-based sweeps={r[2]} converged={r[3]} "
                  f"affected={r[6]} successors-without-policy-pair={r[7]}")
    with open(out / "model_based_convergence.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "games", "sweeps", "converged", "last_max_change", "n_states_valued",
                    "n_affected_by_missing_successor", "n_successors_without_policy_pair"])
        w.writerows(conv)
    with open(out / "state_comparison.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "group", "usable_ace", "player_sum", "dealer_showing", "games",
                    "visits_so_far", "model_based", "monte_carlo", "td0"])
        w.writerows(cmp_rows)
    (out / "cutoffs.json").write_text(json.dumps(cut_report, indent=2) + "\n")
    print("done ->", out)


if __name__ == "__main__":
    main()
