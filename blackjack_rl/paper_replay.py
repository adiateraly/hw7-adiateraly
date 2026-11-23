"""Part C: replay the four supplied histories and compare with the paper work.

Run:  python -m blackjack_rl.paper_replay
Output: outputs/paper-replay/
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

from .common import ACTION_NAME, END, HIT, STICK, ALPHA, GAMMA, Transition, baseline_action
from .learners import MonteCarlo, ModelBased, QLearning, TD0

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "paper-replay"
PAPER = ROOT / "paper" / "paper_results.json"
TOL_MATCH = 1e-9


class Tee:
    def __init__(self, path):
        self.f = open(path, "w")

    def write(self, s):
        sys.__stdout__.write(s)
        self.f.write(s)

    def flush(self):
        sys.__stdout__.flush()
        self.f.flush()


def load_states():
    with open(DATA / "states.csv") as f:
        return {r["label"]: int(r["player_sum"]) for r in csv.DictReader(f)}


def load_games():
    """Returns {game: [Transition]} exactly as supplied, in file order."""
    games = defaultdict(list)
    with open(DATA / "paper-games.csv") as f:
        for r in csv.DictReader(f):
            nxt = END if r["next_state"] == END else r["next_state"]
            games[int(r["game"])].append(
                Transition(r["state"], int(r["action"]), float(r["reward"]), nxt))
    return dict(sorted(games.items()))


def p(x):  # short printing, full precision kept internally
    return f"{x:.6f}"


def run(write_files: bool = True) -> dict:
    sums = load_states()
    games = load_games()
    policy = lambda s: baseline_action(sums[s])
    labels = sorted(sums)
    prog = {}  # (method, item, where) -> float
    rows = defaultdict(list)  # csv rows per output file

    print(f"alpha={ALPHA}  gamma={GAMMA}  policy: hit below 20, stick at 20/21")
    print("policy on A-E:", {s: ACTION_NAME[policy(s)] for s in labels})

    # ---------------- 1. model-based ----------------
    print("\n=== 1. MODEL-BASED ===")
    mb = ModelBased()
    for g in games.values():
        for t in g:
            mb.observe(t)
    for (s, a), n in sorted(mb.n_sa.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        print(f"N({s},{ACTION_NAME[a]}) = {n}")
        prog[("model-based", f"N({s},{ACTION_NAME[a]})", "counts")] = float(n)
        rows["model.csv"].append(["N", s, ACTION_NAME[a], "", n, ""])
    for (s, a, s2) in mb.triples():
        T, R = mb.T(s, a, s2), mb.R(s, a, s2)
        print(f"T({s},{ACTION_NAME[a]},{s2}) = {mb.n_sas[(s, a, s2)]}/{mb.n_sa[(s, a)]} = {p(T)}   "
              f"R = {mb.r_sum[(s, a, s2)]:+.0f}/{mb.n_sas[(s, a, s2)]} = {p(R)}")
        prog[("model-based", f"T({s},{ACTION_NAME[a]},{s2})", "model")] = T
        prog[("model-based", f"R({s},{ACTION_NAME[a]},{s2})", "model")] = R
        rows["model.csv"].append(["T,R", s, ACTION_NAME[a], s2, repr(T), repr(R)])
    res = mb.evaluate(policy, sweeps=2)
    print("Unobserved (state, policy action) pairs: ", sorted(res.unobserved) or "none")
    for k in (1, 2):
        print(f"V{k}: " + "  ".join(f"{s}={p(res.history[k][s])}" for s in labels))
        for s in labels:
            prog[("model-based", f"V({s})", f"sweep {k}")] = res.history[k][s]
            rows["value_sweeps.csv"].append([k, s, repr(res.history[k][s])])

    # ---------------- 2. Monte Carlo ----------------
    print("\n=== 2. MONTE CARLO (every visit, one pass) ===")
    mc = MonteCarlo(keep_returns=True)
    for gid, g in games.items():
        for (s, G) in mc.observe_episode(g):
            print(f"game {gid}: visit {s}  return {G:+.0f}")
            rows["mc_returns.csv"].append([gid, s, repr(G)])
    for s in labels:
        avg = mc.value(s)
        print(f"{s}: returns {[int(x) for x in mc.returns[s]]}  count {mc.counts[s]}  average {p(avg)}")
        prog[("monte-carlo", f"average({s})", "one pass")] = avg
        prog[("monte-carlo", f"count({s})", "one pass")] = float(mc.counts[s])
        rows["mc_summary.csv"].append([s, mc.counts[s], repr(avg)])

    # ---------------- 3. TD(0) ----------------
    print("\n=== 3. TD(0)  V_new = V + 0.1 (r + V' - V) ===")
    td = TD0()
    for ps in (1, 2):
        print(f"-- pass {ps}")
        for gid, g in games.items():
            for step, t in enumerate(g, 1):
                u = td.update(t)
                print(f"g{gid} s{step}: V({t.state}) old={p(u.old)}  r={u.reward:+.0f}  "
                      f"V'({t.next_state})={p(u.succ)}  target={p(u.target)}  new={p(u.new)}")
                prog[("td0", f"V({t.state})", f"pass {ps} game {gid} step {step}")] = u.new
                rows["td_updates.csv"].append([ps, gid, step, t.state, repr(u.old), u.reward,
                                               t.next_state, repr(u.succ), repr(u.target), repr(u.new)])
    print("Final V: " + "  ".join(f"{s}={p(td.V[s])}" for s in labels))
    for s in labels:
        prog[("td0", f"V({s})", "final")] = td.V[s]

    # ---------------- 4. Q-learning ----------------
    print("\n=== 4. Q-LEARNING  Q_new = Q + 0.1 (r + max Q' - Q), max over hit AND stick ===")
    ql = QLearning()
    for ps in (1, 2):
        print(f"-- pass {ps}")
        for gid, g in games.items():
            for step, t in enumerate(g, 1):
                u = ql.update(t)
                name = f"Q({t.state},{ACTION_NAME[t.action]})"
                print(f"g{gid} s{step}: {name} old={p(u.old)}  r={u.reward:+.0f}  "
                      f"max Q'({t.next_state})={p(u.succ)}  target={p(u.target)}  new={p(u.new)}")
                prog[("q-learning", name, f"pass {ps} game {gid} step {step}")] = u.new
                rows["q_updates.csv"].append([ps, gid, step, t.state, ACTION_NAME[t.action], repr(u.old),
                                              u.reward, t.next_state, repr(u.succ), repr(u.target), repr(u.new)])
    print("Final Q and greedy action:")
    for s in labels:
        q = ql.q(s)
        best = ql.best_actions(s)
        greedy = "tie" if len(best) > 1 else ACTION_NAME[best[0]]
        print(f"{s}: Q(stick)={p(q[STICK])}  Q(hit)={p(q[HIT])}  greedy={greedy}")
        rows["q_final.csv"].append([s, repr(q[STICK]), repr(q[HIT]), greedy])

    # ---------------- paper vs program ----------------
    print("\n=== PAPER vs PROGRAM ===")
    comp = []
    for e in json.loads(PAPER.read_text()):
        paper = float(Fraction(e["paper"]))
        key = (e["method"], e["item"], e["where"])
        if key not in prog:
            raise KeyError(f"no program value for {key}")
        got = prog[key]
        diff = got - paper
        shown = 0.0 if abs(diff) < 1e-12 else diff
        comp.append([e["method"], e["item"], e["where"], e["paper"], repr(got), f"{shown:.3g}",
                     "match" if abs(diff) < TOL_MATCH else "DIFFERS"])
    header = ["method", "state / state-action", "sweep / pass / update",
              "paper", "program", "difference", "status"]
    for r in comp:
        print(" | ".join(str(x) for x in r))
    bad = [r for r in comp if r[-1] != "match"]
    print(f"\n{len(comp) - len(bad)}/{len(comp)} entries match; {len(bad)} differ.")

    if write_files:
        OUT.mkdir(parents=True, exist_ok=True)
        heads = {
            "model.csv": ["kind", "state", "action", "next_state", "value1", "value2"],
            "value_sweeps.csv": ["sweep", "state", "V"],
            "mc_returns.csv": ["game", "state", "return"],
            "mc_summary.csv": ["state", "count", "average"],
            "td_updates.csv": ["pass", "game", "step", "state", "old", "reward", "next_state",
                               "V_next", "target", "new"],
            "q_updates.csv": ["pass", "game", "step", "state", "action", "old", "reward", "next_state",
                              "max_Q_next", "target", "new"],
            "q_final.csv": ["state", "Q_stick", "Q_hit", "greedy"],
        }
        for name, body in rows.items():
            with open(OUT / name, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(heads[name])
                w.writerows(body)
        with open(OUT / "paper_vs_program.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(comp)
        with open(OUT / "paper_vs_program.md", "w") as f:
            f.write("| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n")
            for r in comp:
                f.write("| " + " | ".join(str(x) for x in r) + " |\n")
    return {"prog": prog, "comparison": comp}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(OUT / "replay_output.txt")
    try:
        run()
    finally:
        sys.stdout.f.close()
        sys.stdout = sys.__stdout__


if __name__ == "__main__":
    main()
