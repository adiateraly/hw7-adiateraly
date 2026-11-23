"""The four learning methods, implemented from scratch (no RL libraries).

States are any hashable objects: tuples (sum, dealer, ace) in simulations,
string labels "A".."E" in the paper replay.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable, Hashable, NamedTuple

from .common import ACTIONS, END, GAMMA, ALPHA, Transition


# --------------------------------------------------------------------------
# 1. Model-based learning
# --------------------------------------------------------------------------
class EvalResult(NamedTuple):
    history: list            # history[k] = {state: V_k(state)}; history[0] is V0 = 0
    values: dict             # final V
    sweeps: int              # sweeps actually performed
    converged: bool
    last_delta: float
    unobserved: set          # states seen, but (state, policy(state)) never observed
    affected: set            # states whose value depends on missing successor information


class ModelBased:
    """Counts outcomes, estimates T and R, then evaluates a fixed policy on the learned model."""

    def __init__(self, gamma: float = GAMMA):
        self.gamma = gamma
        self.n_sa = defaultdict(int)
        self.n_sas = defaultdict(int)
        self.r_sum = defaultdict(float)
        self.states = set()

    def observe(self, t: Transition) -> None:
        self.n_sa[(t.state, t.action)] += 1
        self.n_sas[(t.state, t.action, t.next_state)] += 1
        self.r_sum[(t.state, t.action, t.next_state)] += t.reward
        self.states.add(t.state)
        if t.next_state != END:
            self.states.add(t.next_state)

    def T(self, s, a, s2) -> float:
        return self.n_sas[(s, a, s2)] / self.n_sa[(s, a)]

    def R(self, s, a, s2) -> float:
        return self.r_sum[(s, a, s2)] / self.n_sas[(s, a, s2)]

    def triples(self):
        return sorted(self.n_sas, key=lambda k: (str(k[0]), k[1], str(k[2])))

    def evaluate(self, policy: Callable[[Hashable], int], sweeps: int | None = None,
                 tol: float = 1e-6, max_sweeps: int = 10_000) -> EvalResult:
        """Synchronous policy evaluation from V0 = 0 on the learned model.

        sweeps=k runs exactly k sweeps (paper replay). Otherwise stop when the
        largest change is below tol, or after max_sweeps (then converged=False).
        Only the policy's action is used (no maximisation).
        """
        observed = {s for s in self.states if self.n_sa.get((s, policy(s)), 0) > 0}
        unobserved = self.states - observed
        # successors of observed pairs that have no observed policy pair themselves
        direct = set()
        succ = defaultdict(set)
        for (s, a, s2) in self.n_sas:
            if a != policy(s) or s2 == END:
                continue
            succ[s].add(s2)
            if s2 not in observed:
                direct.add(s)
        affected = set(direct)
        changed = True
        while changed:
            changed = False
            for s, nxt in succ.items():
                if s not in affected and nxt & affected:
                    affected.add(s)
                    changed = True

        V = {s: 0.0 for s in observed}
        history = [dict(V)]
        limit = sweeps if sweeps is not None else max_sweeps
        converged, delta, k = False, float("inf"), 0
        while k < limit:
            newV = {s: 0.0 for s in observed}   # synchronous: only V_k is read below
            for (s, a, s2), n in self.n_sas.items():
                if s in observed and a == policy(s):
                    p = n / self.n_sa[(s, a)]
                    r = self.r_sum[(s, a, s2)] / n
                    nxt = 0.0 if s2 == END else V.get(s2, 0.0)  # unobserved successor: 0, flagged
                    newV[s] += p * (r + self.gamma * nxt)
            delta = max((abs(newV[s] - V[s]) for s in observed), default=0.0)
            V = newV
            history.append(dict(V))
            k += 1
            if sweeps is None and delta < tol:
                converged = True
                break
        if sweeps is not None:
            converged = delta < tol
        return EvalResult(history, V, k, converged, delta, unobserved, affected)


# --------------------------------------------------------------------------
# 2. Direct evaluation (every-visit Monte Carlo)
# --------------------------------------------------------------------------
class MonteCarlo:
    def __init__(self, gamma: float = GAMMA, keep_returns: bool = False):
        self.gamma = gamma
        self.keep = keep_returns
        self.sums = defaultdict(float)
        self.counts = defaultdict(int)
        self.returns = defaultdict(list)

    def observe_episode(self, episode: list[Transition]):
        """Process one complete game. Returns [(state, return)] in forward order."""
        G, out = 0.0, []
        for t in reversed(episode):
            G = t.reward + self.gamma * G
            out.append((t.state, G))
        out.reverse()
        for s, g in out:
            self.sums[s] += g
            self.counts[s] += 1
            if self.keep:
                self.returns[s].append(g)
        return out

    def value(self, s):
        return self.sums[s] / self.counts[s] if self.counts.get(s) else None


# --------------------------------------------------------------------------
# 3. TD(0) policy evaluation
# --------------------------------------------------------------------------
class UpdateRecord(NamedTuple):
    key: Hashable
    old: float
    reward: float
    succ: float
    target: float
    new: float


class TD0:
    def __init__(self, alpha: float = ALPHA, gamma: float = GAMMA):
        self.alpha, self.gamma = alpha, gamma
        self.V = defaultdict(float)
        self.counts = defaultdict(int)

    def update(self, t: Transition) -> UpdateRecord:
        old = self.V[t.state]
        succ = 0.0 if t.next_state == END else self.V[t.next_state]
        target = t.reward + self.gamma * succ
        new = old + self.alpha * (target - old)
        self.V[t.state] = new
        self.counts[t.state] += 1
        return UpdateRecord(t.state, old, t.reward, succ, target, new)

    def value(self, s):
        return self.V[s] if self.counts.get(s) else None


# --------------------------------------------------------------------------
# 4. Q-learning (off-policy control)
# --------------------------------------------------------------------------
class QLearning:
    def __init__(self, alpha: float = ALPHA, gamma: float = GAMMA):
        self.alpha, self.gamma = alpha, gamma
        self.Q = {}   # state -> [Q(stick), Q(hit)]
        self.N = {}   # state -> [visits(stick), visits(hit)]

    def q(self, s):
        return self.Q.get(s, [0.0, 0.0])

    def update(self, t: Transition) -> UpdateRecord:
        q = self.Q.setdefault(t.state, [0.0, 0.0])
        n = self.N.setdefault(t.state, [0, 0])
        old = q[t.action]
        # maximise over BOTH actions, including untried ones (initialised to 0)
        succ = 0.0 if t.next_state == END else max(self.q(t.next_state))
        target = t.reward + self.gamma * succ
        new = old + self.alpha * (target - old)
        q[t.action] = new
        n[t.action] += 1
        return UpdateRecord((t.state, t.action), old, t.reward, succ, target, new)

    def best_actions(self, s) -> list[int]:
        q = self.q(s)
        m = max(q)
        return [a for a in ACTIONS if q[a] == m]

    def act(self, s, epsilon: float, rng) -> int:
        """Epsilon-greedy; exploration is uniform over both actions; greedy ties broken randomly."""
        if epsilon > 0 and rng.random() < epsilon:
            return int(rng.integers(len(ACTIONS)))
        best = self.best_actions(s)
        return best[0] if len(best) == 1 else best[int(rng.integers(len(best)))]
