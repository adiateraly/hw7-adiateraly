"""Run with: python -m pytest -q   (no gymnasium needed)"""
import pytest

from blackjack_rl.common import END, HIT, STICK, Transition
from blackjack_rl.learners import ModelBased, MonteCarlo, QLearning, TD0
from blackjack_rl import paper_replay


def test_paper_replay_matches_paper():
    comp = paper_replay.run(write_files=False)["comparison"]
    assert comp and all(r[-1] == "match" for r in comp)


def test_td_terminal_continuation_is_zero():
    td = TD0()
    u = td.update(Transition("S", HIT, 1.0, END))
    assert u.new == pytest.approx(0.1)


def test_q_maximises_over_untried_actions():
    q = QLearning()
    q.Q["B"] = [-0.5, 0.0]          # untried hit stays at 0 and is the max
    u = q.update(Transition("A", HIT, 0.0, "B"))
    assert u.succ == 0.0


def test_mc_every_visit_counts_repeated_state():
    mc = MonteCarlo()
    mc.observe_episode([Transition("X", HIT, 0, "X"), Transition("X", STICK, 1, END)])
    assert mc.counts["X"] == 2 and mc.value("X") == 1.0


def test_model_based_unobserved_successor_is_flagged():
    mb = ModelBased()
    mb.observe(Transition("A", HIT, 0.0, "B"))   # B never observed as a source
    res = mb.evaluate(lambda s: HIT)
    assert "B" in res.unobserved and "A" in res.affected and "B" not in res.values


def test_model_based_nonconvergence_reported():
    mb = ModelBased()
    mb.observe(Transition("A", HIT, 1.0, "A"))   # value grows forever with gamma=1
    res = mb.evaluate(lambda s: HIT, max_sweeps=50)
    assert not res.converged and res.sweeps == 50
