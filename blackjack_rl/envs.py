"""Environment access. Learners only ever see (observation, reward, terminal flag)."""
from __future__ import annotations

from .common import SAFETY_LIMIT, state_key


def make_env():
    import gymnasium as gym
    return gym.make("Blackjack-v1", natural=False, sab=False)


def play_game(env, choose_action, first_seed=None, safety_limit: int = SAFETY_LIMIT):
    """Play one game. Seeds the env only if first_seed is given (first reset of a run).

    choose_action(state_tuple) -> 0/1.  Returns (steps, cutoff) where steps is a list of
    dicts: state, action, reward, next_obs (raw tuple), terminal.
    A game with `safety_limit` player actions and no terminal flag is a cut-off.
    """
    obs, _ = env.reset(seed=first_seed) if first_seed is not None else env.reset()
    s = state_key(obs)
    steps = []
    for _ in range(safety_limit):
        a = int(choose_action(s))
        nobs, r, terminated, truncated, _ = env.step(a)
        done = bool(terminated or truncated)
        steps.append({"state": s, "action": a, "reward": float(r),
                      "next_obs": state_key(nobs), "terminal": done})
        if done:
            return steps, False
        s = state_key(nobs)
    return steps, True
