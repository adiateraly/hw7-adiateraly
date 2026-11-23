"""Grid tables by player sum x dealer showing card, separately for usable ace yes/no."""
from __future__ import annotations

SUMS = list(range(4, 22))
DEALER = list(range(1, 11))
MISSING = "-"


def grid_md(title: str, cell) -> str:
    """cell(sum, dealer, ace) -> str or None (None = missing). Returns markdown for both ace flags."""
    out = []
    for ace in (0, 1):
        out.append(f"**{title} — usable ace: {'yes' if ace else 'no'}**\n")
        out.append("| sum \\ dealer | " + " | ".join(str(d) for d in DEALER) + " |")
        out.append("|---|" + "---|" * len(DEALER))
        for s in SUMS:
            cells = [cell(s, d, ace) for d in DEALER]
            if all(c is None for c in cells):
                continue
            out.append(f"| {s} | " + " | ".join(MISSING if c is None else c for c in cells) + " |")
        out.append("")
    return "\n".join(out)


def grid_csv_rows(cell, ace: int):
    rows = [["player_sum"] + DEALER]
    for s in SUMS:
        rows.append([s] + [("" if (c := cell(s, d, ace)) is None else c) for d in DEALER])
    return rows
