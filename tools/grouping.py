"""
Single implementation of "which rigid body does this CAD component belong to?".

This logic was duplicated across four scripts, which is how the linear-bearing
carriages ended up in the wrong body: they live under `R6Base Assembly-1` in the
CAD tree but physically ride the rail with the cart, and a first-match rule gave
them to the base.

MOST SPECIFIC PREFIX WINS. `R6Base Assembly-1/R6Linear_Bearings` therefore beats
`R6Base Assembly-1`, and an exact name beats any prefix.
"""

from __future__ import annotations


def body_of(name: str, cfg: dict) -> str | None:
    """Return the body owning `name`, preferring exact matches, then the longest
    matching prefix. Returns None if nothing matches."""
    best_body, best_len = None, -1
    for body, rules in (cfg.get("bodies") or {}).items():
        for e in rules.get("exact", []) or []:
            if name == e:
                return body
        for p in rules.get("prefix", []) or []:
            if name.startswith(p) and len(p) > best_len:
                best_body, best_len = body, len(p)
    return best_body


def label_frame(df, cfg, column: str = "body"):
    """Add a body column to a bodies_raw dataframe."""
    df[column] = [body_of(n, cfg) for n in df["name"]]
    return df
