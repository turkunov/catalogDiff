#!/usr/bin/env python3
"""Pick the fixed manual-test set: N pairs per legacy class, one pair per Avito item where possible.

The output keeps the legacy class for later analysis; the labeling UI never shows it.
Rows are shuffled so that pairs of one class do not come in a block.
"""
import argparse
import random

import pandas as pd

COLUMNS = [
    "q_item_id", "d_item_id", "q_title", "d_title", "q_description", "d_description",
    "q_key_params", "d_key_params", "q_image_id", "d_image_id",
]


def select(df: pd.DataFrame, per_class: int, seed: int) -> tuple[pd.DataFrame, dict[int, int]]:
    rng = random.Random(seed)
    df = df.drop_duplicates(["q_item_id", "d_item_id"]).copy()
    df["legacy_class"] = df["label"].astype(int)
    used_q: set = set()
    chosen: list[int] = []
    shortfall: dict[int, int] = {}
    for cls in range(1, 6):
        idx = df.index[df["legacy_class"] == cls].tolist()
        rng.shuffle(idx)
        fresh = [i for i in idx if df.at[i, "q_item_id"] not in used_q]
        # Greedy: prefer Avito items not used yet, one pair per item within the fresh pass.
        picked: list[int] = []
        seen_here: set = set()
        for i in fresh:
            q = df.at[i, "q_item_id"]
            if q in seen_here:
                continue
            picked.append(i)
            seen_here.add(q)
            if len(picked) == per_class:
                break
        picked_set = set(picked)
        rest = [i for i in idx if i not in picked_set]
        picked += rest[: per_class - len(picked)]
        if len(picked) < per_class:
            shortfall[cls] = per_class - len(picked)
        used_q.update(df.loc[picked, "q_item_id"])
        chosen += picked
    out = df.loc[chosen, COLUMNS + ["legacy_class"]].sample(frac=1, random_state=seed).reset_index(drop=True)
    out.insert(0, "idx", range(len(out)))
    return out, shortfall


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-class", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    df = pd.read_csv(args.pairs, lineterminator="\n")
    out, shortfall = select(df, args.per_class, args.seed)
    out.to_csv(args.out, index=False)
    print(f"rows={len(out)} unique_q={out.q_item_id.nunique()} per_class={out.legacy_class.value_counts().sort_index().to_dict()} shortfall={shortfall}")


if __name__ == "__main__":
    main()
