"""
Do availability features beat the results-only model? Measured, not assumed.

Everything derived from past results has already been shown to be exhausted:
a 2-parameter Elo matched a 46-feature ensemble. Squad availability is the
first genuinely NEW information added to this model, so it gets the same
treatment every other candidate got - chosen on validation, scored once on
test, and compared with a paired bootstrap rather than by eyeballing the
fourth decimal place.

The honest possibilities are that it helps, or that it does not and the market's
edge comes from somewhere else. Both are useful answers.

Run:  python src/evaluate_availability.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from elo_calibration import LeagueCalibrator
from match_metrics import log_loss_multi, brier_multi, accuracy, _clean
from match_model import train, predict_proba

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FEATS = os.path.join(BASE, "data", "processed", "match_features.csv")
AVAIL = os.path.join(BASE, "data", "processed", "availability.csv")

TRAIN_SEASONS = [2016, 2017, 2018, 2019, 2020, 2021]
VAL_SEASONS = [2022, 2023]
TEST_SEASONS = [2024, 2025]

AVAIL_COLS = ["availability", "key_absent", "xi_continuity", "injured_squad"]
RNG = np.random.default_rng(42)
pd.set_option("display.width", 220)


def per_match_loss(y, p):
    p = _clean(np.asarray(p, float))
    y = np.asarray(y, int)
    return -np.log(p[np.arange(len(y)), y])


def boot(a, b, n=4000):
    d = np.asarray(a) - np.asarray(b)
    bs = np.array([d[RNG.integers(0, len(d), len(d))].mean() for _ in range(n)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return d.mean(), lo, hi, bool(lo > 0 or hi < 0)


def attach(df: pd.DataFrame) -> pd.DataFrame:
    """Join per-club availability onto each match, for both sides."""
    av = pd.read_csv(AVAIL)
    av["date"] = pd.to_datetime(av["date"], errors="coerce")
    av = av.dropna(subset=["date", "short_name"])
    av = av.drop_duplicates(["date", "short_name"])

    keep = ["date", "short_name"] + AVAIL_COLS
    for side in ("home", "away"):
        right = av[keep].rename(columns={
            "short_name": f"{side}_team",
            **{c: f"{side}_{c}" for c in AVAIL_COLS}})
        df = df.merge(right, on=["date", f"{side}_team"], how="left")

    for c in AVAIL_COLS:
        df[f"{c}_diff"] = df[f"home_{c}"] - df[f"away_{c}"]
    return df


def main():
    df = pd.read_csv(FEATS, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    if "tier" in df.columns:
        df = df[df["tier"] == 1]
    df = df.sort_values("date").reset_index(drop=True)

    df = attach(df)

    cov = df["home_availability"].notna() & df["away_availability"].notna()
    print(f"matches: {len(df):,}   with availability on both sides: "
          f"{int(cov.sum()):,} ({cov.mean():.1%})")

    by_season = df.groupby("season").apply(
        lambda g: (g["home_availability"].notna()
                   & g["away_availability"].notna()).mean(),
        include_groups=False)
    print("\ncoverage by season:")
    print((by_season * 100).round(1).to_string())

    # Only matches with data on both sides can test the question.
    df = df[cov].reset_index(drop=True)

    tr = df[df["season"].isin(TRAIN_SEASONS)]
    va = df[df["season"].isin(VAL_SEASONS)]
    te = df[df["season"].isin(TEST_SEASONS)]
    print(f"\ntrain {len(tr):,}   val {len(va):,}   test {len(te):,}")
    if len(va) == 0 or len(te) == 0:
        print("Not enough covered matches to evaluate.")
        return

    # ---------------------------------------------------------------- #
    print("\n" + "=" * 74)
    print("Do the features correlate with the outcome at all?")
    print("=" * 74)
    feat_cols = ([f"home_{c}" for c in AVAIL_COLS]
                 + [f"away_{c}" for c in AVAIL_COLS]
                 + [f"{c}_diff" for c in AVAIL_COLS])
    cors = (df[feat_cols].corrwith(df["result"].astype(float))
            .abs().sort_values(ascending=False))
    print(cors.round(4).to_string())
    print(f"\n  elo_diff for reference: "
          f"{abs(df['elo_diff'].corr(df['result'].astype(float))):.4f}")

    # ---------------------------------------------------------------- #
    print("\n" + "=" * 74)
    print("BASELINE — Elo with per-league calibration")
    print("=" * 74)
    cal = LeagueCalibrator().fit(tr["elo_diff"].to_numpy(),
                                 tr["result"].to_numpy(int),
                                 tr["competition"].to_numpy())
    base_va = cal.predict(va["elo_diff"].to_numpy(), va["competition"].to_numpy())
    base_te = cal.predict(te["elo_diff"].to_numpy(), te["competition"].to_numpy())
    print(f"  VAL  {log_loss_multi(va['result'].to_numpy(int), base_va):.4f}")
    print(f"  TEST {log_loss_multi(te['result'].to_numpy(int), base_te):.4f}")

    # ---------------------------------------------------------------- #
    print("\n" + "=" * 74)
    print("CANDIDATE — Elo probabilities stacked with availability")
    print("=" * 74)

    def with_elo(frame):
        p = cal.predict(frame["elo_diff"].to_numpy(),
                        frame["competition"].to_numpy())
        out = frame.copy()
        out["elo_p_home"], out["elo_p_draw"], out["elo_p_away"] = p.T
        out["elo_logit"] = np.log(np.clip(p[:, 0], 1e-6, None)
                                  / np.clip(p[:, 2], 1e-6, None))
        return out

    tr_s, va_s, te_s = with_elo(tr), with_elo(va), with_elo(te)
    elo_p = ["elo_p_home", "elo_p_draw", "elo_p_away", "elo_logit"]

    results = []
    for name, cols in [
        ("Elo probs only", elo_p),
        ("Elo + availability", elo_p + feat_cols),
        ("Elo + diffs only", elo_p + [f"{c}_diff" for c in AVAIL_COLS]),
    ]:
        bst, used, _ = train(tr_s, va_s, features=cols,
                             params={"max_depth": 3, "lambda": 3.0,
                                     "min_child_weight": 30})
        pv = predict_proba(bst, va_s, used)
        pt = predict_proba(bst, te_s, used)
        results.append((name, pv, pt))
        print(f"  {name:<22} VAL {log_loss_multi(va_s['result'].to_numpy(int), pv):.4f}"
              f"   TEST {log_loss_multi(te_s['result'].to_numpy(int), pt):.4f}")

    # ---------------------------------------------------------------- #
    print("\n" + "=" * 74)
    print("TEST — is any difference real?")
    print("=" * 74)
    y_te = te["result"].to_numpy(int)
    l_base = per_match_loss(y_te, base_te)

    rows = [{"model": "Elo calibrated (baseline)",
             "log_loss": round(log_loss_multi(y_te, base_te), 4),
             "brier": round(brier_multi(y_te, base_te), 4),
             "accuracy": round(accuracy(y_te, base_te), 4)}]
    for name, _, pt in results:
        rows.append({"model": name,
                     "log_loss": round(log_loss_multi(y_te, pt), 4),
                     "brier": round(brier_multi(y_te, pt), 4),
                     "accuracy": round(accuracy(y_te, pt), 4)})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n  vs baseline (negative = better), paired bootstrap:")
    for name, _, pt in results:
        d, lo, hi, sig = boot(per_match_loss(y_te, pt), l_base)
        verdict = ("IMPROVES" if sig and d < 0
                   else "worse" if sig else "indistinguishable")
        print(f"    {name:<22} {d:+.4f}  [{lo:+.4f}, {hi:+.4f}]  {verdict}")


if __name__ == "__main__":
    main()
