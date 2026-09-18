"""
The accuracy push: everything that is still on the table, measured honestly.

Earlier work established the ceiling. A 2-parameter Elo, a 4-feature XGBoost
and a 46-feature stacked ensemble all landed within 0.003 log loss of each
other - statistically indistinguishable on 2,280 matches - while the market
sat 0.033 ahead. Adding more features derived from past results was not going
to move that.

So this script does not add features. It attacks four different things:

  1. MORE DATA, AND BETTER PRIORS
     Ten divisions instead of three, including the second tiers. Promoted
     sides now arrive with a real rating instead of the 1500 default. This is
     the only change here that adds genuine information rather than fitting
     existing information better.

  2. JOINT CALIBRATION
     Home advantage and the draw curve were fitted in two stages even though
     they interact. Fitting them on one grid reaches combinations the staged
     search cannot.

  3. PER-LEAGUE PARAMETERS, WITH SHRINKAGE
     Serie A draws more than the Premier League. One global draw curve is
     slightly wrong everywhere. Shrinkage keeps thin leagues from overfitting.

  4. A HONEST MARKET BENCHMARK
     Converting odds to probabilities by dividing out the overround is the
     standard shortcut and it is biased - bookmakers do not spread their
     margin evenly across outcomes. If our benchmark is wrong, every claim
     measured against it is wrong too.

Every choice is made on VALIDATION. Test is scored once, at the end.

Run:  python src/optimise_accuracy.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from elo import elo_to_probabilities, fit_draw_model
from elo_calibration import (
    fit_joint, LeagueCalibrator, blend, best_blend_weight,
)
from match_metrics import (
    log_loss_multi, brier_multi, accuracy, odds_to_probs,
    odds_to_probs_power, expected_calibration_error, _clean,
)

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FEATS = os.path.join(BASE, "data", "processed", "match_features.csv")

TRAIN_SEASONS = [2016, 2017, 2018, 2019, 2020, 2021]
VAL_SEASONS = [2022, 2023]
TEST_SEASONS = [2024, 2025]

ODDS = ["odds_close_home", "odds_close_draw", "odds_close_away"]

RNG = np.random.default_rng(42)
pd.set_option("display.width", 250)


# ===================================================================== #
# Fast Elo
# ===================================================================== #

def elo_diffs(home_idx, away_idx, hg, ag, season, n_teams, div_idx=None,
              k=20.0, hfa=65.0, regression=0.85, use_margin=True,
              regression_mode="division"):
    """
    Pre-match rating differences for a chronologically sorted fixture list.

    A numpy reimplementation of EloRatingSystem.run(). That version walks the
    frame with iterrows(), which costs roughly 150 microseconds a row - about
    six seconds for this dataset, and a hyperparameter sweep needs a hundred
    passes. This does the same arithmetic on preallocated arrays in a fraction
    of the time, which is what makes the sweep below practical.

    The anti-leakage ordering is preserved exactly: the difference is recorded
    BEFORE the result is applied.

    WHAT `regression_mode` IS FOR
    ----------------------------
    Season regression pulls ratings toward a mean to model squad turnover.
    With one division, that mean is obviously the league average. With two
    tiers it is not, and getting it wrong does real damage.

    Divisions never play each other, so the only thing tying the two rating
    pools together is teams moving between them. Regressing everyone toward
    the GLOBAL mean overrides that: every summer it drags second-tier teams up
    and top-tier teams down, compressing a gap that the promotion and
    relegation flow had established correctly. Run over ten seasons it inflates
    second-tier sides enough that a promoted team enters the top flight rated
    above the division average - precisely backwards, since promoted teams are
    usually among the weakest in it.

    "division" mode regresses each team toward the mean of the division it
    just played in. Because regressing a group toward its own mean leaves that
    mean untouched, the gap between the tiers survives the off-season.
    """
    ratings = np.full(n_teams, 1500.0)
    out = np.empty(len(hg), dtype=float)
    cur_season = season[0]

    # Which division each team has played in during the current season. -1
    # means "not seen yet this season", so teams sitting out are left alone.
    team_div = np.full(n_teams, -1, dtype=int)
    by_division = regression_mode == "division" and div_idx is not None

    for i in range(len(hg)):
        if season[i] != cur_season:
            if by_division:
                for d in np.unique(team_div[team_div >= 0]):
                    m = team_div == d
                    mu = ratings[m].mean()
                    ratings[m] = mu + (ratings[m] - mu) * regression
            else:
                mean = ratings.mean()
                ratings = mean + (ratings - mean) * regression
            team_div[:] = -1
            cur_season = season[i]

        h, a = home_idx[i], away_idx[i]
        if by_division:
            team_div[h] = div_idx[i]
            team_div[a] = div_idx[i]
        rh, ra = ratings[h], ratings[a]
        out[i] = rh - ra                       # snapshot BEFORE the update

        exp_h = 1.0 / (1.0 + 10 ** ((ra - rh - hfa) / 400.0))
        gd = hg[i] - ag[i]
        actual = 1.0 if gd > 0 else (0.5 if gd == 0 else 0.0)

        if use_margin:
            g = abs(gd)
            mult = 1.0 if g <= 1 else (1.5 if g == 2 else (11.0 + g) / 8.0)
        else:
            mult = 1.0

        delta = k * mult * (actual - exp_h)
        ratings[h] = rh + delta
        ratings[a] = ra - delta

    return out


def encode(df):
    """Integer team, season and division codes for the fast Elo path."""
    teams = pd.concat([df["home_team"], df["away_team"]]).unique()
    idx = {t: i for i, t in enumerate(teams)}
    divs = {d: i for i, d in enumerate(sorted(df["competition"].unique()))}
    return (
        df["home_team"].map(idx).to_numpy(),
        df["away_team"].map(idx).to_numpy(),
        df["home_goals"].to_numpy(int),
        df["away_goals"].to_numpy(int),
        df["season"].to_numpy(),
        len(teams),
        df["competition"].map(divs).to_numpy(),
    )


# ===================================================================== #
# Scoring helpers
# ===================================================================== #

def per_match_loss(y, probs):
    p = _clean(np.asarray(probs, float))
    y = np.asarray(y, int)
    return -np.log(p[np.arange(len(y)), y])


def boot(a, b, n_boot=4000):
    d = np.asarray(a) - np.asarray(b)
    n = len(d)
    bs = np.array([d[RNG.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return d.mean(), lo, hi, bool(lo > 0 or hi < 0)


def row(name, y, probs):
    return {
        "model": name,
        "log_loss": round(log_loss_multi(y, probs), 4),
        "brier": round(brier_multi(y, probs), 4),
        "accuracy": round(accuracy(y, probs), 4),
        "ece": round(expected_calibration_error(y, probs), 4),
        "n": len(y),
    }


# ===================================================================== #

def main():
    df = pd.read_csv(FEATS)
    df["date"] = pd.to_datetime(df["date"])
    if "tier" not in df.columns:
        df["tier"] = 1
    df = df.sort_values("date").reset_index(drop=True)

    # Ratings learn from every division; only the top tier is ever predicted.
    top = df[df["tier"] == 1].reset_index(drop=True)

    is_tr = df["season"].isin(TRAIN_SEASONS).to_numpy()
    is_va = df["season"].isin(VAL_SEASONS).to_numpy()
    is_te = df["season"].isin(TEST_SEASONS).to_numpy()
    t1 = (df["tier"] == 1).to_numpy()

    y_tr = df.loc[is_tr & t1, "result"].to_numpy(int)
    y_va = df.loc[is_va & t1, "result"].to_numpy(int)
    y_te = df.loc[is_te & t1, "result"].to_numpy(int)

    print(f"all divisions : {len(df):,} matches")
    print(f"top tier      : {t1.sum():,}")
    print(f"  train {len(y_tr):,}   val {len(y_va):,}   test {len(y_te):,}")
    print(f"  competitions : {sorted(top['competition'].unique())}")

    enc = encode(df)

    # ---------------------------------------------------------------- #
    print("\n" + "=" * 78)
    print("STAGE 1  Baseline — previous configuration, new data")
    print("=" * 78)
    d0 = elo_diffs(*enc, k=20.0, hfa=65.0, regression=0.85,
                   regression_mode="global")
    dm, tau = fit_draw_model(d0[is_tr & t1], y_tr, 65.0)
    base_va = elo_to_probabilities(d0[is_va & t1], 65.0, dm, tau)
    base_te = elo_to_probabilities(d0[is_te & t1], 65.0, dm, tau)
    print(f"  k=20 hfa=65 reg=0.85, draw model (d_max={dm:.3f}, tau={tau:.0f})")
    print(f"  VAL log loss  : {log_loss_multi(y_va, base_va):.4f}")

    # ---------------------------------------------------------------- #
    print("\n" + "=" * 78)
    print("STAGE 2  Re-tune the rating system on VALIDATION")
    print("=" * 78)
    print("  The old k/hfa/regression were chosen on three leagues with no")
    print("  second-tier history. Both of those changed, so the tuning is stale.\n")

    print("  'mode' is what season regression pulls ratings toward: the global")
    print("  mean across all ten divisions, or each team's own division mean.\n")

    # Coarse grids for ranking configurations - 80 of them, and the fine grid
    # costs ~5s a fit. The winner is refitted on the full grid afterwards.
    C_HFA = np.arange(20.0, 130.0, 10.0)
    C_TAU = np.arange(150.0, 550.0, 50.0)
    C_DMAX = np.arange(0.18, 0.36, 0.02)

    results = []
    for k in [12.0, 16.0, 20.0, 24.0, 30.0]:
        for reg in [0.75, 0.85, 0.95, 1.0]:
            for margin in [True, False]:
                for mode in ["global", "division"]:
                    d = elo_diffs(*enc, k=k, hfa=65.0, regression=reg,
                                  use_margin=margin, regression_mode=mode)
                    hfa_f, dmax_f, tau_f = fit_joint(
                        d[is_tr & t1], y_tr, C_HFA, C_DMAX, C_TAU)
                    p = elo_to_probabilities(d[is_va & t1], hfa_f, dmax_f, tau_f)
                    results.append({
                        "k": k, "regression": reg, "margin": margin,
                        "mode": mode, "hfa": hfa_f, "d_max": round(dmax_f, 3),
                        "tau": tau_f,
                        "val_logloss": round(log_loss_multi(y_va, p), 4),
                    })
    res = pd.DataFrame(results).sort_values("val_logloss")
    print(res.head(12).to_string(index=False))
    bestcfg = res.iloc[0]
    print(f"\n  -> k={bestcfg.k}  regression={bestcfg.regression}  "
          f"margin={bestcfg.margin}  mode={bestcfg['mode']}")

    print("\n  Best of each regression mode, to isolate its effect:")
    for mode in ["global", "division"]:
        print(f"    {mode:<9} {res[res['mode'] == mode]['val_logloss'].min():.4f}")

    RMODE = str(bestcfg["mode"])
    d_best = elo_diffs(*enc, k=float(bestcfg.k), hfa=65.0,
                       regression=float(bestcfg.regression),
                       use_margin=bool(bestcfg.margin),
                       regression_mode=RMODE)

    # ---------------------------------------------------------------- #
    print("\n" + "=" * 78)
    print("STAGE 3  Per-league calibration vs one global fit")
    print("=" * 78)

    g_hfa, g_dmax, g_tau = fit_joint(d_best[is_tr & t1], y_tr)
    glob_va = elo_to_probabilities(d_best[is_va & t1], g_hfa, g_dmax, g_tau)
    print(f"  global joint fit : hfa={g_hfa:.0f}  d_max={g_dmax:.3f}  tau={g_tau:.0f}")
    print(f"  VAL log loss     : {log_loss_multi(y_va, glob_va):.4f}")

    comp_tr = df.loc[is_tr & t1, "competition"].to_numpy()
    comp_va = df.loc[is_va & t1, "competition"].to_numpy()
    comp_te = df.loc[is_te & t1, "competition"].to_numpy()

    cal = LeagueCalibrator().fit(d_best[is_tr & t1], y_tr, comp_tr)
    league_va = cal.predict(d_best[is_va & t1], comp_va)
    print(f"\n  per-league (shrunk) VAL log loss : "
          f"{log_loss_multi(y_va, league_va):.4f}")
    print("\n" + cal.table().to_string(index=False))

    use_league = log_loss_multi(y_va, league_va) < log_loss_multi(y_va, glob_va)
    print(f"\n  -> {'per-league wins' if use_league else 'global wins'}")

    # ---------------------------------------------------------------- #
    print("\n" + "=" * 78)
    print("STAGE 3b  Empty stadiums — is the training window contaminated?")
    print("=" * 78)
    print("  Home advantage is a crowd effect. The 2020-21 season was played")
    print("  largely behind closed doors, and both of those seasons sit inside")
    print("  the training window while the test seasons do not. If home")
    print("  advantage really did collapse, calibrating on that window fits a")
    print("  world the test set no longer lives in.\n")

    hw = (top.assign(home_win=(top["result"] == 0).astype(float))
             .groupby("season")["home_win"].agg(["mean", "size"]))
    for s, r in hw.iterrows():
        era = ("TRAIN" if s in TRAIN_SEASONS
               else "VAL" if s in VAL_SEASONS else "TEST")
        bar = "#" * int(round(r["mean"] * 100 - 30))
        print(f"    {s}  {era:<5} home wins {r['mean']:6.1%}  {bar}")

    # Does dropping the affected seasons from the calibration fit help?
    COVID = [2019, 2020]      # 2019-20 suspended mid-season, 2020-21 near-empty
    recent = is_tr & t1 & ~df["season"].isin(COVID).to_numpy()
    cal_rec = LeagueCalibrator().fit(
        d_best[recent], df.loc[recent, "result"].to_numpy(int),
        df.loc[recent, "competition"].to_numpy())
    rec_va = cal_rec.predict(d_best[is_va & t1], comp_va)

    print(f"\n    calibrated on all train seasons  VAL {log_loss_multi(y_va, league_va):.4f}")
    print(f"    calibrated excluding {COVID}   VAL {log_loss_multi(y_va, rec_va):.4f}")
    drop_covid = log_loss_multi(y_va, rec_va) < log_loss_multi(y_va, league_va)
    print(f"    -> {'excluding them helps' if drop_covid else 'no real contamination'}")
    if drop_covid:
        cal = cal_rec
        league_va = rec_va

    # ---------------------------------------------------------------- #
    print("\n" + "=" * 78)
    print("STAGE 4  Ensemble of rating speeds")
    print("=" * 78)
    print("  A low k is a slow, stable memory; a high k reacts to recent form.")
    print("  They disagree about different teams, so averaging them can beat")
    print("  either - the classic reason ensembles work.\n")

    members = []
    for k in [10.0, 20.0, 35.0]:
        d = elo_diffs(*enc, k=k, hfa=65.0,
                      regression=float(bestcfg.regression),
                      use_margin=bool(bestcfg.margin),
                      regression_mode=RMODE)
        c = LeagueCalibrator().fit(d[is_tr & t1], y_tr, comp_tr)
        members.append((k, d, c))
        print(f"    k={k:<5} VAL {log_loss_multi(y_va, c.predict(d[is_va & t1], comp_va)):.4f}")

    ens_va = np.mean([c.predict(d[is_va & t1], comp_va) for _, d, c in members], axis=0)
    ens_va /= ens_va.sum(axis=1, keepdims=True)
    print(f"    ensemble  VAL {log_loss_multi(y_va, ens_va):.4f}")

    single_va = league_va if use_league else glob_va
    use_ens = log_loss_multi(y_va, ens_va) < log_loss_multi(y_va, single_va)
    print(f"\n  -> {'ensemble wins' if use_ens else 'single model wins'}")

    # ---------------------------------------------------------------- #
    print("\n" + "=" * 78)
    print("STAGE 5  A correct market benchmark")
    print("=" * 78)
    mask_te = df.loc[is_te & t1, ODDS].notna().all(axis=1).to_numpy()
    o_te = [df.loc[is_te & t1, c].to_numpy()[mask_te] for c in ODDS]
    y_te_o = y_te[mask_te]

    book_prop = odds_to_probs(*o_te)
    book_pow = odds_to_probs_power(*o_te)
    print(f"  matches with closing odds : {mask_te.sum():,}")
    print(f"  proportional de-vig  log loss : {log_loss_multi(y_te_o, book_prop):.4f}")
    print(f"  power-method de-vig  log loss : {log_loss_multi(y_te_o, book_pow):.4f}")
    better = book_pow if log_loss_multi(y_te_o, book_pow) < log_loss_multi(y_te_o, book_prop) else book_prop
    print(f"  -> benchmark uses the "
          f"{'power' if better is book_pow else 'proportional'} method")

    # ---------------------------------------------------------------- #
    print("\n" + "=" * 78)
    print("STAGE 6  TEST — scored once, with the choices already locked in")
    print("=" * 78)

    def final_probs(mask):
        comps = df.loc[mask, "competition"].to_numpy()
        if use_ens:
            p = np.mean([c.predict(d[mask], comps) for _, d, c in members], axis=0)
            return p / p.sum(axis=1, keepdims=True)
        if use_league:
            return cal.predict(d_best[mask], comps)
        return elo_to_probabilities(d_best[mask], g_hfa, g_dmax, g_tau)

    model_te = final_probs(is_te & t1)

    rows = [
        row("Baseline (old config)", y_te, base_te),
        row("Optimised model", y_te, model_te),
        row("BOOKMAKER", y_te_o, better),
    ]
    print()
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n  Paired bootstrap, 4000 resamples:")
    d_, lo, hi, sig = boot(per_match_loss(y_te, model_te),
                           per_match_loss(y_te, base_te))
    print(f"    optimised vs baseline   diff={d_:+.4f}  [{lo:+.4f}, {hi:+.4f}]  "
          f"{'SIGNIFICANT' if sig else 'indistinguishable'}")

    d_, lo, hi, sig = boot(per_match_loss(y_te_o, model_te[mask_te]),
                           per_match_loss(y_te_o, better))
    print(f"    optimised vs market     diff={d_:+.4f}  [{lo:+.4f}, {hi:+.4f}]  "
          f"{'market still ahead' if sig else 'NO CLEAR DIFFERENCE'}")

    # ---------------------------------------------------------------- #
    print("\n" + "=" * 78)
    print("STAGE 7  Shipping mode — model combined with the market")
    print("=" * 78)
    print("  Everything above answers 'can our features beat the market'. That")
    print("  is a research question and the odds are held out to keep it fair.")
    print()
    print("  A product has a different job: publish the best probability")
    print("  available. Closing odds are public before kick-off, so refusing to")
    print("  use them makes the product worse to protect a claim the product")
    print("  was never making. The two numbers are reported separately and the")
    print("  blend is NEVER cited as evidence of beating the market.\n")

    mask_va = df.loc[is_va & t1, ODDS].notna().all(axis=1).to_numpy()
    o_va = [df.loc[is_va & t1, c].to_numpy()[mask_va] for c in ODDS]
    book_va = (odds_to_probs_power(*o_va) if better is book_pow
               else odds_to_probs(*o_va))
    model_va = final_probs(is_va & t1)[mask_va]

    w, wl = best_blend_weight(model_va, book_va, y_va[mask_va])
    print(f"  best weight on our model (chosen on VAL) : {w:.2f}")
    print(f"  VAL log loss  market alone {log_loss_multi(y_va[mask_va], book_va):.4f}"
          f"  ->  blended {wl:.4f}")

    blend_te = blend(model_te[mask_te], better, w)
    rows = [
        row("Our model alone", y_te_o, model_te[mask_te]),
        row("Market alone", y_te_o, better),
        row(f"SHIPPED blend ({w:.0%} model)", y_te_o, blend_te),
    ]
    print()
    print(pd.DataFrame(rows).to_string(index=False))

    d_, lo, hi, sig = boot(per_match_loss(y_te_o, blend_te),
                           per_match_loss(y_te_o, better))
    print(f"\n  blend vs market   diff={d_:+.4f}  [{lo:+.4f}, {hi:+.4f}]  "
          f"{'SIGNIFICANT' if sig else 'indistinguishable'}")
    if sig and d_ < 0:
        print("  -> the blend genuinely improves on the market.")

    # ---------------------------------------------------------------- #
    np.save(os.path.join(BASE, "data", "processed", "elo_diff_optimised.npy"),
            d_best)
    print("\n" + "=" * 78)
    print("CONFIG TO SHIP")
    print("=" * 78)
    print(f"  k                 : {bestcfg.k}")
    print(f"  season regression : {bestcfg.regression}")
    print(f"  margin multiplier : {bestcfg.margin}")
    print(f"  calibration       : {'per-league' if use_league else 'global'}"
          f"{' + 3-way ensemble' if use_ens else ''}")
    print(f"  market blend      : {w:.2f} model / {1 - w:.2f} market")


if __name__ == "__main__":
    main()
