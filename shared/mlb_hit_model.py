"""
shared/mlb_hit_model.py

Predictive model for MLB player prop markets. Takes a parsed MLBTicker plus
live stats from MLB Stats API and returns the probability that the player
hits the ≥N threshold.

Model structure:
  ── Batter hit props ──────────────────────────────────────────────────
    Hits:          Binomial(n=expected_AB, p=adjusted_AVG)
    Total bases:   Poisson(λ=expected_AB × adjusted_SLG)
    Home runs:     Binomial(n=expected_PA, p=adjusted_HR_rate)
    HR-or-run:     approximate via HR_rate + R_rate

  ── Pitcher strikeout props ────────────────────────────────────────────
    Ks:            Poisson(λ=expected_IP × K9 / 9)

Adjustments applied to base rates (in order):
  1. Blend season stats (70%) with last-10-games rolling (30%).
     Rolling form matters but a single bad week shouldn't torch a 500-PA sample.
  2. Park factor — Coors = 1.14, Petco = 0.93. Applied as sqrt for hits
     (hits are less park-dependent than runs).
  3. Pitcher quality — higher K/9 opposing pitcher reduces batter AVG.
     Empirical rule of thumb: each K/9 above league avg reduces AVG by ~0.6%.
  4. v12.2: Platoon splits — LHB vs LHP get -8%, RHB vs LHP get +8%, etc.

None of these are ML — they're closed-form probability calculations with
defensible priors. The entire point is to be differentiated and explainable.
Once we have 500+ resolved predictions, we can bolt on ML residual correction.

Dependencies: only numpy (already in requirements.txt). No scipy needed —
Binomial and Poisson are computed from log-space factorials for numerical
stability on small inputs.
"""

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from shared.kalshi_ticker_parser import MLBTicker
from shared.mlb_stats_fetcher import (
    BatterSeasonStats,
    BatterRollingStats,
    PitcherSeasonStats,
    PARK_FACTORS,
)

logger = logging.getLogger(__name__)


# ── Model constants ───────────────────────────────────────────────────────────

LEAGUE_AVG_K9        = 8.8       # 2024-2025 MLB average K/9 for starters
LEAGUE_AVG_AVG       = 0.243     # 2024-2025 MLB batting average
LEAGUE_AVG_SLG       = 0.399     # 2024-2025 MLB slugging
LEAGUE_AVG_HR_RATE   = 0.032     # HR per PA

DEFAULT_EXPECTED_AB  = 4         # typical starter gets ~4 ABs in a 9-inning game
DEFAULT_EXPECTED_PA  = 4.3       # PAs include walks (~7% walk rate → 4 AB × 1.07)
DEFAULT_STARTER_IP   = 5.5       # typical MLB starter in 2026

# Weight on rolling (last-10) stats when blending with season stats
ROLLING_WEIGHT = 0.30

# Minimum sample sizes to trust stats. Below these, blend toward league avg.
MIN_PA_FOR_STATS = 30            # <30 PA → lean heavily on league avg
MIN_IP_FOR_STATS = 10            # <10 IP → lean heavily on league avg

# Confidence scaling — how much we trust the model's output.
# The model is most confident when we have ample data for both batter and
# pitcher, and the prediction is far from 0.5.
MAX_CONFIDENCE = 0.85
MIN_CONFIDENCE = 0.40


# ── Probability helpers ───────────────────────────────────────────────────────

def _log_factorial(n: int) -> float:
    return math.lgamma(n + 1)


def binomial_pmf(k: int, n: int, p: float) -> float:
    """P(X = k) for X ~ Binomial(n, p)."""
    if k < 0 or k > n or n <= 0:
        return 0.0
    p = max(min(p, 1.0), 0.0)
    if p == 0.0:
        return 1.0 if k == 0 else 0.0
    if p == 1.0:
        return 1.0 if k == n else 0.0
    log_coef = _log_factorial(n) - _log_factorial(k) - _log_factorial(n - k)
    log_pmf  = log_coef + k * math.log(p) + (n - k) * math.log(1 - p)
    return math.exp(log_pmf)


def binomial_tail(k_min: int, n: int, p: float) -> float:
    """P(X >= k_min) for X ~ Binomial(n, p)."""
    if k_min <= 0:
        return 1.0
    if k_min > n:
        return 0.0
    return sum(binomial_pmf(k, n, p) for k in range(k_min, n + 1))


def poisson_pmf(k: int, lam: float) -> float:
    """P(X = k) for X ~ Poisson(lam)."""
    if k < 0 or lam < 0:
        return 0.0
    if lam == 0:
        return 1.0 if k == 0 else 0.0
    log_pmf = k * math.log(lam) - lam - _log_factorial(k)
    return math.exp(log_pmf)


def poisson_tail(k_min: int, lam: float, k_max_iter: int = 30) -> float:
    """P(X >= k_min) for X ~ Poisson(lam).

    Computes 1 - CDF(k_min - 1). Sums head probabilities (more stable for
    small lambda, small k_min — the regime we care about).
    """
    if k_min <= 0:
        return 1.0
    head = sum(poisson_pmf(k, lam) for k in range(0, k_min))
    return max(0.0, 1.0 - head)


# ── Stat blending ─────────────────────────────────────────────────────────────

def _blend_to_league(value: float, sample_size: int, min_sample: int,
                     league_avg: float) -> float:
    """Shrink toward league average when sample is thin.

    Above min_sample → no shrinkage
    Below min_sample → linear blend proportional to sample fraction
    """
    if sample_size >= min_sample:
        return value
    if sample_size <= 0:
        return league_avg
    w = sample_size / min_sample
    return w * value + (1 - w) * league_avg


def _blend_season_and_rolling(
    season_val:  float,
    rolling_val: Optional[float],
) -> float:
    """Weighted average of season and last-10 stats."""
    if rolling_val is None:
        return season_val
    return (1 - ROLLING_WEIGHT) * season_val + ROLLING_WEIGHT * rolling_val


def _pitcher_adjustment(pitcher_k9: Optional[float]) -> float:
    """Multiplicative adjustment to batter AVG based on opposing pitcher quality.

    A pitcher with K/9 = league avg → factor = 1.0
    Each K/9 above league avg → ~0.6% reduction in batter AVG
    Floor at 0.85, ceiling at 1.10.
    """
    if pitcher_k9 is None or pitcher_k9 <= 0:
        return 1.0
    delta = pitcher_k9 - LEAGUE_AVG_K9
    factor = 1.0 - 0.006 * delta
    return max(0.85, min(1.10, factor))


# ── Platoon split adjustments (v12.2) ────────────────────────────────────────
# Source: Historical MLB split data (2015-2024)
#
# RHB vs LHP: +8% (batter advantage — sees ball better from opposite side)
# LHB vs RHP: +8% (batter advantage)
# RHB vs RHP: -5% (pitcher advantage, same-side)
# LHB vs LHP: -8% (pitcher advantage, most extreme — LHP are specialists)
# Switch hitters: slight advantage (they always bat from platoon side)

PLATOON_ADJUSTMENTS = {
    # (batter_hand, pitcher_hand) → multiplier on batter AVG/SLG
    ("R", "L"): 1.08,   # RHB vs LHP — batter advantage
    ("L", "R"): 1.08,   # LHB vs RHP — batter advantage
    ("R", "R"): 0.95,   # RHB vs RHP — pitcher advantage
    ("L", "L"): 0.92,   # LHB vs LHP — pitcher advantage (strongest)
    ("S", "L"): 1.04,   # Switch vs LHP — slight advantage (bats R)
    ("S", "R"): 1.04,   # Switch vs RHP — slight advantage (bats L)
}


def _platoon_adjustment(batter_hand: Optional[str], pitcher_hand: Optional[str]) -> float:
    """Multiplicative adjustment based on batter/pitcher handedness matchup.

    Returns 1.0 if either hand is unknown.
    """
    if not batter_hand or not pitcher_hand:
        return 1.0

    bh = batter_hand.upper()
    ph = pitcher_hand.upper()

    return PLATOON_ADJUSTMENTS.get((bh, ph), 1.0)


# ── Model prediction dataclass ───────────────────────────────────────────────

@dataclass
class ModelPrediction:
    prob_yes:       float          # probability the market resolves YES
    lam_or_p:       float          # underlying rate (expected value / p)
    distribution:   str            # "binomial" or "poisson"
    n_trials:       Optional[int]  # for binomial
    confidence:     float          # 0.4–0.85
    rationale:      str            # human-readable explanation
    features:       np.ndarray     # feature vector for downstream logging


# ── Main prediction functions ────────────────────────────────────────────────

def predict_hits(
    ticker:         MLBTicker,
    batter_season:  Optional[BatterSeasonStats],
    batter_rolling: Optional[BatterRollingStats],
    pitcher_stats:  Optional[PitcherSeasonStats],
) -> ModelPrediction:
    """P(batter gets ≥ threshold hits)."""

    # Base AVG from season (blended with league avg if sample is thin)
    if batter_season and batter_season.at_bats > 0:
        season_avg = batter_season.avg
        sample_pa  = batter_season.plate_apps
    else:
        season_avg = LEAGUE_AVG_AVG
        sample_pa  = 0

    season_avg = _blend_to_league(
        season_avg, sample_pa, MIN_PA_FOR_STATS, LEAGUE_AVG_AVG,
    )

    # Blend with rolling form
    rolling_avg = batter_rolling.avg if batter_rolling and batter_rolling.at_bats > 0 else None
    blended_avg = _blend_season_and_rolling(season_avg, rolling_avg)

    # Park factor — sqrt of run factor for hits (less extreme than for runs)
    home_team_id = ticker.home_team_id
    park_factor  = PARK_FACTORS.get(home_team_id, 1.00) if home_team_id else 1.00
    park_hit_factor = math.sqrt(park_factor)

    # Pitcher K/9 adjustment
    pitcher_k9 = pitcher_stats.k_per_9 if pitcher_stats else None
    pitcher_factor = _pitcher_adjustment(pitcher_k9)

    # v12.2: Platoon adjustment based on batter/pitcher handedness
    batter_hand = getattr(batter_season, 'bat_hand', None) if batter_season else None
    pitcher_hand = getattr(pitcher_stats, 'hand', None) if pitcher_stats else None
    platoon_factor = _platoon_adjustment(batter_hand, pitcher_hand)

    # Final adjusted AVG
    adj_avg = blended_avg * park_hit_factor * pitcher_factor * platoon_factor
    adj_avg = max(0.05, min(0.50, adj_avg))  # sanity floor/ceiling

    # P(≥ threshold hits) from Binomial(n=AB, p=adj_avg)
    prob = binomial_tail(ticker.threshold, DEFAULT_EXPECTED_AB, adj_avg)

    # Confidence scales with sample depth and distance from 0.5
    sample_confidence = min(1.0, sample_pa / 200)
    edge_confidence   = abs(prob - 0.5) * 2
    confidence = MIN_CONFIDENCE + (MAX_CONFIDENCE - MIN_CONFIDENCE) * (
        0.6 * sample_confidence + 0.4 * edge_confidence
    )

    # Build rationale string
    rationale = (
        f"{ticker.player_display} | "
        f"season AVG={season_avg:.3f} (n={sample_pa}PA) | "
    )
    if rolling_avg:
        rationale += f"rolling={rolling_avg:.3f} | "
    else:
        rationale += "rolling=n/a | "
    rationale += f"park={park_hit_factor:.2f} | "
    if pitcher_k9:
        rationale += f"pitcher K/9={pitcher_k9:.1f} adj={pitcher_factor:.2f} | "
    else:
        rationale += "pitcher=unknown | "
    # v12.2: Add platoon info to rationale
    if batter_hand and pitcher_hand:
        rationale += f"platoon={batter_hand}v{pitcher_hand} adj={platoon_factor:.2f} | "
    rationale += f"adj_AVG={adj_avg:.3f} → P(≥{ticker.threshold} hits in {DEFAULT_EXPECTED_AB}AB)={prob:.3f}"

    # v12.2: Add platoon_factor to feature vector
    features = np.array([
        blended_avg, adj_avg, park_hit_factor, pitcher_factor, platoon_factor,
        float(ticker.threshold), float(sample_pa),
    ])

    return ModelPrediction(
        prob_yes     = prob,
        lam_or_p     = adj_avg,
        distribution = "binomial",
        n_trials     = DEFAULT_EXPECTED_AB,
        confidence   = confidence,
        rationale    = rationale,
        features     = features,
    )


def predict_total_bases(
    ticker:         MLBTicker,
    batter_season:  Optional[BatterSeasonStats],
    batter_rolling: Optional[BatterRollingStats],
    pitcher_stats:  Optional[PitcherSeasonStats],
) -> ModelPrediction:
    """P(batter gets ≥ threshold total bases).

    Model: Poisson(λ = AB × adjusted_SLG). Approximation — real TB
    distribution is multinomial, but Poisson is a decent first-order model
    and doesn't require estimating individual 1B/2B/3B/HR rates.
    """
    if batter_season and batter_season.at_bats > 0:
        season_slg = batter_season.slg
        sample_pa  = batter_season.plate_apps
    else:
        season_slg = LEAGUE_AVG_SLG
        sample_pa  = 0

    season_slg = _blend_to_league(
        season_slg, sample_pa, MIN_PA_FOR_STATS, LEAGUE_AVG_SLG,
    )

    rolling_slg = None
    if batter_rolling and batter_rolling.at_bats > 0:
        rolling_slg = batter_rolling.total_bases / batter_rolling.at_bats
    blended_slg = _blend_season_and_rolling(season_slg, rolling_slg)

    home_team_id = ticker.home_team_id
    park_factor  = PARK_FACTORS.get(home_team_id, 1.00) if home_team_id else 1.00
    # Total bases scale closer to runs than hits → use park_factor directly, not sqrt
    park_tb_factor = park_factor

    pitcher_k9 = pitcher_stats.k_per_9 if pitcher_stats else None
    pitcher_factor = _pitcher_adjustment(pitcher_k9)

    # v12.2: Platoon adjustment
    batter_hand = getattr(batter_season, 'bat_hand', None) if batter_season else None
    pitcher_hand = getattr(pitcher_stats, 'hand', None) if pitcher_stats else None
    platoon_factor = _platoon_adjustment(batter_hand, pitcher_hand)

    adj_slg = blended_slg * park_tb_factor * pitcher_factor * platoon_factor
    adj_slg = max(0.10, min(1.00, adj_slg))

    lam = DEFAULT_EXPECTED_AB * adj_slg
    prob = poisson_tail(ticker.threshold, lam)

    sample_confidence = min(1.0, sample_pa / 200)
    edge_confidence   = abs(prob - 0.5) * 2
    confidence = MIN_CONFIDENCE + (MAX_CONFIDENCE - MIN_CONFIDENCE) * (
        0.6 * sample_confidence + 0.4 * edge_confidence
    )

    rationale = (
        f"{ticker.player_display} | "
        f"season SLG={season_slg:.3f} (n={sample_pa}PA) | "
        f"park={park_tb_factor:.2f} pitcher_adj={pitcher_factor:.2f} | "
    )
    if batter_hand and pitcher_hand:
        rationale += f"platoon={batter_hand}v{pitcher_hand} adj={platoon_factor:.2f} | "
    rationale += f"λ={lam:.2f} → P(≥{ticker.threshold} TB)={prob:.3f}"

    # v12.2: Add platoon_factor to feature vector
    features = np.array([
        blended_slg, adj_slg, park_tb_factor, pitcher_factor, platoon_factor,
        float(ticker.threshold), float(sample_pa),
    ])

    return ModelPrediction(
        prob_yes     = prob,
        lam_or_p     = lam,
        distribution = "poisson",
        n_trials     = None,
        confidence   = confidence,
        rationale    = rationale,
        features     = features,
    )


def predict_home_runs(
    ticker:         MLBTicker,
    batter_season:  Optional[BatterSeasonStats],
    batter_rolling: Optional[BatterRollingStats],
    pitcher_stats:  Optional[PitcherSeasonStats],
) -> ModelPrediction:
    """P(batter hits ≥ threshold home runs)."""

    if batter_season and batter_season.plate_apps > 0:
        season_hr_rate = batter_season.home_runs / batter_season.plate_apps
        sample_pa      = batter_season.plate_apps
    else:
        season_hr_rate = LEAGUE_AVG_HR_RATE
        sample_pa      = 0

    season_hr_rate = _blend_to_league(
        season_hr_rate, sample_pa, MIN_PA_FOR_STATS, LEAGUE_AVG_HR_RATE,
    )

    rolling_hr_rate = None
    if batter_rolling and batter_rolling.at_bats > 0:
        # Use AB as denominator for rolling since we don't track walks there
        rolling_hr_rate = batter_rolling.home_runs / batter_rolling.at_bats
    blended_hr_rate = _blend_season_and_rolling(season_hr_rate, rolling_hr_rate)

    home_team_id = ticker.home_team_id
    park_factor  = PARK_FACTORS.get(home_team_id, 1.00) if home_team_id else 1.00
    # HRs are heavily park-dependent — use factor^1.2 for stronger effect
    park_hr_factor = park_factor ** 1.2

    pitcher_k9 = pitcher_stats.k_per_9 if pitcher_stats else None
    # Pitcher K/9 is a weaker proxy for HR suppression — use half the effect
    pitcher_factor = 1.0 + 0.5 * (_pitcher_adjustment(pitcher_k9) - 1.0)

    # v12.2: Platoon adjustment — HR effect is ~30% stronger than AVG effect
    batter_hand = getattr(batter_season, 'bat_hand', None) if batter_season else None
    pitcher_hand = getattr(pitcher_stats, 'hand', None) if pitcher_stats else None
    base_platoon = _platoon_adjustment(batter_hand, pitcher_hand)
    # Amplify platoon effect for HRs (power is more platoon-sensitive than contact)
    if base_platoon != 1.0:
        platoon_factor = 1.0 + (base_platoon - 1.0) * 1.3
    else:
        platoon_factor = 1.0

    adj_hr_rate = blended_hr_rate * park_hr_factor * pitcher_factor * platoon_factor
    adj_hr_rate = max(0.001, min(0.20, adj_hr_rate))

    # Binomial over PAs (not ABs) since walks don't produce HRs but we use PA as trials
    n_trials = int(round(DEFAULT_EXPECTED_PA))
    prob = binomial_tail(ticker.threshold, n_trials, adj_hr_rate)

    sample_confidence = min(1.0, sample_pa / 300)
    edge_confidence   = abs(prob - 0.5) * 2
    confidence = MIN_CONFIDENCE + (MAX_CONFIDENCE - MIN_CONFIDENCE) * (
        0.5 * sample_confidence + 0.5 * edge_confidence
    )

    rationale = (
        f"{ticker.player_display} | "
        f"HR/PA={blended_hr_rate:.4f} (n={sample_pa}PA) | "
        f"park={park_hr_factor:.2f} | "
    )
    if batter_hand and pitcher_hand:
        rationale += f"platoon={batter_hand}v{pitcher_hand} adj={platoon_factor:.2f} | "
    rationale += f"adj_rate={adj_hr_rate:.4f} → P(≥{ticker.threshold} HR in {n_trials}PA)={prob:.3f}"

    # v12.2: Add platoon_factor to feature vector
    features = np.array([
        blended_hr_rate, adj_hr_rate, park_hr_factor, pitcher_factor, platoon_factor,
        float(ticker.threshold), float(sample_pa),
    ])

    return ModelPrediction(
        prob_yes     = prob,
        lam_or_p     = adj_hr_rate,
        distribution = "binomial",
        n_trials     = n_trials,
        confidence   = confidence,
        rationale    = rationale,
        features     = features,
    )


def predict_pitcher_ks(
    ticker:         MLBTicker,
    pitcher_season: Optional[PitcherSeasonStats],
) -> ModelPrediction:
    """P(pitcher records ≥ threshold strikeouts)."""

    if pitcher_season and pitcher_season.innings > 0:
        k9          = pitcher_season.k_per_9
        sample_ip   = pitcher_season.innings
    else:
        k9          = LEAGUE_AVG_K9
        sample_ip   = 0

    k9 = _blend_to_league(k9, int(sample_ip), MIN_IP_FOR_STATS, LEAGUE_AVG_K9)

    lam = DEFAULT_STARTER_IP * k9 / 9.0
    prob = poisson_tail(ticker.threshold, lam)

    sample_confidence = min(1.0, sample_ip / 100)
    edge_confidence   = abs(prob - 0.5) * 2
    confidence = MIN_CONFIDENCE + (MAX_CONFIDENCE - MIN_CONFIDENCE) * (
        0.6 * sample_confidence + 0.4 * edge_confidence
    )

    rationale = (
        f"{ticker.player_display} | "
        f"K/9={k9:.2f} (n={sample_ip:.1f}IP) | "
        f"expected IP={DEFAULT_STARTER_IP} | "
        f"λ={lam:.2f} → P(≥{ticker.threshold} K)={prob:.3f}"
    )

    features = np.array([
        k9, lam, float(ticker.threshold), float(sample_ip),
    ])

    return ModelPrediction(
        prob_yes     = prob,
        lam_or_p     = lam,
        distribution = "poisson",
        n_trials     = None,
        confidence   = confidence,
        rationale    = rationale,
        features     = features,
    )


# ── Unified entry point ──────────────────────────────────────────────────────

def predict_mlb_prop(
    ticker:         MLBTicker,
    batter_season:  Optional[BatterSeasonStats] = None,
    batter_rolling: Optional[BatterRollingStats] = None,
    pitcher_stats:  Optional[PitcherSeasonStats] = None,
) -> Optional[ModelPrediction]:
    """Dispatch to the right sub-predictor based on prop_code."""
    code = ticker.prop_code
    if code == "HIT":
        return predict_hits(ticker, batter_season, batter_rolling, pitcher_stats)
    if code == "TB":
        return predict_total_bases(ticker, batter_season, batter_rolling, pitcher_stats)
    if code in ("HR", "HRR"):
        # HRR (HR or run) — approximated as HR probability (underestimate, but
        # closer to truth than flat prior). A proper HRR model needs run-scored
        # rate which requires batting-order context.
        return predict_home_runs(ticker, batter_season, batter_rolling, pitcher_stats)
    if code == "KS":
        return predict_pitcher_ks(ticker, pitcher_stats)
    logger.debug("no model for prop_code=%s", code)
    return None


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from shared.kalshi_ticker_parser import parse_mlb_ticker
    from shared.mlb_stats_fetcher import (
        lookup_player, fetch_batter_stats, fetch_batter_rolling,
        fetch_pitcher_stats,
    )
    from shared.kalshi_ticker_parser import MLB_TEAMS

    print("=" * 78)
    print("MLB hit-rate model — end-to-end self-test (v12.2 with platoon splits)")
    print("=" * 78)

    # Test platoon adjustment function
    print("\n── Platoon adjustment verification ──")
    test_cases = [
        ("R", "L", 1.08, "RHB vs LHP"),
        ("L", "R", 1.08, "LHB vs RHP"),
        ("R", "R", 0.95, "RHB vs RHP"),
        ("L", "L", 0.92, "LHB vs LHP"),
        ("S", "L", 1.04, "Switch vs LHP"),
        ("S", "R", 1.04, "Switch vs RHP"),
        (None, "R", 1.00, "Unknown batter"),
        ("R", None, 1.00, "Unknown pitcher"),
    ]
    for bh, ph, expected, desc in test_cases:
        result = _platoon_adjustment(bh, ph)
        status = "✓" if abs(result - expected) < 0.001 else "✗"
        print(f"  {status} {desc}: {result:.2f} (expected {expected:.2f})")

    # Walk a few tickers through the entire pipeline
    print("\n── End-to-end ticker tests ──")
    TICKERS = [
        "KXMLBHIT-26APR092140COLSD-SDFTATIS23-1",   # ≥1 hit — should be ~0.65
        "KXMLBHIT-26APR092140COLSD-SDFTATIS23-2",   # ≥2 hits — should be ~0.25
        "KXMLBHIT-26APR092140COLSD-SDFTATIS23-3",   # ≥3 hits — should be ~0.05
        "KXMLBTB-26APR092140COLSD-SDMMACHADO13-2",  # ≥2 TB — common
        "KXMLBHR-26APR092140COLSD-SDFTATIS23-1",    # ≥1 HR — ~0.12
    ]

    for raw in TICKERS:
        print(f"\n── {raw} ─────────────────────────────────")
        ticker = parse_mlb_ticker(raw)
        if not ticker:
            print("  ✗ parse failed")
            continue

        team_id = MLB_TEAMS[ticker.player_team_code][0]
        player = lookup_player(
            team_id, ticker.player_first, ticker.player_last, ticker.player_jersey,
        )
        if not player:
            print(f"  ✗ player lookup failed for {ticker.player_display}")
            continue

        season  = fetch_batter_stats(player.player_id)
        rolling = fetch_batter_rolling(player.player_id, n_games=10)

        pred = predict_mlb_prop(ticker, season, rolling, pitcher_stats=None)
        if not pred:
            print(f"  ✗ no prediction returned")
            continue

        print(f"  → {pred.rationale}")
        print(f"    prob_yes={pred.prob_yes:.3f}  confidence={pred.confidence:.2f}")

    print("\n" + "=" * 78)
    print("Done. If probabilities vary per threshold/player, model is working.")
    print("=" * 78)