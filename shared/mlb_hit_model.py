"""
shared/mlb_hit_model.py  (v14 — dedicated HRR predictor)

Changes vs v13:
  1. NEW: predict_hrr() — models HRR (HR + Run scored) as an additive Poisson
     using R/PA + HR/PA. Previously HRR was routed through predict_home_runs,
     which used HR/PA alone — producing near-zero raw probabilities (~0.001)
     that the sports calibration offset (+0.13) pinned to 0.13 regardless of
     threshold. See 2026-04-23 audit: all 8 MLB HRR trades showed avg_prob ≈
     0.130 across thresholds 2, 3, and 5. That was the calibration floor, not
     a prediction.

  2. NEW: _get_runs_per_pa() — defensive helper that tries batter_season fields
     ('runs_scored', 'runs', 'R') and falls back to OBP-based estimation if
     none exist. This keeps the model resilient to BatterSeasonStats schema
     variations.

  3. NEW: LEAGUE_AVG_R_RATE constant — 0.125 runs scored per PA (2024-2025
     MLB average for non-pitcher batters).

  4. predict_mlb_prop() dispatch updated: HRR now routes to predict_hrr()
     instead of sharing predict_home_runs() with HR.

  HRR model design:
    HRR event count per game = HR count + R count (additive)
    rate_per_PA = HR/PA + R/PA
    λ = expected_PA × rate_per_PA × park × pitcher × platoon
    P(HRR ≥ k) = poisson_tail(k, λ)

    This produces threshold-sensitive probabilities:
      League avg batter:  λ ≈ 0.69, P(≥1)=0.50, P(≥2)=0.16, P(≥3)=0.034, P(≥5)=0.0007
      Top-tier leadoff:   λ ≈ 1.10, P(≥1)=0.67, P(≥2)=0.30, P(≥3)=0.10,  P(≥5)=0.007

    That matches what the 8 observed HRR market prices (19-45¢ YES) imply,
    with enough dispersion to generate real NO edge at higher thresholds.

  All other logic (HIT, TB, HR, KS, platoon, park, WHIP/BAA pitcher) unchanged.

Changes vs v12.2 (retained from v13):
  1. Dynamic expected AB via _estimate_expected_ab (walk rate + game total)
  2. WHIP + BAA pitcher adjustment in _pitcher_adjustment
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

LEAGUE_AVG_K9         = 8.8       # 2024-2025 MLB average K/9 for starters
LEAGUE_AVG_AVG        = 0.243     # 2024-2025 MLB batting average
LEAGUE_AVG_SLG        = 0.399     # 2024-2025 MLB slugging
LEAGUE_AVG_HR_RATE    = 0.032     # HR per PA
LEAGUE_AVG_R_RATE     = 0.125     # v14: Runs scored per PA (non-pitcher batters)
LEAGUE_AVG_OBP        = 0.315     # v14: On-base percentage
LEAGUE_AVG_WHIP       = 1.28      # 2024-2025 MLB average WHIP
LEAGUE_AVG_GAME_TOTAL = 8.5       # Average Vegas O/U
LEAGUE_AVG_PA_PER_GAME = 38.5     # Average team PA per 9 innings

DEFAULT_EXPECTED_PA  = 4.28      # Fallback: league avg PA per batter
DEFAULT_STARTER_IP   = 5.5       # typical MLB starter in 2026

# Weight on rolling (last-10) stats when blending with season stats
ROLLING_WEIGHT = 0.30

# Minimum sample sizes to trust stats. Below these, blend toward league avg.
MIN_PA_FOR_STATS = 30            # <30 PA → lean heavily on league avg
MIN_IP_FOR_STATS = 10            # <10 IP → lean heavily on league avg

# Confidence scaling — how much we trust the model's output.
MAX_CONFIDENCE = 0.85
MIN_CONFIDENCE = 0.40


# ── Dynamic expected AB estimation (v13) ──────────────────────────────────────

def _estimate_expected_ab(
    batter_season: Optional[BatterSeasonStats],
    game_total: Optional[float] = None,
) -> tuple[float, float]:
    """Estimate expected AB and PA for a batter in a specific game.

    Returns (expected_ab, expected_pa).

    Factors:
      1. Game total (Vegas O/U) → higher total = more PA for everyone
      2. Player walk rate → high-OBP batters convert fewer PA to AB

    Without game total, uses league average (4.28 PA).
    """
    if game_total and game_total > 0:
        team_pa = LEAGUE_AVG_PA_PER_GAME * (game_total / LEAGUE_AVG_GAME_TOTAL)
        base_pa = team_pa / 9.0
    else:
        base_pa = DEFAULT_EXPECTED_PA

    base_pa = max(3.0, min(6.0, base_pa))

    if batter_season and batter_season.plate_apps > 30:
        walk_rate = batter_season.walks / batter_season.plate_apps
        ab_fraction = 1.0 - walk_rate - 0.01
        ab_fraction = max(0.80, min(0.96, ab_fraction))
    else:
        ab_fraction = 0.92

    expected_pa = base_pa
    expected_ab = base_pa * ab_fraction

    return expected_ab, expected_pa


# ── Runs-per-PA extractor (v14) ──────────────────────────────────────────────

def _get_runs_per_pa(batter_season: Optional[BatterSeasonStats]) -> Optional[float]:
    """Return runs scored per PA for this batter, or None if unknowable.

    Tries direct stat fields first ('runs_scored', 'runs', 'R'). Falls back to
    OBP-based estimation: R/PA ≈ OBP × 0.30 (league-average scoring rate given
    on base, which captures lineup context implicitly through runs-created
    environment).

    If neither direct runs data nor OBP is available, returns None — caller
    should substitute LEAGUE_AVG_R_RATE.
    """
    if not batter_season or batter_season.plate_apps <= 0:
        return None

    # Direct field (preferred)
    for attr in ('runs_scored', 'runs', 'R'):
        runs = getattr(batter_season, attr, None)
        if runs is not None and runs >= 0:
            return float(runs) / float(batter_season.plate_apps)

    # OBP-based estimate
    obp = getattr(batter_season, 'obp', None)
    if obp is None or obp <= 0:
        # Estimate OBP from AVG + walk rate + ~1% HBP
        try:
            walk_rate = batter_season.walks / batter_season.plate_apps
            obp = batter_season.avg * (batter_season.at_bats / batter_season.plate_apps) + walk_rate + 0.01
        except (ZeroDivisionError, AttributeError):
            return None

    # League-avg scoring rate if on base ≈ 0.30
    # (This includes extra-base hits, being driven in, stealing home, etc.)
    return float(obp) * 0.30


def _get_rolling_runs_per_pa(batter_rolling: Optional[BatterRollingStats]) -> Optional[float]:
    """Same logic for rolling stats. Returns None if not computable."""
    if not batter_rolling:
        return None
    pa = getattr(batter_rolling, 'plate_apps', None) or getattr(batter_rolling, 'at_bats', 0)
    if not pa or pa <= 0:
        return None

    for attr in ('runs_scored', 'runs', 'R'):
        runs = getattr(batter_rolling, attr, None)
        if runs is not None and runs >= 0:
            return float(runs) / float(pa)

    return None


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
    """P(X >= k_min) for X ~ Poisson(lam)."""
    if k_min <= 0:
        return 1.0
    head = sum(poisson_pmf(k, lam) for k in range(0, k_min))
    return max(0.0, 1.0 - head)


# ── Stat blending ─────────────────────────────────────────────────────────────

def _blend_to_league(value: float, sample_size: int, min_sample: int,
                     league_avg: float) -> float:
    """Shrink toward league average when sample is thin."""
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


# ── Pitcher adjustment (v13: WHIP + BAA based) ───────────────────────────────

def _pitcher_adjustment(pitcher_stats: Optional[PitcherSeasonStats]) -> float:
    """Multiplicative adjustment to batter AVG based on opposing pitcher quality.

    Uses WHIP and BAA (batting average against) instead of K/9 alone.
    K/9 measures strikeouts, not hit suppression. A high-K pitcher can still
    give up lots of hits (high WHIP).

    Components:
      - BAA: direct measure of hit-allowing tendency (50% weight)
      - WHIP: total baserunners per inning (30% weight)
      - K/9: retained as secondary signal (20% weight)

    Returns multiplicative factor (0.85 to 1.15).
    """
    if pitcher_stats is None:
        return 1.0

    adjustments = []

    if pitcher_stats.innings > 0 and pitcher_stats.hits_allowed > 0:
        batters_faced = pitcher_stats.innings * 4.3
        if batters_faced > 0:
            baa = pitcher_stats.hits_allowed / batters_faced
            baa = _blend_to_league(baa, int(pitcher_stats.innings), MIN_IP_FOR_STATS, LEAGUE_AVG_AVG)
            baa_factor = baa / LEAGUE_AVG_AVG
            adjustments.append(('baa', baa_factor, 0.50))

    if pitcher_stats.whip > 0 and pitcher_stats.innings >= MIN_IP_FOR_STATS:
        whip = pitcher_stats.whip
        whip_factor = whip / LEAGUE_AVG_WHIP
        adjustments.append(('whip', whip_factor, 0.30))

    if pitcher_stats.k_per_9 > 0:
        k9 = pitcher_stats.k_per_9
        k9 = _blend_to_league(k9, int(pitcher_stats.innings), MIN_IP_FOR_STATS, LEAGUE_AVG_K9)
        delta = k9 - LEAGUE_AVG_K9
        k9_factor = 1.0 - 0.004 * delta
        adjustments.append(('k9', k9_factor, 0.20))

    if not adjustments:
        return 1.0

    total_weight = sum(w for _, _, w in adjustments)
    weighted_factor = sum(f * w for _, f, w in adjustments) / total_weight

    result = max(0.85, min(1.15, weighted_factor))

    logger.debug(
        "[pitcher_adj] whip=%.2f baa_approx k9=%.1f → factor=%.3f (components: %s)",
        pitcher_stats.whip, pitcher_stats.k_per_9, result,
        ", ".join(f"{name}={f:.3f}×{w:.0%}" for name, f, w in adjustments),
    )

    return result


# ── Platoon split adjustments (v12.2) ────────────────────────────────────────

PLATOON_ADJUSTMENTS = {
    ("R", "L"): 1.08,
    ("L", "R"): 1.08,
    ("R", "R"): 0.95,
    ("L", "L"): 0.92,
    ("S", "L"): 1.04,
    ("S", "R"): 1.04,
}


def _platoon_adjustment(batter_hand: Optional[str], pitcher_hand: Optional[str]) -> float:
    if not batter_hand or not pitcher_hand:
        return 1.0
    bh = batter_hand.upper()
    ph = pitcher_hand.upper()
    return PLATOON_ADJUSTMENTS.get((bh, ph), 1.0)


# ── Model prediction dataclass ───────────────────────────────────────────────

@dataclass
class ModelPrediction:
    prob_yes:       float
    lam_or_p:       float
    distribution:   str
    n_trials:       Optional[int]
    confidence:     float
    rationale:      str
    features:       np.ndarray


# ── Main prediction functions ────────────────────────────────────────────────

def predict_hits(
    ticker:         MLBTicker,
    batter_season:  Optional[BatterSeasonStats],
    batter_rolling: Optional[BatterRollingStats],
    pitcher_stats:  Optional[PitcherSeasonStats],
    game_total:     Optional[float] = None,
) -> ModelPrediction:
    """P(batter gets ≥ threshold hits)."""

    if batter_season and batter_season.at_bats > 0:
        season_avg = batter_season.avg
        sample_pa  = batter_season.plate_apps
    else:
        season_avg = LEAGUE_AVG_AVG
        sample_pa  = 0

    season_avg = _blend_to_league(
        season_avg, sample_pa, MIN_PA_FOR_STATS, LEAGUE_AVG_AVG,
    )

    rolling_avg = batter_rolling.avg if batter_rolling and batter_rolling.at_bats > 0 else None
    blended_avg = _blend_season_and_rolling(season_avg, rolling_avg)

    home_team_id = ticker.home_team_id
    park_factor  = PARK_FACTORS.get(home_team_id, 1.00) if home_team_id else 1.00
    park_hit_factor = math.sqrt(park_factor)

    pitcher_factor = _pitcher_adjustment(pitcher_stats)

    batter_hand = getattr(batter_season, 'bat_hand', None) if batter_season else None
    pitcher_hand = getattr(pitcher_stats, 'hand', None) if pitcher_stats else None
    platoon_factor = _platoon_adjustment(batter_hand, pitcher_hand)

    adj_avg = blended_avg * park_hit_factor * pitcher_factor * platoon_factor
    adj_avg = max(0.05, min(0.50, adj_avg))

    expected_ab, expected_pa = _estimate_expected_ab(batter_season, game_total)

    n_ab = int(round(expected_ab))
    n_ab = max(2, min(7, n_ab))
    prob = binomial_tail(ticker.threshold, n_ab, adj_avg)

    sample_confidence = min(1.0, sample_pa / 200)
    edge_confidence   = abs(prob - 0.5) * 2
    data_quality = 0.5
    if pitcher_stats and pitcher_stats.innings >= MIN_IP_FOR_STATS:
        data_quality += 0.25
    if game_total:
        data_quality += 0.25

    confidence = MIN_CONFIDENCE + (MAX_CONFIDENCE - MIN_CONFIDENCE) * (
        0.40 * sample_confidence + 0.30 * edge_confidence + 0.30 * data_quality
    )

    rationale = (
        f"{ticker.player_display} | "
        f"season AVG={season_avg:.3f} (n={sample_pa}PA) | "
    )
    if rolling_avg:
        rationale += f"rolling={rolling_avg:.3f} | "
    else:
        rationale += "rolling=n/a | "
    rationale += f"park={park_hit_factor:.2f} | "
    if pitcher_stats:
        rationale += (
            f"pitcher WHIP={pitcher_stats.whip:.2f} "
            f"K/9={pitcher_stats.k_per_9:.1f} "
            f"adj={pitcher_factor:.3f} | "
        )
    else:
        rationale += "pitcher=unknown | "
    if batter_hand and pitcher_hand:
        rationale += f"platoon={batter_hand}v{pitcher_hand} adj={platoon_factor:.2f} | "
    if game_total:
        rationale += f"O/U={game_total:.1f} | "
    rationale += (
        f"adj_AVG={adj_avg:.3f} AB={n_ab} "
        f"→ P(≥{ticker.threshold} hits)={prob:.3f}"
    )

    features = np.array([
        blended_avg, adj_avg, park_hit_factor, pitcher_factor, platoon_factor,
        float(ticker.threshold), float(sample_pa), expected_ab,
    ])

    return ModelPrediction(
        prob_yes     = prob,
        lam_or_p     = adj_avg,
        distribution = "binomial",
        n_trials     = n_ab,
        confidence   = confidence,
        rationale    = rationale,
        features     = features,
    )


def predict_total_bases(
    ticker:         MLBTicker,
    batter_season:  Optional[BatterSeasonStats],
    batter_rolling: Optional[BatterRollingStats],
    pitcher_stats:  Optional[PitcherSeasonStats],
    game_total:     Optional[float] = None,
) -> ModelPrediction:
    """P(batter gets ≥ threshold total bases)."""

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
    park_tb_factor = park_factor

    pitcher_factor = _pitcher_adjustment(pitcher_stats)

    batter_hand = getattr(batter_season, 'bat_hand', None) if batter_season else None
    pitcher_hand = getattr(pitcher_stats, 'hand', None) if pitcher_stats else None
    platoon_factor = _platoon_adjustment(batter_hand, pitcher_hand)

    adj_slg = blended_slg * park_tb_factor * pitcher_factor * platoon_factor
    adj_slg = max(0.10, min(1.00, adj_slg))

    expected_ab, _ = _estimate_expected_ab(batter_season, game_total)

    lam = expected_ab * adj_slg
    prob = poisson_tail(ticker.threshold, lam)

    sample_confidence = min(1.0, sample_pa / 200)
    edge_confidence   = abs(prob - 0.5) * 2
    data_quality = 0.5
    if pitcher_stats and pitcher_stats.innings >= MIN_IP_FOR_STATS:
        data_quality += 0.25
    if game_total:
        data_quality += 0.25

    confidence = MIN_CONFIDENCE + (MAX_CONFIDENCE - MIN_CONFIDENCE) * (
        0.40 * sample_confidence + 0.30 * edge_confidence + 0.30 * data_quality
    )

    rationale = (
        f"{ticker.player_display} | "
        f"season SLG={season_slg:.3f} (n={sample_pa}PA) | "
        f"park={park_tb_factor:.2f} pitcher_adj={pitcher_factor:.3f} | "
    )
    if batter_hand and pitcher_hand:
        rationale += f"platoon={batter_hand}v{pitcher_hand} adj={platoon_factor:.2f} | "
    if game_total:
        rationale += f"O/U={game_total:.1f} | "
    rationale += f"AB={expected_ab:.1f} λ={lam:.2f} → P(≥{ticker.threshold} TB)={prob:.3f}"

    features = np.array([
        blended_slg, adj_slg, park_tb_factor, pitcher_factor, platoon_factor,
        float(ticker.threshold), float(sample_pa), expected_ab,
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
    game_total:     Optional[float] = None,
) -> ModelPrediction:
    """P(batter hits ≥ threshold home runs).

    v14: Only called for prop_code == "HR" now. HRR gets its own predictor.
    """

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
        rolling_hr_rate = batter_rolling.home_runs / batter_rolling.at_bats
    blended_hr_rate = _blend_season_and_rolling(season_hr_rate, rolling_hr_rate)

    home_team_id = ticker.home_team_id
    park_factor  = PARK_FACTORS.get(home_team_id, 1.00) if home_team_id else 1.00
    park_hr_factor = park_factor ** 1.2

    base_pitcher_factor = _pitcher_adjustment(pitcher_stats)
    pitcher_factor = 1.0 + 0.5 * (base_pitcher_factor - 1.0)

    batter_hand = getattr(batter_season, 'bat_hand', None) if batter_season else None
    pitcher_hand = getattr(pitcher_stats, 'hand', None) if pitcher_stats else None
    base_platoon = _platoon_adjustment(batter_hand, pitcher_hand)
    if base_platoon != 1.0:
        platoon_factor = 1.0 + (base_platoon - 1.0) * 1.3
    else:
        platoon_factor = 1.0

    adj_hr_rate = blended_hr_rate * park_hr_factor * pitcher_factor * platoon_factor
    adj_hr_rate = max(0.001, min(0.20, adj_hr_rate))

    _, expected_pa = _estimate_expected_ab(batter_season, game_total)
    n_trials = int(round(expected_pa))
    n_trials = max(2, min(7, n_trials))

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
    if game_total:
        rationale += f"O/U={game_total:.1f} | "
    rationale += f"adj_rate={adj_hr_rate:.4f} PA={n_trials} → P(≥{ticker.threshold} HR)={prob:.3f}"

    features = np.array([
        blended_hr_rate, adj_hr_rate, park_hr_factor, pitcher_factor, platoon_factor,
        float(ticker.threshold), float(sample_pa), float(n_trials),
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


def predict_hrr(
    ticker:         MLBTicker,
    batter_season:  Optional[BatterSeasonStats],
    batter_rolling: Optional[BatterRollingStats],
    pitcher_stats:  Optional[PitcherSeasonStats],
    game_total:     Optional[float] = None,
) -> ModelPrediction:
    """P(batter records ≥ threshold HRR events) where HRR = HR + Run scored.

    NEW IN v14. Before v14, HRR was routed through predict_home_runs, which
    modeled only HR/PA. HR/PA ≈ 0.032 is nearly 4× smaller than R/PA ≈ 0.125,
    so the raw probabilities at threshold ≥ 2 were near zero — and the
    +0.13 sports calibration offset in consensus_engine pinned the output
    to ~0.13 regardless of threshold. That's the 0.130 avg_prob signature
    observed across all 8 HRR trades in the 2026-04-23 audit.

    Model: HRR is additive — count both HRs and Runs separately.
      rate_per_PA  = HR/PA + R/PA
      λ            = expected_PA × rate_per_PA × park × pitcher × platoon
      P(HRR ≥ k)   = Poisson tail

    Run scoring captures lineup context (teammates driving you in) that
    HR-only models miss. A leadoff hitter on a high-scoring team has a
    high HRR rate even with modest power, because they get on base and
    get driven home.

    Uses expected_PA (not AB) since walks contribute to run-scoring.

    Park factor: uses park_factor^0.6 — runs are less park-sensitive than
    HRs (park_factor^1.2) but more than hits (park_factor^0.5).

    Pitcher adjustment: full strength — run suppression is WHIP-driven,
    which is exactly what _pitcher_adjustment measures.
    """

    # ── Batter HR rate ───────────────────────────────────────────────────
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
        rolling_hr_rate = batter_rolling.home_runs / batter_rolling.at_bats
    blended_hr_rate = _blend_season_and_rolling(season_hr_rate, rolling_hr_rate)

    # ── Batter R rate ─────────────────────────────────────────────────────
    season_r_rate = _get_runs_per_pa(batter_season)
    if season_r_rate is None:
        season_r_rate = LEAGUE_AVG_R_RATE
        r_source = "league_avg"
    else:
        r_source = "data"

    season_r_rate = _blend_to_league(
        season_r_rate, sample_pa, MIN_PA_FOR_STATS, LEAGUE_AVG_R_RATE,
    )

    rolling_r_rate = _get_rolling_runs_per_pa(batter_rolling)
    blended_r_rate = _blend_season_and_rolling(season_r_rate, rolling_r_rate)

    # ── Combined HRR rate ────────────────────────────────────────────────
    combined_rate = blended_hr_rate + blended_r_rate

    # ── Environmental adjustments ────────────────────────────────────────
    home_team_id = ticker.home_team_id
    park_factor  = PARK_FACTORS.get(home_team_id, 1.00) if home_team_id else 1.00
    # Runs are moderately park-sensitive (between hits^0.5 and HR^1.2)
    park_run_factor = park_factor ** 0.6

    # Full pitcher adjustment — WHIP directly measures baserunner suppression,
    # which caps downstream run scoring.
    pitcher_factor = _pitcher_adjustment(pitcher_stats)

    # Platoon dampened 50% — HR platoon matters, R platoon matters less
    # (teammates drive you in regardless of who pitched when you walked)
    batter_hand = getattr(batter_season, 'bat_hand', None) if batter_season else None
    pitcher_hand = getattr(pitcher_stats, 'hand', None) if pitcher_stats else None
    base_platoon = _platoon_adjustment(batter_hand, pitcher_hand)
    platoon_factor = 1.0 + (base_platoon - 1.0) * 0.5

    adj_rate = combined_rate * park_run_factor * pitcher_factor * platoon_factor
    # Sanity bounds: 0.02 floor keeps tail nonzero; 0.40 ceiling caps elite
    # leadoff hitters + Coors + bad pitcher combos
    adj_rate = max(0.02, min(0.40, adj_rate))

    # ── Expected PA (not AB — walks can score runs) ──────────────────────
    _, expected_pa = _estimate_expected_ab(batter_season, game_total)

    # ── Poisson tail ─────────────────────────────────────────────────────
    lam = expected_pa * adj_rate
    prob = poisson_tail(ticker.threshold, lam)

    # ── Confidence ───────────────────────────────────────────────────────
    sample_confidence = min(1.0, sample_pa / 200)
    edge_confidence   = abs(prob - 0.5) * 2
    data_quality = 0.4  # Base lower than HIT/TB because R/PA estimate may be OBP-derived
    if r_source == "data":
        data_quality += 0.15
    if pitcher_stats and pitcher_stats.innings >= MIN_IP_FOR_STATS:
        data_quality += 0.20
    if game_total:
        data_quality += 0.15

    confidence = MIN_CONFIDENCE + (MAX_CONFIDENCE - MIN_CONFIDENCE) * (
        0.40 * sample_confidence + 0.30 * edge_confidence + 0.30 * data_quality
    )

    # ── Rationale ────────────────────────────────────────────────────────
    rationale = (
        f"{ticker.player_display} | "
        f"HR/PA={blended_hr_rate:.4f} R/PA={blended_r_rate:.3f} ({r_source}) "
        f"(n={sample_pa}PA) | "
        f"combined={combined_rate:.3f} | "
        f"park={park_run_factor:.2f} pitcher={pitcher_factor:.3f} "
        f"platoon={platoon_factor:.2f} | "
    )
    if game_total:
        rationale += f"O/U={game_total:.1f} | "
    rationale += (
        f"adj_rate={adj_rate:.3f} PA={expected_pa:.1f} λ={lam:.3f} "
        f"→ P(≥{ticker.threshold} HRR)={prob:.3f}"
    )

    features = np.array([
        blended_hr_rate, blended_r_rate, combined_rate,
        adj_rate, park_run_factor, pitcher_factor, platoon_factor,
        float(ticker.threshold), float(sample_pa), expected_pa,
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


def predict_pitcher_ks(
    ticker:         MLBTicker,
    pitcher_season: Optional[PitcherSeasonStats],
    game_total:     Optional[float] = None,
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
    game_total:     Optional[float] = None,
) -> Optional[ModelPrediction]:
    """Dispatch to the right sub-predictor based on prop_code.

    v14 change: HRR now routes to predict_hrr() (dedicated HR+Run model),
    not predict_home_runs(). HR continues to use predict_home_runs.
    """
    code = ticker.prop_code
    if code == "HIT":
        return predict_hits(ticker, batter_season, batter_rolling, pitcher_stats, game_total)
    if code == "TB":
        return predict_total_bases(ticker, batter_season, batter_rolling, pitcher_stats, game_total)
    if code == "HR":
        return predict_home_runs(ticker, batter_season, batter_rolling, pitcher_stats, game_total)
    if code == "HRR":
        return predict_hrr(ticker, batter_season, batter_rolling, pitcher_stats, game_total)
    if code == "KS":
        return predict_pitcher_ks(ticker, pitcher_stats, game_total)
    logger.debug("no model for prop_code=%s", code)
    return None


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=" * 78)
    print("MLB hit-rate model — self-test (v14: HRR predictor)")
    print("=" * 78)

    # ── Test _get_runs_per_pa with various input shapes ──
    print("\n── _get_runs_per_pa() input resilience ──")

    class MockWithRunsScored:
        plate_apps = 500
        at_bats = 450
        walks = 50
        avg = 0.270
        runs_scored = 75  # direct field
        home_runs = 12

    class MockWithRunsShort:
        plate_apps = 500
        at_bats = 450
        walks = 50
        avg = 0.270
        runs = 75  # alt field name
        home_runs = 12

    class MockWithOBP:
        plate_apps = 500
        at_bats = 450
        walks = 50
        avg = 0.270
        obp = 0.360
        home_runs = 12

    class MockWithNothing:
        plate_apps = 500
        at_bats = 450
        walks = 50
        avg = 0.270
        home_runs = 12

    class MockEmpty:
        plate_apps = 0
        at_bats = 0
        walks = 0
        avg = 0
        home_runs = 0

    r1 = _get_runs_per_pa(MockWithRunsScored())
    r2 = _get_runs_per_pa(MockWithRunsShort())
    r3 = _get_runs_per_pa(MockWithOBP())
    r4 = _get_runs_per_pa(MockWithNothing())
    r5 = _get_runs_per_pa(MockEmpty())
    r6 = _get_runs_per_pa(None)

    print(f"  With runs_scored field:  {r1:.4f}  (expected: 0.1500)")
    print(f"  With runs field:         {r2:.4f}  (expected: 0.1500)")
    print(f"  With obp field:          {r3:.4f}  (expected: 0.1080)")
    print(f"  With nothing (AVG+BB):   {r4:.4f}  (expected: ~0.1055)")
    print(f"  Empty stats:             {r5}       (expected: None)")
    print(f"  None input:              {r6}       (expected: None)")

    assert abs(r1 - 0.150) < 0.001, "direct runs_scored failed"
    assert abs(r2 - 0.150) < 0.001, "direct runs failed"
    assert abs(r3 - 0.108) < 0.001, "OBP-derived failed"
    assert r4 is not None and 0.08 < r4 < 0.12, f"AVG+BB fallback failed: {r4}"
    assert r5 is None, "empty should return None"
    assert r6 is None, "None should return None"
    print("  ✓ All _get_runs_per_pa assertions passed")

    # ── Test predict_hrr — threshold sensitivity ──
    print("\n── predict_hrr() threshold sensitivity (league-avg batter) ──")
    print("  This is the MAIN test: different thresholds must give different probs.")
    print("  Pre-v14 bug: all thresholds returned ~0.001 raw → 0.13 after calibration.\n")

    # Minimal fake MLBTicker — just the fields predict_hrr reads
    class MockTicker:
        def __init__(self, threshold):
            self.threshold = threshold
            self.home_team_id = 119  # LAD
            self.player_display = "TestPlayer"

    class MockBatter:
        plate_apps = 500
        at_bats = 450
        walks = 50
        avg = 0.270
        home_runs = 15
        runs_scored = 65  # R/PA = 0.13, HR/PA = 0.030
        bat_hand = "R"

    class MockPitcher:
        innings = 50.0
        hits_allowed = 45
        whip = 1.20
        k_per_9 = 9.0
        hand = "L"

    probs = {}
    for k in [1, 2, 3, 5]:
        p = predict_hrr(MockTicker(k), MockBatter(), None, MockPitcher(), game_total=8.5)
        probs[k] = p.prob_yes
        print(f"  Threshold ≥{k}: P={p.prob_yes:.4f} | λ={p.lam_or_p:.3f} | conf={p.confidence:.3f}")
        print(f"    {p.rationale}")

    # Threshold-monotonicity assertion
    assert probs[1] > probs[2] > probs[3] > probs[5], (
        f"Probabilities must decrease with threshold: {probs}"
    )
    # Sanity: probs should span a real range, not all be ~0.13
    assert probs[1] > 0.30, f"P(≥1) should be >30% for this batter, got {probs[1]}"
    assert probs[5] < 0.01, f"P(≥5) should be <1%, got {probs[5]}"
    print("\n  ✓ Threshold-monotonicity passed (the 0.130 bug is dead)")

    # ── Test elite leadoff vs 9-hole hitter ──
    print("\n── predict_hrr() context sensitivity (elite leadoff vs weak 9-hole) ──")

    class EliteBatter:
        plate_apps = 600
        at_bats = 520
        walks = 65
        avg = 0.310
        home_runs = 28
        runs_scored = 110  # R/PA = 0.183, HR/PA = 0.047
        bat_hand = "L"

    class WeakBatter:
        plate_apps = 400
        at_bats = 370
        walks = 25
        avg = 0.220
        home_runs = 5
        runs_scored = 35  # R/PA = 0.088, HR/PA = 0.013
        bat_hand = "R"

    elite = predict_hrr(MockTicker(2), EliteBatter(), None, MockPitcher(), game_total=11.0)
    weak  = predict_hrr(MockTicker(2), WeakBatter(), None, MockPitcher(), game_total=7.0)

    print(f"  Elite leadoff (high R/PA + high O/U):  P(≥2 HRR) = {elite.prob_yes:.3f}")
    print(f"  Weak 9-hole (low R/PA + low O/U):      P(≥2 HRR) = {weak.prob_yes:.3f}")
    assert elite.prob_yes > weak.prob_yes * 2, "elite must be at least 2× weak"
    print("  ✓ Context sensitivity confirmed")

    # ── Compare to what pre-v14 predict_home_runs returned for same inputs ──
    print("\n── v14 vs v13 for the same HRR market ──")

    t3 = MockTicker(3)
    old_hr = predict_home_runs(t3, MockBatter(), None, MockPitcher(), game_total=8.5)
    new_hrr = predict_hrr(t3, MockBatter(), None, MockPitcher(), game_total=8.5)

    print(f"  OLD (predict_home_runs for HRR ≥3):  P={old_hr.prob_yes:.4f}")
    print(f"  NEW (predict_hrr for HRR ≥3):        P={new_hrr.prob_yes:.4f}")
    print(f"  After +0.13 sports calibration offset:")
    print(f"    OLD calibrated: {min(0.95, old_hr.prob_yes + 0.13):.4f}  ← this is the 0.13 bug")
    print(f"    NEW calibrated: {min(0.95, new_hrr.prob_yes + 0.13):.4f}  ← threshold-aware")

    print("\n" + "=" * 78)
    print("Done. HRR predictor verified.")
    print("=" * 78)
    sys.exit(0)