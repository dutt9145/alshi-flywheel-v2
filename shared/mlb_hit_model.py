"""
shared/mlb_hit_model.py  (v13 — dynamic AB + WHIP + BAA pitcher adjustment)

Changes vs v12.2:
  1. Dynamic expected AB: Replaced fixed DEFAULT_EXPECTED_AB=4 with
     _estimate_expected_ab() that adjusts for:
       - Player walk rate (high OBP batters get fewer AB per PA)
       - Game total line when available (high O/U → more PA for everyone)
     A leadoff hitter in a 12-run game gets ~5.2 AB vs ~3.3 for a
     9-hole hitter in a 6.5 game. This single variable was causing
     the largest prediction errors.
     
  2. WHIP-based pitcher adjustment: Replaced K/9-only proxy with a
     combined WHIP + BAA adjustment. K/9 measures strikeouts, not hit
     suppression. A high-K pitcher can still give up hits (high WHIP).
     Now uses:
       - BAA (batting avg against) = hits_allowed / batters_faced_approx
       - WHIP for overall baserunner pressure
       - K/9 retained as secondary signal
     
  3. All prediction functions now accept optional game_total parameter
     for dynamic AB estimation.

  All other logic (platoon splits, park factors, stat blending) unchanged.
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
LEAGUE_AVG_WHIP      = 1.28      # 2024-2025 MLB average WHIP
LEAGUE_AVG_GAME_TOTAL = 8.5      # Average Vegas O/U
LEAGUE_AVG_PA_PER_GAME = 38.5    # Average team PA per 9 innings

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
    # Base PA from game total
    if game_total and game_total > 0:
        # Scale team PA linearly with game total
        # O/U 8.5 → 38.5 team PA, O/U 12 → ~54 team PA, O/U 6.5 → ~29 team PA
        team_pa = LEAGUE_AVG_PA_PER_GAME * (game_total / LEAGUE_AVG_GAME_TOTAL)
        base_pa = team_pa / 9.0
    else:
        base_pa = DEFAULT_EXPECTED_PA
    
    # Clamp to reasonable range
    base_pa = max(3.0, min(6.0, base_pa))
    
    # Convert PA to AB using player's walk rate
    if batter_season and batter_season.plate_apps > 30:
        walk_rate = batter_season.walks / batter_season.plate_apps
        # Also subtract HBP approximation (~1% of PA)
        ab_fraction = 1.0 - walk_rate - 0.01
        ab_fraction = max(0.80, min(0.96, ab_fraction))
    else:
        ab_fraction = 0.92  # League average
    
    expected_pa = base_pa
    expected_ab = base_pa * ab_fraction
    
    return expected_ab, expected_pa


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
    
    v13: Uses WHIP and BAA (batting average against) instead of K/9 alone.
    K/9 measures strikeouts, not hit suppression. A high-K pitcher can still
    give up lots of hits (high WHIP). 
    
    Components:
      - BAA (batting average against): direct measure of hit-allowing tendency
        BAA below league avg → pitcher suppresses hits → reduce batter AVG
      - WHIP: measures total baserunners per inning
        High WHIP → pitcher gives up hits+walks → boost batter AVG
      - K/9: retained as secondary signal (high-K pitchers create more outs)
    
    Returns multiplicative factor (0.85 to 1.15).
    Floor at 0.85, ceiling at 1.15.
    """
    if pitcher_stats is None:
        return 1.0
    
    adjustments = []
    
    # BAA-based adjustment (strongest signal for hit prediction)
    if pitcher_stats.innings > 0 and pitcher_stats.hits_allowed > 0:
        # Approximate batters faced: innings * ~4.3 batters per inning
        # (3 outs + walks/hits/errors ≈ 4.3 BF/IP on average)
        batters_faced = pitcher_stats.innings * 4.3
        if batters_faced > 0:
            baa = pitcher_stats.hits_allowed / batters_faced
            baa = _blend_to_league(baa, int(pitcher_stats.innings), MIN_IP_FOR_STATS, LEAGUE_AVG_AVG)
            # Ratio vs league avg: BAA 0.220 vs avg 0.243 → factor 0.91
            baa_factor = baa / LEAGUE_AVG_AVG
            adjustments.append(('baa', baa_factor, 0.50))  # 50% weight
    
    # WHIP-based adjustment
    if pitcher_stats.whip > 0 and pitcher_stats.innings >= MIN_IP_FOR_STATS:
        whip = pitcher_stats.whip
        # WHIP 1.10 vs avg 1.28 → pitcher suppresses baserunners → factor 0.86
        whip_factor = whip / LEAGUE_AVG_WHIP
        adjustments.append(('whip', whip_factor, 0.30))  # 30% weight
    
    # K/9-based adjustment (secondary — high K = fewer balls in play)
    if pitcher_stats.k_per_9 > 0:
        k9 = pitcher_stats.k_per_9
        k9 = _blend_to_league(k9, int(pitcher_stats.innings), MIN_IP_FOR_STATS, LEAGUE_AVG_K9)
        delta = k9 - LEAGUE_AVG_K9
        k9_factor = 1.0 - 0.004 * delta  # Reduced from 0.006 — K/9 is weaker signal for hits
        adjustments.append(('k9', k9_factor, 0.20))  # 20% weight
    
    if not adjustments:
        return 1.0
    
    # Weighted average of all adjustment factors
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

    # Park factor
    home_team_id = ticker.home_team_id
    park_factor  = PARK_FACTORS.get(home_team_id, 1.00) if home_team_id else 1.00
    park_hit_factor = math.sqrt(park_factor)

    # v13: WHIP + BAA pitcher adjustment (replaces K/9-only)
    pitcher_factor = _pitcher_adjustment(pitcher_stats)

    # Platoon adjustment
    batter_hand = getattr(batter_season, 'bat_hand', None) if batter_season else None
    pitcher_hand = getattr(pitcher_stats, 'hand', None) if pitcher_stats else None
    platoon_factor = _platoon_adjustment(batter_hand, pitcher_hand)

    # Final adjusted AVG
    adj_avg = blended_avg * park_hit_factor * pitcher_factor * platoon_factor
    adj_avg = max(0.05, min(0.50, adj_avg))

    # v13: Dynamic expected AB based on walk rate + game total
    expected_ab, expected_pa = _estimate_expected_ab(batter_season, game_total)

    # P(≥ threshold hits) from Binomial(n=AB, p=adj_avg)
    n_ab = int(round(expected_ab))
    n_ab = max(2, min(7, n_ab))  # Sanity bounds
    prob = binomial_tail(ticker.threshold, n_ab, adj_avg)

    # Confidence scales with sample depth, distance from 0.5, and data quality
    sample_confidence = min(1.0, sample_pa / 200)
    edge_confidence   = abs(prob - 0.5) * 2
    # v13: Boost confidence when we have pitcher data + game total
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

    # v13: WHIP + BAA pitcher adjustment
    pitcher_factor = _pitcher_adjustment(pitcher_stats)

    batter_hand = getattr(batter_season, 'bat_hand', None) if batter_season else None
    pitcher_hand = getattr(pitcher_stats, 'hand', None) if pitcher_stats else None
    platoon_factor = _platoon_adjustment(batter_hand, pitcher_hand)

    adj_slg = blended_slg * park_tb_factor * pitcher_factor * platoon_factor
    adj_slg = max(0.10, min(1.00, adj_slg))

    # v13: Dynamic expected AB
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
        rolling_hr_rate = batter_rolling.home_runs / batter_rolling.at_bats
    blended_hr_rate = _blend_season_and_rolling(season_hr_rate, rolling_hr_rate)

    home_team_id = ticker.home_team_id
    park_factor  = PARK_FACTORS.get(home_team_id, 1.00) if home_team_id else 1.00
    park_hr_factor = park_factor ** 1.2

    # v13: WHIP + BAA pitcher adjustment, half-strength for HRs
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

    # v13: Dynamic expected PA
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
    
    v13: All sub-predictors now accept optional game_total for dynamic AB.
    """
    code = ticker.prop_code
    if code == "HIT":
        return predict_hits(ticker, batter_season, batter_rolling, pitcher_stats, game_total)
    if code == "TB":
        return predict_total_bases(ticker, batter_season, batter_rolling, pitcher_stats, game_total)
    if code in ("HR", "HRR"):
        return predict_home_runs(ticker, batter_season, batter_rolling, pitcher_stats, game_total)
    if code == "KS":
        return predict_pitcher_ks(ticker, pitcher_stats, game_total)
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
    print("MLB hit-rate model — self-test (v13: dynamic AB + WHIP)")
    print("=" * 78)

    # Test dynamic AB estimation
    print("\n── Dynamic AB estimation ──")
    
    # Mock a high-OBP batter (walks a lot)
    class MockBatter:
        plate_apps = 200
        walks = 40  # 20% walk rate
    
    class MockBatterLowBB:
        plate_apps = 200
        walks = 10  # 5% walk rate
    
    ab_high_bb, pa_high_bb = _estimate_expected_ab(MockBatter(), game_total=None)
    ab_low_bb, pa_low_bb = _estimate_expected_ab(MockBatterLowBB(), game_total=None)
    ab_high_ou, pa_high_ou = _estimate_expected_ab(MockBatter(), game_total=12.0)
    ab_low_ou, pa_low_ou = _estimate_expected_ab(MockBatter(), game_total=6.5)
    
    print(f"  High BB (20%), avg O/U:  AB={ab_high_bb:.2f}, PA={pa_high_bb:.2f}")
    print(f"  Low BB (5%), avg O/U:    AB={ab_low_bb:.2f}, PA={pa_low_bb:.2f}")
    print(f"  High BB, O/U=12.0:       AB={ab_high_ou:.2f}, PA={pa_high_ou:.2f}")
    print(f"  High BB, O/U=6.5:        AB={ab_low_ou:.2f}, PA={pa_low_ou:.2f}")
    
    # Verify the range makes sense
    assert ab_high_bb < ab_low_bb, "High-walk batter should have fewer AB"
    assert pa_high_ou > pa_low_ou, "High O/U should mean more PA"
    print("  ✓ All dynamic AB assertions passed")

    # Test pitcher adjustment
    print("\n── WHIP-based pitcher adjustment ──")
    
    class MockAcePitcher:
        innings = 80.0
        hits_allowed = 50
        whip = 0.95
        k_per_9 = 11.5
        hand = "R"
    
    class MockBadPitcher:
        innings = 60.0
        hits_allowed = 80
        whip = 1.65
        k_per_9 = 6.0
        hand = "L"
    
    ace_adj = _pitcher_adjustment(MockAcePitcher())
    bad_adj = _pitcher_adjustment(MockBadPitcher())
    none_adj = _pitcher_adjustment(None)
    
    print(f"  Ace (WHIP=0.95, K/9=11.5): {ace_adj:.3f}")
    print(f"  Bad (WHIP=1.65, K/9=6.0):  {bad_adj:.3f}")
    print(f"  Unknown:                    {none_adj:.3f}")
    
    assert ace_adj < 1.0, "Ace pitcher should suppress hits"
    assert bad_adj > 1.0, "Bad pitcher should boost hits"
    assert none_adj == 1.0, "Unknown pitcher should be neutral"
    print("  ✓ All pitcher adjustment assertions passed")

    print("\n" + "=" * 78)
    print("Done. Dynamic AB + WHIP/BAA pitcher model verified.")
    print("=" * 78)