"""
shared/nba_props_model.py

Predicts NBA player prop outcomes using Poisson/Binomial distributions.
Mirrors shared/mlb_hit_model.py architecture.

Model logic:
  Points:         Poisson(λ = blended_ppg)
  Rebounds:        Poisson(λ = blended_rpg)
  Assists:         Poisson(λ = blended_apg)
  3-Pointers:     Poisson(λ = blended_3pm_pg)
  Steals:          Poisson(λ = blended_stl_pg)
  Blocks:          Poisson(λ = blended_blk_pg)

Blending: 70% season average + 30% rolling (last 10 games).
Confidence scales with games played — more data = higher confidence.
"""

import logging
import math
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class NBAPrediction:
    """Prediction output for a single NBA player prop."""
    prob_yes:    float   # P(stat >= threshold)
    confidence:  float   # 0.0–1.0
    rationale:   str


def _poisson_cdf_ge(lam: float, k: int) -> float:
    """P(X >= k) for Poisson(λ), computed as 1 - P(X <= k-1)."""
    if lam <= 0:
        return 0.0 if k > 0 else 1.0
    if k <= 0:
        return 1.0

    # P(X <= k-1) = sum of e^(-λ) * λ^i / i! for i=0..k-1
    cdf = 0.0
    log_lam = math.log(lam) if lam > 0 else -float("inf")
    for i in range(k):
        log_pmf = -lam + i * log_lam - math.lgamma(i + 1)
        cdf += math.exp(log_pmf)

    return max(0.0, min(1.0, 1.0 - cdf))


def _blend(season_val: float, rolling_val: Optional[float], season_weight: float = 0.70) -> float:
    """Blend season and rolling averages."""
    if rolling_val is None:
        return season_val
    return season_weight * season_val + (1.0 - season_weight) * rolling_val


def predict_nba_prop(
    parsed,          # NBATicker from nba_ticker_parser
    season_stats,    # PlayerStats from nba_stats_fetcher
    rolling_stats,   # RollingStats (optional)
) -> Optional[NBAPrediction]:
    """
    Predict P(player achieves >= threshold for given prop).

    Parameters
    ----------
    parsed : NBATicker
        Parsed ticker with prop_code, threshold, player info.
    season_stats : PlayerStats
        Season averages.
    rolling_stats : RollingStats or None
        Rolling averages over recent games.

    Returns
    -------
    NBAPrediction or None if the prop type isn't supported.
    """
    if season_stats is None:
        return None

    threshold = parsed.threshold
    prop_code = parsed.prop_code
    gp = season_stats.games_played

    # ── Confidence: scales with sample size ────────────────────────────────
    # 10 games = 0.40, 30 games = 0.55, 60+ games = 0.70
    base_confidence = min(0.70, 0.30 + 0.007 * gp)

    # ── Points (PTS) ──────────────────────────────────────────────────────
    if prop_code == "PTS":
        season_ppg = season_stats.points_pg
        rolling_ppg = rolling_stats.points_pg if rolling_stats else None
        lam = _blend(season_ppg, rolling_ppg)

        prob = _poisson_cdf_ge(lam, threshold)

        # Points have higher variance than Poisson suggests (fouls, blowouts).
        # Adjust toward 0.5 slightly for extreme predictions.
        prob = _variance_adjust(prob, factor=0.92)

        return NBAPrediction(
            prob_yes=round(prob, 4),
            confidence=round(base_confidence, 2),
            rationale=(
                f"season={season_ppg:.1f}ppg | "
                f"rolling={rolling_ppg:.1f if rolling_ppg else 'N/A'}ppg | "
                f"λ={lam:.1f} → P(≥{threshold})={prob:.3f}"
            ),
        )

    # ── Rebounds (REB) ────────────────────────────────────────────────────
    if prop_code == "REB":
        season_rpg = season_stats.rebounds_pg
        rolling_rpg = rolling_stats.rebounds_pg if rolling_stats else None
        lam = _blend(season_rpg, rolling_rpg)

        prob = _poisson_cdf_ge(lam, threshold)

        return NBAPrediction(
            prob_yes=round(prob, 4),
            confidence=round(base_confidence, 2),
            rationale=(
                f"season={season_rpg:.1f}rpg | "
                f"rolling={rolling_rpg:.1f if rolling_rpg else 'N/A'}rpg | "
                f"λ={lam:.1f} → P(≥{threshold})={prob:.3f}"
            ),
        )

    # ── Assists (AST) ─────────────────────────────────────────────────────
    if prop_code == "AST":
        season_apg = season_stats.assists_pg
        rolling_apg = rolling_stats.assists_pg if rolling_stats else None
        lam = _blend(season_apg, rolling_apg)

        prob = _poisson_cdf_ge(lam, threshold)

        return NBAPrediction(
            prob_yes=round(prob, 4),
            confidence=round(base_confidence, 2),
            rationale=(
                f"season={season_apg:.1f}apg | "
                f"rolling={rolling_apg:.1f if rolling_apg else 'N/A'}apg | "
                f"λ={lam:.1f} → P(≥{threshold})={prob:.3f}"
            ),
        )

    # ── 3-Pointers (3PT) ──────────────────────────────────────────────────
    if prop_code == "3PT":
        season_3pm = season_stats.three_pm_pg
        rolling_3pm = rolling_stats.three_pm_pg if rolling_stats else None
        lam = _blend(season_3pm, rolling_3pm)

        prob = _poisson_cdf_ge(lam, threshold)

        return NBAPrediction(
            prob_yes=round(prob, 4),
            confidence=round(base_confidence, 2),
            rationale=(
                f"season={season_3pm:.1f} 3pm/g | "
                f"rolling={rolling_3pm:.1f if rolling_3pm else 'N/A'} | "
                f"λ={lam:.1f} → P(≥{threshold})={prob:.3f}"
            ),
        )

    # ── Steals (STL) ──────────────────────────────────────────────────────
    if prop_code == "STL":
        season_stl = season_stats.steals_pg
        rolling_stl = rolling_stats.steals_pg if rolling_stats else None
        lam = _blend(season_stl, rolling_stl)

        prob = _poisson_cdf_ge(lam, threshold)

        return NBAPrediction(
            prob_yes=round(prob, 4),
            confidence=round(base_confidence + 0.05, 2),  # small bonus for rarity
            rationale=(
                f"season={season_stl:.1f}spg | "
                f"λ={lam:.1f} → P(≥{threshold})={prob:.3f}"
            ),
        )

    # ── Blocks (BLK) ─────────────────────────────────────────────────────
    if prop_code == "BLK":
        season_blk = season_stats.blocks_pg
        rolling_blk = rolling_stats.blocks_pg if rolling_stats else None
        lam = _blend(season_blk, rolling_blk)

        prob = _poisson_cdf_ge(lam, threshold)

        return NBAPrediction(
            prob_yes=round(prob, 4),
            confidence=round(base_confidence + 0.05, 2),
            rationale=(
                f"season={season_blk:.1f}bpg | "
                f"λ={lam:.1f} → P(≥{threshold})={prob:.3f}"
            ),
        )

    logger.debug("[nba_model] unsupported prop code: %s", prop_code)
    return None


def _variance_adjust(prob: float, factor: float = 0.92) -> float:
    """
    Pull extreme predictions toward 0.5 to account for real-world variance
    that Poisson underestimates (blowouts, foul trouble, early exits).

    factor=0.92 means a raw 0.95 becomes 0.92*0.95 + 0.08*0.5 = 0.914
    """
    return factor * prob + (1.0 - factor) * 0.5


# ── Self-test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from shared.nba_stats_fetcher import PlayerStats, RollingStats
    from shared.nba_ticker_parser import NBATicker

    print("=" * 78)
    print("NBA props model — self-test (synthetic data)")
    print("=" * 78)

    # Synthetic Jayson Tatum stats
    tatum_season = PlayerStats(
        player_id=1628369, games_played=65, minutes_pg=36.2,
        points_pg=27.3, rebounds_pg=8.5, assists_pg=4.8,
        steals_pg=1.1, blocks_pg=0.6,
        three_pm_pg=3.1, three_pa_pg=8.5, three_pct=0.365,
        fga_pg=19.8, fta_pg=5.2, turnovers_pg=2.8, usage_rate=30.5,
    )
    tatum_rolling = RollingStats(
        n_games=10, points_pg=29.5, rebounds_pg=9.0,
        assists_pg=5.2, three_pm_pg=3.4, steals_pg=1.2,
        blocks_pg=0.7, minutes_pg=37.0,
    )

    tests = [
        ("PTS", "points", 25),
        ("PTS", "points", 30),
        ("PTS", "points", 35),
        ("REB", "rebounds", 8),
        ("REB", "rebounds", 12),
        ("AST", "assists", 5),
        ("3PT", "three_pointers", 3),
        ("3PT", "three_pointers", 5),
        ("STL", "steals", 2),
        ("BLK", "blocks", 2),
    ]

    for code, name, threshold in tests:
        ticker = NBATicker(
            raw_ticker=f"KXNBA{code}-TEST",
            prop_prefix=f"KXNBA{code}",
            prop_code=code,
            prop_name=name,
            game_date="2026-04-10",
            away_team_code="BOS",
            home_team_code="CLE",
            player_team_code="BOS",
            player_first="J",
            player_last="TATUM",
            player_jersey="0",
            threshold=threshold,
        )

        pred = predict_nba_prop(ticker, tatum_season, tatum_rolling)
        if pred:
            print(f"  {name:15s} ≥{threshold:3d} → P={pred.prob_yes:.3f}  conf={pred.confidence:.2f}")
        else:
            print(f"  {name:15s} ≥{threshold:3d} → UNSUPPORTED")

    print(f"\n{'=' * 78}")
    print("If probabilities vary per threshold, model is working.")
    print(f"{'=' * 78}")