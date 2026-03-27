"""
shared/bayesian_poly_model.py  (v3 — feature dimension mismatch guard)

Changes vs v2:
  1. predict() now checks feature vector length against the trained pipeline's
     expected input shape before inference. If a mismatch is detected (e.g.
     WeatherBot grew from 7 → 9 features after NOAA integration), the stored
     model resets gracefully to bayes_only mode rather than crashing sklearn.
  2. Everything else unchanged from v2 (Supabase persistence, disk cache, etc.)
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.special import expit  # sigmoid
from scipy.stats import beta as beta_dist
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from config.settings import (
    POLY_DEGREE, BETA_ALPHA_PRIOR, BETA_BETA_PRIOR, MIN_TRAINING_SAMPLES
)

logger = logging.getLogger(__name__)


class BayesianPolyModel:
    """
    Sector-level probability estimator.

    Parameters
    ----------
    sector   : human-readable name used for logging and persistence
    degree   : polynomial feature degree (default from settings)
    alpha0   : Beta prior α — pseudo-successes before any data
    beta0    : Beta prior β — pseudo-failures  before any data
    """

    def __init__(
        self,
        sector: str,
        degree: int   = POLY_DEGREE,
        alpha0: float = BETA_ALPHA_PRIOR,
        beta0:  float = BETA_BETA_PRIOR,
    ):
        self.sector  = sector
        self.degree  = degree
        self.alpha0  = alpha0
        self.beta0   = beta0
        self.trained = False

        # Bayesian running tallies (updated whenever we see a resolved contract)
        self.alpha_post = alpha0
        self.beta_post  = beta0
        self.n_obs      = 0

        # sklearn pipeline: poly → scale → logistic
        self._pipeline = Pipeline([
            ("poly",  PolynomialFeatures(degree=degree, include_bias=False)),
            ("scale", StandardScaler()),
            ("clf",   LogisticRegression(
                max_iter=1000,
                C=1.0,
                class_weight="balanced",
                solver="lbfgs",
            )),
        ])

        # Training buffers
        self._X_buf: list = []
        self._y_buf: list = []

    # ── Training ───────────────────────────────────────────────────────────────

    def add_observation(self, features: np.ndarray, resolved_yes: bool) -> None:
        """
        Feed one resolved contract into the training buffer and update the
        Bayesian posterior immediately (online update).
        """
        self._X_buf.append(features.copy())
        self._y_buf.append(int(resolved_yes))
        self.n_obs += 1

        # Online Bayesian update of Beta posterior
        if resolved_yes:
            self.alpha_post += 1
        else:
            self.beta_post += 1

        # Retrain sklearn pipeline if we have enough data
        if self.n_obs >= MIN_TRAINING_SAMPLES:
            self._fit_pipeline()

    def _fit_pipeline(self) -> None:
        X = np.array(self._X_buf)
        y = np.array(self._y_buf)
        # Only train if we have at least one example of each class
        if len(np.unique(y)) < 2:
            logger.warning("[%s] Only one class in training data, skipping ML fit",
                           self.sector)
            return
        self._pipeline.fit(X, y)
        self.trained = True
        logger.info("[%s] Model re-trained on %d samples", self.sector, len(y))

    # ── Inference ──────────────────────────────────────────────────────────────

    def predict(self, features: np.ndarray) -> dict:
        """
        Return a probability dict:
        {
            "prob"      : float,   # posterior P(YES)
            "lower"     : float,   # 90 % CI lower
            "upper"     : float,   # 90 % CI upper
            "confidence": float,   # 0–1, derived from CI width
            "method"    : str,     # 'ml+bayes' | 'bayes_only'
        }
        """
        # ── Feature dimension guard (NEW in v3) ────────────────────────────────
        # If the stored pipeline was trained on a different feature count (e.g.
        # WeatherBot grew from 7 → 9 features after NOAA integration), reset the
        # ML model gracefully rather than crashing sklearn with a shape mismatch.
        if self.trained:
            try:
                expected = self._pipeline.named_steps["poly"].n_features_in_
                if features.shape[0] != expected:
                    logger.warning(
                        "[%s] Feature dim mismatch: got %d, expected %d — "
                        "resetting ML model, falling back to bayes_only",
                        self.sector, features.shape[0], expected,
                    )
                    self.trained = False
                    self._X_buf  = []
                    self._y_buf  = []
                    self.n_obs   = 0
            except Exception as e:
                logger.warning("[%s] Dimension check failed: %s", self.sector, e)
        # ── End guard ──────────────────────────────────────────────────────────

        bayes_prior = self.alpha_post / (self.alpha_post + self.beta_post)

        if not self.trained:
            # Pure Bayesian: use Beta posterior credible interval
            ci_lo, ci_hi = beta_dist.interval(
                0.90, self.alpha_post, self.beta_post
            )
            width = ci_hi - ci_lo
            confidence = max(0.0, 1.0 - width)  # wide CI → low confidence
            return {
                "prob":       bayes_prior,
                "lower":      ci_lo,
                "upper":      ci_hi,
                "confidence": confidence,
                "method":     "bayes_only",
            }

        # ML probability
        X = features.reshape(1, -1)
        ml_prob = float(self._pipeline.predict_proba(X)[0][1])

        # Blend ML with Bayesian prior
        # Weight toward ML as we accumulate more data
        confidence_weight = min(self.n_obs / 100.0, 0.85)
        posterior = confidence_weight * ml_prob + (1 - confidence_weight) * bayes_prior

        # Approximate CI from Bayesian posterior width (shrinks as n grows)
        ci_lo, ci_hi = beta_dist.interval(
            0.90, self.alpha_post, self.beta_post
        )
        # Shift CI to be centred on blended posterior
        half_width = (ci_hi - ci_lo) / 2
        ci_lo = max(0.01, posterior - half_width)
        ci_hi = min(0.99, posterior + half_width)
        width  = ci_hi - ci_lo
        confidence = max(0.0, 1.0 - width)

        return {
            "prob":       round(posterior, 4),
            "lower":      round(ci_lo, 4),
            "upper":      round(ci_hi, 4),
            "confidence": round(confidence, 4),
            "method":     "ml+bayes",
        }

    # ── Calibration ───────────────────────────────────────────────────────────

    def brier_score(self) -> Optional[float]:
        """Brier score on the training buffer (in-sample, for monitoring)."""
        if not self.trained or len(self._y_buf) < 10:
            return None
        X = np.array(self._X_buf)
        y = np.array(self._y_buf)
        probs = self._pipeline.predict_proba(X)[:, 1]
        return float(np.mean((probs - y) ** 2))

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, directory: str = "models") -> None:
        """
        Save model to local disk (fast cache) AND Supabase (persistent).

        Local disk is fast but wiped on every Railway restart/redeploy.
        Supabase copy survives indefinitely and is loaded on cold start.
        """
        from shared.calibration_logger import save_model_state

        # Serialize once, use for both destinations
        blob = pickle.dumps(self)

        # 1. Local disk — fast cache for the current deploy session
        Path(directory).mkdir(exist_ok=True)
        path = Path(directory) / f"{self.sector}_model.pkl"
        try:
            with open(path, "wb") as f:
                f.write(blob)
            logger.info("[%s] Model saved to disk (%d obs)", self.sector, self.n_obs)
        except Exception as e:
            logger.warning("[%s] Disk save failed: %s", self.sector, e)

        # 2. Supabase — persistent across restarts and redeploys
        try:
            save_model_state(self.sector, blob)
            logger.info("[%s] Model saved to Supabase (%d obs)", self.sector, self.n_obs)
        except Exception as e:
            logger.warning("[%s] Supabase model save failed (disk copy still valid): %s",
                           self.sector, e)

    @classmethod
    def load(cls, sector: str, directory: str = "models") -> "BayesianPolyModel":
        """
        Load model using a three-tier fallback:
          1. Local disk  — fastest, used within the same deploy session
          2. Supabase    — persistent, survives Railway restarts/redeploys
          3. Fresh model — cold start when no state exists yet

        This fixes the core issue where Railway's ephemeral filesystem
        wiped all trained model state on every restart, forcing our_p=0.500.
        """
        from shared.calibration_logger import load_model_state

        # Tier 1: local disk (fastest)
        path = Path(directory) / f"{sector}_model.pkl"
        if path.exists():
            try:
                with open(path, "rb") as f:
                    model = pickle.load(f)
                logger.info("[%s] Model loaded from disk (%d obs)", sector, model.n_obs)
                return model
            except Exception as e:
                logger.warning("[%s] Disk load failed, trying Supabase: %s", sector, e)

        # Tier 2: Supabase (persistent across deploys)
        try:
            blob = load_model_state(sector)
            if blob:
                model = pickle.loads(blob)
                logger.info("[%s] Model loaded from Supabase (%d obs)", sector, model.n_obs)
                # Warm the local disk cache for this session
                try:
                    Path(directory).mkdir(exist_ok=True)
                    with open(path, "wb") as f:
                        f.write(blob)
                except Exception:
                    pass
                return model
        except Exception as e:
            logger.warning("[%s] Supabase model load failed: %s", sector, e)

        # Tier 3: fresh model — no state exists yet
        logger.info("[%s] No saved model found anywhere, starting fresh", sector)
        return cls(sector=sector)