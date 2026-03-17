"""
shared/bayesian_poly_model.py

The heart of the flywheel.  Each sector bot instantiates one of these.

Pipeline
--------
1. Raw sector features arrive as a 1-D numpy array.
2. Polynomial features of degree-N are computed.
3. A logistic regression is fitted on historical resolved contracts
   (X = poly features, y = 1 if YES resolved, 0 if NO).
4. The ML probability is blended with a Beta-Binomial Bayesian prior
   using a data-confidence weight.
5. The posterior probability is returned alongside a calibrated
   standard-error estimate.

Why polynomial + Bayesian?
--------------------------
Polynomial features capture non-linear relationships (e.g. CPI at extreme
values matters differently than CPI near the mean).  The Bayesian prior
prevents overconfidence on thin data — when we have few resolved contracts
the prior pulls the estimate toward 0.5, which is exactly what we want
to avoid reckless betting.
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
        degree: int  = POLY_DEGREE,
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
        Path(directory).mkdir(exist_ok=True)
        path = Path(directory) / f"{self.sector}_model.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("[%s] Model saved to %s", self.sector, path)

    @classmethod
    def load(cls, sector: str, directory: str = "models") -> "BayesianPolyModel":
        path = Path(directory) / f"{sector}_model.pkl"
        if not path.exists():
            logger.info("[%s] No saved model found, starting fresh", sector)
            return cls(sector=sector)
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info("[%s] Model loaded (%d obs)", sector, model.n_obs)
        return model
