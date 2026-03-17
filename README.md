# Kalshi Flywheel — 6-Sector Consensus Trading Bot

A fully autonomous Kalshi prediction market trading system built on
Bayesian polynomial models with a 6-bot consensus gate.

## Architecture

```
Data layer  →  6 sector bots (Bayesian poly model)
            →  Consensus engine (all 6 must agree)
            →  Kelly sizer (¼ Kelly, corr. adjusted)
            →  Kalshi API execution
            →  Calibration logger (SQLite)
            →  Nightly retrain (flywheel feedback)
```

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure secrets
cp .env.example .env
# Edit .env with your keys

# 3. Run in demo mode (no real orders)
python orchestrator.py
```

## The 6 Sectors

| Bot | Keywords | Primary data source |
|---|---|---|
| EconomicsBot | CPI, NFP, Fed, GDP | FRED API (free) |
| CryptoBot | BTC, ETH, SOL, XRP | CoinGecko + Fear & Greed |
| PoliticsBot | Election, Congress, bill | Polymarket API + NewsAPI |
| WeatherBot | Temp, rain, storm, hurricane | Open-Meteo (free, no key) |
| TechBot | Earnings, IPO, AI launch | NewsAPI |
| SportsBot | NBA, NFL, MLB, NHL, UFC | ESPN unofficial API |

## Consensus Gate

A trade only executes when ALL 3 conditions are met simultaneously:

1. **Direction agreement** — all 6 bots predict the same YES or NO
2. **Edge threshold** — weighted-average edge > 5% above market price
3. **Confidence floor** — weighted-average model confidence > 65%

Bots are weighted by inverse Brier score — better-calibrated bots
carry more weight as evidence accumulates.

## Bayesian Polynomial Model

Each bot uses:
- **Beta(2, 2) prior** — starts centred at 0.5 with low conviction
- **Degree-3 polynomial features** — captures non-linear relationships
- **Online Bayesian update** — prior tightens with every resolved contract
- **Blended posterior** — ML estimate weighted against Bayesian prior
  using data-confidence weighting (max 85% ML weight at 100+ observations)

## Kelly Sizing

```
f_full = (p × (b + 1) − 1) / b
f_safe = f_full × 0.25   (quarter-Kelly safety cap)
```

Additional hard caps:
- No single trade exceeds 4% of bankroll
- No sector exceeds 15% of bankroll simultaneously

## Files

```
kalshi-flywheel/
├── config/settings.py          ← all tunable parameters
├── shared/
│   ├── kalshi_client.py        ← RSA auth + REST wrapper
│   ├── bayesian_poly_model.py  ← core probability model
│   ├── data_fetchers.py        ← sector-specific data pulls
│   ├── consensus_engine.py     ← 3-gate trade gate
│   ├── kelly_sizer.py          ← position sizing
│   └── calibration_logger.py  ← SQLite persistence
├── bots/
│   ├── base_bot.py             ← abstract base class
│   └── sector_bots.py         ← all 6 concrete sector bots
├── orchestrator.py             ← run loop + scheduler
├── requirements.txt
└── .env.example
```

## The Flywheel

```
Better data → more accurate p̂ → larger edge →
bigger Kelly stake → more P&L → fund better data sources
```

The nightly retrain loop is what makes it compound:
- Every resolved contract becomes a training observation
- Brier score tracks calibration and weights bots in consensus
- Models persist to disk and improve over weeks/months
# alshi-flywheel-v2
