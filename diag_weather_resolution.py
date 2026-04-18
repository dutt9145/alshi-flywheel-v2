import os
from dotenv import load_dotenv
import pathlib

load_dotenv(pathlib.Path('.env'), override=True)

from shared.kalshi_client import KalshiClient
client = KalshiClient()

# Real tickers from DB that should have resolved (Apr 17 or earlier)
test_tickers = {
    "crypto": [
        "KXBTC-26APR1703-B74850",
        "KXBTCD-26APR1703-T74599.99",
    ],
    "financial_markets": [
        "KXHOILW-26APR1717-T2.999",
        "KXWTI-26APR17-T75.99",
        "KXTRUFEGGS-26APR17-T2.765",
    ],
    "sports": [
        "KXATPCHALLENGERMATCH-26APR16ABOSEY-ABO",
        "KXATPCHALLENGERMATCH-26APR16BOLNOG-BOL",
    ],
}

for sector, tickers in test_tickers.items():
    print(f"\n=== {sector.upper()} ===")
    for ticker in tickers:
        try:
            market = client.get_market(ticker)
            if market:
                print(f"{ticker[:50]}:")
                print(f"  status={market.get('status')}, result={market.get('result')}")
            else:
                print(f"{ticker[:50]}: NOT FOUND")
        except Exception as e:
            print(f"{ticker[:50]}: ERROR - {str(e)[:60]}")