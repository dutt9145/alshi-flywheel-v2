import os
from dotenv import load_dotenv
import pathlib

load_dotenv(pathlib.Path('.env'), override=True)

import psycopg2

# Get unresolved tickers the same way Phase 3 does (ASC order)
db_url = os.getenv("DATABASE_URL")
conn = psycopg2.connect(db_url)
cur = conn.cursor()

cur.execute("""
    SELECT ticker FROM signals 
    WHERE outcome IS NULL 
    GROUP BY ticker 
    ORDER BY MAX(created_at) ASC 
    LIMIT 20
""")
tickers = [r[0] for r in cur.fetchall()]
conn.close()

print(f"First 20 unresolved tickers (oldest first):")
for t in tickers:
    print(f"  {t}")

# Now test a few
from shared.kalshi_client import KalshiClient
client = KalshiClient()

print(f"\nLooking up first 5:")
for ticker in tickers[:5]:
    market = client.get_market(ticker)
    status = market.get('status', 'NOT_FOUND')
    result = market.get('result', '')
    print(f"  {ticker}: status={status}, result={result}")