# fast SEC ingestion pipeline that pulls S&P 500 10-K filings using the SEC submissions API.
# It maps tickers to CIKs, fetches filing metadata, identifies the latest 10-K per company, and
# downloads documents in parallel while respecting SEC rate limits. This gives me a high-quality corpus
# for RAG ingestion in minutes rather than hours

import os
import time
import requests
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# =========================
# CONFIG (DO NOT OVER-TUNE)
# =========================
SAVE_DIR = "sec_10k_fast"
YEAR = 2023
MAX_WORKERS = 8   # SAFE for SEC
FORM_TYPE = "10-K"

HEADERS = {
    "User-Agent": "MehulVaidya mehul@example.com"
}

os.makedirs(SAVE_DIR, exist_ok=True)

BASE = "https://www.sec.gov"
SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"

# =========================
# GET S&P 500 CIK LIST
# =========================
def get_sp500_ciks():
    url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
    df = pd.read_csv(url)

    cik_map = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers=HEADERS
    ).json()

    ticker_to_cik = {
        v["ticker"]: str(v["cik_str"]).zfill(10)
        for v in cik_map.values()
    }

    return [
        ticker_to_cik[t]
        for t in df["Symbol"]
        if t in ticker_to_cik
    ]

#CIK for 500 will be in this format ['0000320193', '0000789019', ...]
# =========================
# FIND LATEST 10-K
# =========================
def find_latest_10k(cik):
    try:
        r = requests.get(SUBMISSIONS.format(cik=cik), headers=HEADERS, timeout=10)
        data = r.json()

        filings = data["filings"]["recent"]
        for i, form in enumerate(filings["form"]):
            if form == FORM_TYPE and filings["filingDate"][i].startswith(str(YEAR)):
                accession = filings["accessionNumber"][i].replace("-", "")
                primary = filings["primaryDocument"][i]
                return f"{BASE}/Archives/edgar/data/{int(cik)}/{accession}/{primary}"
    except:
        return None


# =========================
# DOWNLOAD FILE
# =========================
def download_10k(url):
    if not url:
        return

    fname = url.split("/")[-1]
    path = os.path.join(SAVE_DIR, fname)

    if os.path.exists(path):
        return

    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code == 200:
            with open(path, "wb") as f:
                f.write(r.content)
    except:
        pass


# =========================
# MAIN
# =========================
print("🔍 Fetching S&P 500 CIKs...")
ciks = get_sp500_ciks()
print(f"✔ Found {len(ciks)} companies\n")

print("📄 Locating latest 10-K filings...")
urls = []

for cik in tqdm(ciks):
    url = find_latest_10k(cik)
    if url:
        urls.append(url)
    time.sleep(0.05)

print(f"\n⬇ Downloading {len(urls)} 10-K filings...\n")

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
    list(tqdm(pool.map(download_10k, urls), total=len(urls)))

print("\n✅ DONE")
print(f"📁 Saved to: {SAVE_DIR}")
