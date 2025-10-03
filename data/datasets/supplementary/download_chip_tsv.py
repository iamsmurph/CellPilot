#%%
import pandas as pd
import time
import os
from urllib.parse import urlparse
import random

#%%
pth = "/data/rbg/users/seanmurphy/CellPilot/data/datasets/supplementary/chip_atlas_analysis_list.csv"
# Try common encodings if utf-8 fails
try:
    df = pd.read_csv(pth, encoding="utf-8")
except UnicodeDecodeError:
    try:
        df = pd.read_csv(pth, encoding="latin1")
    except UnicodeDecodeError:
        df = pd.read_csv(pth, encoding="cp1252")
#%%
links_series = (
    df.loc[df["Genome assembly"].eq("hg38"), "Colocalization (TSV)"]
    .astype(str)
    .str.strip()
)
links = links_series[links_series.str.startswith("http")].unique()
# %%
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0"})
retry = Retry(
    total=5,
    backoff_factor=0.7,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=frozenset(["GET"]),
)
adapter = HTTPAdapter(max_retries=retry)
session.mount("https://", adapter)
session.mount("http://", adapter)

# Build mapping of URL -> (tf_a, cell_class)
subset = df.loc[
    df["Genome assembly"].eq("hg38"),
    ["Antigen", "Cell type class in Colocalization", "Colocalization (TSV)"],
].copy()
subset["Colocalization (TSV)"] = subset["Colocalization (TSV)"].astype(str).str.strip()
subset = subset[subset["Colocalization (TSV)"].str.startswith("http")]

meta_by_url = {}
for _, row in subset.iterrows():
    url = row["Colocalization (TSV)"]
    if url not in meta_by_url:
        meta_by_url[url] = (
            row["Antigen"],
            row["Cell type class in Colocalization"],
        )

urls = list(meta_by_url.keys())

# Output directory for TSVs
output_dir = "/data/rbg/users/seanmurphy/CellPilot/data/datasets/supplementary/chip_colocs"
os.makedirs(output_dir, exist_ok=True)

download_ok = 0
download_skip = 0
download_fail = 0

def get_filename_from_url(file_url: str) -> str:
    parsed = urlparse(file_url)
    base = os.path.basename(parsed.path)
    return base or "download.tsv"

def looks_like_tsv(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(4096)
        lower = head.lower()
        if b"<html" in lower or b"<!doctype html" in lower:
            return False
        # basic TSV check: contains tab and likely header tokens
        has_tab = b"\t" in head
        has_expected = (b"Average" in head) or (b"SRX" in head)
        return has_tab and has_expected
    except Exception:
        return False

def download_with_resume(url: str, dest: str, max_attempts: int = 3) -> bool:
    tmp = dest + ".part"
    start = os.path.getsize(tmp) if os.path.exists(tmp) else 0

    for attempt in range(1, max_attempts + 1):
        headers = {}
        mode = "wb"
        if start > 0:
            headers["Range"] = f"bytes={start}-"
            mode = "ab"

        try:
            with session.get(url, stream=True, timeout=(10, 120), headers=headers) as r:
                if r.status_code == 416:
                    if os.path.exists(tmp):
                        os.replace(tmp, dest)
                    return True
                if r.status_code not in (200, 206):
                    if r.status_code == 200 and start > 0:
                        start = 0
                        if os.path.exists(tmp):
                            os.remove(tmp)
                        continue
                    r.raise_for_status()

                if r.status_code == 200 and start > 0:
                    start = 0
                    if os.path.exists(tmp):
                        os.remove(tmp)
                    return download_with_resume(url, dest, max_attempts=max_attempts)

                os.makedirs(os.path.dirname(tmp), exist_ok=True)
                bytes_written = 0
                with open(tmp, mode) as f:
                    for chunk in r.iter_content(chunk_size=64 * 1024):
                        if not chunk:
                            continue
                        f.write(chunk)
                        bytes_written += len(chunk)

                if bytes_written > 0 or os.path.exists(dest):
                    os.replace(tmp, dest)
                    if looks_like_tsv(dest):
                        return True
                    else:
                        if os.path.exists(dest):
                            os.remove(dest)
                        if os.path.exists(tmp):
                            os.remove(tmp)
                        raise IOError("Downloaded content failed TSV sniff")
        except Exception:
            if attempt == max_attempts:
                return False
            time.sleep((2 ** (attempt - 1)) * 0.7 + random.uniform(0, 0.3))
            continue

    return False

# Download all TSVs
for url in urls:
    filename = get_filename_from_url(url)
    dest_path = os.path.join(output_dir, filename)

    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0 and looks_like_tsv(dest_path):
        download_skip += 1
        continue

    ok = download_with_resume(url, dest_path, max_attempts=3)
    if ok:
        download_ok += 1
    else:
        download_fail += 1

    time.sleep(0.2 + random.uniform(0, 0.2))

print(f"Downloaded: {download_ok}, Skipped: {download_skip}, Failed: {download_fail}")

# Build tidy TF–TF table
records = []
tidy_csv_path = os.path.join(output_dir, "colo_tidy_hg38.csv")

def read_tsv_with_fallbacks(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep="\t", encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return pd.read_csv(path, sep="\t", encoding="latin1")
        except UnicodeDecodeError:
            return pd.read_csv(path, sep="\t", encoding="cp1252")

for url, (tf_a, cell_class) in meta_by_url.items():
    filename = get_filename_from_url(url)
    tsv_path = os.path.join(output_dir, filename)
    if not os.path.exists(tsv_path) or os.path.getsize(tsv_path) == 0:
        continue
    try:
        tsv_df = read_tsv_with_fallbacks(tsv_path)
        # Drop unnamed index-like columns if present
        cols = [c for c in tsv_df.columns if not str(c).startswith("Unnamed:")]
        tsv_df = tsv_df[cols]

        # Partner TF column: prefer common names first, then fallback
        pref = ["Target", "Factor", "Protein", "Gene", "Antigen"]
        candidate_partner_cols = [c for c in pref if c in tsv_df.columns]
        if not candidate_partner_cols:
            candidate_partner_cols = [
                c for c in tsv_df.columns
                if c not in ("Average", "STRING") and not str(c).startswith("SRX")
            ]
        partner_col = candidate_partner_cols[0] if candidate_partner_cols else (tsv_df.columns[0] if len(tsv_df.columns) else None)
        if partner_col is None:
            continue

        # Make numbers numeric where possible
        if "Average" in tsv_df.columns:
            tsv_df["Average"] = pd.to_numeric(tsv_df["Average"], errors="coerce")
        if "STRING" in tsv_df.columns:
            tsv_df["STRING"] = pd.to_numeric(tsv_df["STRING"], errors="coerce")

        has_average = "Average" in tsv_df.columns
        has_string = "STRING" in tsv_df.columns

        for _, r in tsv_df.iterrows():
            tf_b = str(r.get(partner_col, "")).strip()
            if not tf_b:
                continue
            average_colo = r.get("Average") if has_average else None
            string_score = r.get("STRING") if has_string else None
            records.append({
                "tf_a": tf_a,
                "tf_b": tf_b,
                "cell_class": cell_class,
                "average_colo": average_colo,
                "string_score": string_score,
            })
    except Exception:
        # Skip problematic files but continue aggregation
        continue

if records:
    tidy_df = pd.DataFrame.from_records(records)
    tidy_df.to_csv(tidy_csv_path, index=False)
    print(f"Wrote tidy table: {tidy_csv_path} ({len(tidy_df)} rows)")
else:
    print("No records parsed; tidy table not written.")

# %%
