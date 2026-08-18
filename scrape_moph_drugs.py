"""
Scrape the Lebanon MoPH National Drugs Database into a clean CSV.

Data source (official, public):
https://www.moph.gov.lb/en/Drugs/index/3/4848/lebanon-national-drugs-database

The alphabetical listing pages are server-side rendered, so we can fetch
them with plain requests + BeautifulSoup (no browser automation needed).

Usage:
    pip install requests beautifulsoup4 pandas lxml
    python scrape_moph_drugs.py

Output:
    moph_drugs.csv  (one row per registered product)

Columns:
    moph_id, atc, brand_name, b_g, ingredients, dosage, form, price_lbp, detail_url

Be polite to the government server: this script waits between requests.
A full run takes roughly 20-40 minutes depending on your connection.
Tip: test first with TEST_MODE = True (scrapes only letter A, page 1-2).
"""

import csv
import re
import time

import requests
from bs4 import BeautifulSoup

BASE = "https://www.moph.gov.lb"
LIST_URL = (
    BASE
    + "/en/Drugs/index/3/4848/letter:{letter}/page:{page}/sort:Drug.brand_name/direction:ASC"
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (student project - Lebanon medication alternatives assistant)"
}

DELAY_SECONDS = 0.0
MAX_PAGES_PER_LETTER = 60
TEST_MODE = False

LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)]


def fetch(url: str, retries: int = 3) -> str | None:
    """GET a URL with retries. Returns HTML text or None."""
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                return r.text
            print(f"  ! HTTP {r.status_code} on {url}")
        except requests.RequestException as e:
            print(f"  ! attempt {attempt} failed: {e}")
        time.sleep(2 * attempt)
    return None


def parse_rows(html: str) -> list[dict]:
    """Extract drug rows from a listing page."""
    soup = BeautifulSoup(html, "lxml")
    rows = []
    for tr in soup.select("table tr"):
        cells = tr.find_all("td")
        if len(cells) != 7:
            continue
        link = tr.find("a", href=re.compile(r"/en/Drugs/view/\d+"))
        if not link:
            continue
        detail_url = link["href"]
        if detail_url.startswith("/"):
            detail_url = BASE + detail_url
        moph_id_match = re.search(r"/view/(\d+)", detail_url)
        moph_id = moph_id_match.group(1) if moph_id_match else ""

        atc, name, b_g, ingredients, dosage, form, price = (
            c.get_text(strip=True) for c in cells
        )
        rows.append(
            {
                "moph_id": moph_id,
                "atc": atc,
                "brand_name": name,
                "b_g": b_g,
                "ingredients": ingredients,
                "dosage": dosage,
                "form": form,
                "price_lbp": clean_price(price),
                "detail_url": detail_url,
            }
        )
    return rows


def clean_price(text: str) -> str:
    """'1,075,074 L.L' -> '1075074'. Empty/placeholder prices -> ''."""
    digits = re.sub(r"[^\d]", "", text)
    return digits


def scrape() -> list[dict]:
    all_rows: list[dict] = []
    seen_ids: set[str] = set()

    letters = ["A"] if TEST_MODE else LETTERS
    for letter in letters:
        print(f"=== Letter {letter} ===")
        prev_first_id = None
        max_pages = 2 if TEST_MODE else MAX_PAGES_PER_LETTER

        for page in range(1, max_pages + 1):
            url = LIST_URL.format(letter=letter, page=page)
            html = fetch(url)
            if html is None:
                print(f"  ! giving up on {url}")
                break

            rows = parse_rows(html)
            if not rows:
                break

            first_id = rows[0]["moph_id"]
            if first_id == prev_first_id:
                break
            prev_first_id = first_id

            new = [r for r in rows if r["moph_id"] not in seen_ids]
            for r in new:
                seen_ids.add(r["moph_id"])
            all_rows.extend(new)
            print(f"  page {page}: {len(new)} new rows (total {len(all_rows)})")

            time.sleep(DELAY_SECONDS)

    return all_rows


def save(rows: list[dict], path: str = "moph_drugs.csv") -> None:
    if not rows:
        print("No rows scraped - nothing to save.")
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} products to {path}")


if __name__ == "__main__":
    data = scrape()
    save(data)

    if data:
        n_priced = sum(1 for r in data if r["price_lbp"])
        n_generic = sum(1 for r in data if r["b_g"].upper().startswith("G"))
        print("\n--- Summary ---")
        print(f"Total products:        {len(data)}")
        print(f"With a listed price:   {n_priced}")
        print(f"Marked generic (G):    {n_generic}")
        print("Next step: group by ATC code / active ingredient to build the")
        print("alternatives lookup table.")
