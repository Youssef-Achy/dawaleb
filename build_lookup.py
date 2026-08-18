"""
Build the alternatives lookup index from the scraped MoPH drug data.

Input : data/moph_drugs.csv         (full scrape)  or
        data/moph_drugs_sample.csv  (bundled sample for testing)
Output: data/drug_index.json

The index maps each product to its parsed active ingredients, and groups
products that share the same primary active ingredient — the basis for
suggesting officially registered alternatives.

Usage:
    python build_lookup.py
    python build_lookup.py path/to.csv
"""

import csv
import json
import os
import re
import sys
import unicodedata

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
FULL_CSV = os.path.join(DATA_DIR, "moph_drugs.csv")
SAMPLE_CSV = os.path.join(DATA_DIR, "moph_drugs_sample.csv")
INDEX_PATH = os.path.join(DATA_DIR, "drug_index.json")


def normalize(text: str) -> str:
    """Lowercase, strip accents and punctuation — used as a matching key."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r"[^a-z0-9\u0600-\u06FF ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


SALT_TOKENS = {
    "sodium",
    "disodium",
    "potassium",
    "hcl",
    "hydrochloride",
    "dihydrochloride",
    "hydrobromide",
    "maleate",
    "mesylate",
    "mesilate",
    "besylate",
    "besilate",
    "tosylate",
    "tartrate",
    "bitartrate",
    "citrate",
    "phosphate",
    "diphosphate",
    "hydrogenophosphate",
    "succinate",
    "fumarate",
    "hemifumarate",
    "oxalate",
    "lactate",
    "nitrate",
    "carbonate",
    "bicarbonate",
    "acetate",
    "valerate",
    "propionate",
    "dipropionate",
    "furoate",
    "butyrate",
    "caproate",
    "decanoate",
    "palmitate",
    "stearate",
    "trihydrate",
    "dihydrate",
    "monohydrate",
    "hemihydrate",
    "heptahydrate",
    "pentahydrate",
    "sesquihydrate",
    "hydrate",
    "anhydrous",
    "anhydre",
    "micronized",
    "micronised",
    "trometamol",
    "tromethamine",
    "diethylamine",
    "axetil",
    "proxetil",
    "pivoxil",
    "salt",
    "base",
    "magnesium",
    "calcium",
    "aluminium",
    "aluminum",
    "embonate",
    "pamoate",
    "sulfate",
    "sulphate",
    "hemisulfate",
    "methylsulfate",
    "gluconate",
    "glucuronate",
    "aspartate",
    "orotate",
    "oxide",
    "chloride",
    "bromide",
    "iodide",
}


GUARD_FIRST = {
    "magnesium",
    "calcium",
    "sodium",
    "potassium",
    "zinc",
    "iron",
    "ferrous",
    "ferric",
    "aluminium",
    "aluminum",
    "lithium",
    "barium",
    "ammonium",
    "silver",
    "copper",
    "selenium",
    "chromium",
    "manganese",
}


INGREDIENT_SYNONYMS = {
    "acetaminophen": "paracetamol",
    "simeticone": "simethicone",
    "dextrose glucose": "dextrose",
    "aspirin": "acetylsalicylic acid",
    "aspirine": "acetylsalicylic acid",
    "vitamine c": "vitamin c",
    "acide ascorbique": "vitamin c",
    "ascorbic acid": "vitamin c",
    "vitamine d3": "vitamin d3",
    "colecalciferol": "vitamin d3",
    "cholecalciferol": "vitamin d3",
    "adrenaline": "epinephrine",
    "noradrenaline": "norepinephrine",
    "ciprofloxacin": "ciprofloxacine",
    "salbutamol": "salbutamol",
    "albuterol": "salbutamol",
}

MIN_MOIETY_LEN = 3


def valid_moiety(m: str) -> bool:
    return len(m) >= MIN_MOIETY_LEN


TOKEN_SPELLINGS = {
    "sulphate": "sulfate",
    "sulphur": "sulfur",
}


def moiety(name: str) -> str:
    """'Diclofenac sodium diethylamine' -> 'diclofenac';
    'Magnesium sulfate heptahydrate' -> 'magnesium sulfate' (guarded)."""
    tokens = [TOKEN_SPELLINGS.get(t, t) for t in normalize(name).split()]
    if not tokens:
        return ""
    if tokens[0] in GUARD_FIRST:
        while len(tokens) > 2 and tokens[-1].endswith("hydrate"):
            tokens.pop()
        m = " ".join(tokens)
        return INGREDIENT_SYNONYMS.get(m, m)
    while len(tokens) > 1 and tokens[-1] in SALT_TOKENS:
        tokens.pop()
    m = " ".join(tokens)
    return INGREDIENT_SYNONYMS.get(m, m)


def parse_ingredients(raw: str) -> list[dict]:
    """
    "Paracetamol - 500mg, Diphenhydramine HCl - 25mg"
      -> [{"name": "Paracetamol", "strength": "500mg"},
          {"name": "Diphenhydramine HCl", "strength": "25mg"}]

    MoPH separates ingredients with commas and puts strength after " - ".
    Some entries have no strength; some strengths contain commas inside
    parentheses — we split conservatively on ", " followed by a capital letter.
    """
    parts = re.split(r",\s*(?=[A-Z])", raw.strip())
    out = []
    for part in parts:
        if " - " in part:
            name, strength = part.rsplit(" - ", 1)
        else:
            name, strength = part, ""
        name = name.strip()
        if name:
            out.append({"name": name, "strength": strength.strip()})
    return out


def load_rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def build_index(rows: list[dict]) -> dict:
    products = []
    by_ingredient: dict[str, list[int]] = {}
    by_atc: dict[str, list[int]] = {}

    for row in rows:
        ings = parse_ingredients(row["ingredients"])
        moieties = [moiety(i["name"]) for i in ings]
        valid = [m for m in moieties if valid_moiety(m)]
        primary = valid[0] if valid else ""
        combo_key = "+".join(sorted(m for m in moieties if valid_moiety(m)))
        product = {
            "moph_id": row["moph_id"],
            "atc": row["atc"].strip(),
            "brand_name": row["brand_name"].strip(),
            "b_g": row["b_g"].strip(),
            "ingredients": ings,
            "ingredients_raw": row["ingredients"].strip(),
            "primary_ingredient": primary,
            "combo_key": combo_key,
            "dosage": row["dosage"].strip(),
            "form": row["form"].strip(),
            "price_lbp": int(row["price_lbp"]) if row["price_lbp"] else None,
            "detail_url": row["detail_url"].strip(),
            "name_key": normalize(row["brand_name"]),
        }
        idx = len(products)
        products.append(product)

        for m in moieties:
            if valid_moiety(m):
                by_ingredient.setdefault(m, []).append(idx)
        if product["atc"]:
            by_atc.setdefault(product["atc"], []).append(idx)

    return {"products": products, "by_ingredient": by_ingredient, "by_atc": by_atc}


def main() -> None:
    if len(sys.argv) > 1:
        path = sys.argv[1]
    elif os.path.exists(FULL_CSV):
        path = FULL_CSV
    else:
        path = SAMPLE_CSV
        print("NOTE: full scrape not found, using bundled sample dataset.")

    rows = load_rows(path)
    index = build_index(rows)
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=1)

    n_prod = len(index["products"])
    n_ing = len(index["by_ingredient"])
    multi = sum(1 for v in index["by_ingredient"].values() if len(v) > 1)
    print(
        f"Indexed {n_prod} products, {n_ing} distinct ingredients "
        f"({multi} ingredients have 2+ registered products)."
    )
    print(f"Wrote {INDEX_PATH}")


if __name__ == "__main__":
    main()
