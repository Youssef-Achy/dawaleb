"""
DawaLeb core logic: match a user's query to a registered drug, retrieve
official alternatives, and (optionally) explain them with a free-tier LLM.

Design principles (safety by design):
  1. The system only surfaces data retrieved from the official MoPH dataset.
  2. The LLM may EXPLAIN retrieved records; it may never add drug facts.
  3. Dosing / diagnosis / medical-advice questions are refused in CODE,
     before any LLM call, with a bilingual redirect to a pharmacist/doctor.
  4. Every answer ends with a "confirm with your pharmacist" disclaimer.
  5. The app works with NO API key at all (template answers) — the LLM is
     an optional layer, so the project runs at $0 even with no provider.

LLM provider is configurable via environment variables (OpenAI-compatible
endpoints — Groq and Google Gemini both offer permanent free tiers):
    LLM_BASE_URL   default https://api.groq.com/openai/v1
    LLM_MODEL      default llama-3.3-70b-versatile
    LLM_API_KEY    (or GROQ_API_KEY)
"""

from __future__ import annotations

import json
import os
import re

from rapidfuzz import fuzz, process

from build_lookup import INDEX_PATH, normalize


ALIASES = {
    "اسبرين": "acetylsalicylic acid",
    "أسبرين": "acetylsalicylic acid",
    "اسبيرين": "acetylsalicylic acid",
    "بنادول": "paracetamol",
    "باراسيتامول": "paracetamol",
    "ابيليفي": "abilify",
    "أبيليفي": "abilify",
    "aspirin": "acetylsalicylic acid",
    "aspirine": "acetylsalicylic acid",
    "panadol": "paracetamol",
    "doliprane": "paracetamol",
    "acetaminophen": "paracetamol",
    "vitamin c": "vitamin c",
}


ADVICE_PATTERNS = [
    r"\bhow (much|many)\b",
    r"\bdos(e|age|ing)\b",
    r"\bhow often\b",
    r"\bshould i take\b",
    r"\bcan i take\b",
    r"\bis it safe\b",
    r"\bsafe (for|with|during)\b",
    r"\bpregnan",
    r"\bbreastfeed",
    r"\bmix(ing)? with\b",
    r"\btogether with\b",
    r"\binteract",
    r"\bside effects?\b",
    r"\bwhat happens if\b",
    r"\boverdose\b",
    r"\bdiagnos",
    r"\bdo i have\b",
    r"\bsymptom",
    r"كم (حبة|جرعة|مرة)",
    r"جرعة",
    r"هل (آخذ|اخذ|استعمل|أستعمل)",
    r"هل هو (آمن|امن)",
    r"جانبي",
    r"حامل",
    r"مرضع",
    r"تفاعل",
    r"مع الكحول",
]

DISCLAIMER = {
    "en": (
        "This is registered-product information from the Lebanese Ministry "
        "of Public Health, not medical advice. Always confirm any "
        "substitution with your pharmacist or doctor."
    ),
    "ar": (
        "هذه معلومات عن الأدوية المسجّلة لدى وزارة الصحة العامة اللبنانية "
        "وليست نصيحة طبية. استشر الصيدلي أو الطبيب دائماً قبل أي استبدال."
    ),
}

REFUSAL = {
    "en": (
        "I can only share official registration information (alternatives, "
        "prices, subsidy status). I can't give dosing or medical advice — "
        "please ask your pharmacist or doctor about that. If this is an "
        "emergency, contact the Lebanese Red Cross on 140."
    ),
    "ar": (
        "يمكنني فقط مشاركة معلومات التسجيل الرسمية (البدائل والأسعار). "
        "لا أستطيع تقديم جرعات أو نصائح طبية — يرجى سؤال الصيدلي أو الطبيب. "
        "في الحالات الطارئة اتصل بالصليب الأحمر اللبناني على الرقم 140."
    ),
}

NOT_FOUND = {
    "en": (
        "I couldn't find that medication in the MoPH registered-drugs "
        "database. Check the spelling, or try the active ingredient name."
    ),
    "ar": (
        "لم أجد هذا الدواء في قاعدة بيانات الأدوية المسجّلة لدى وزارة "
        "الصحة. تأكد من الإملاء أو جرّب اسم المادة الفعالة."
    ),
}


def detect_language(text: str) -> str:
    return "ar" if re.search(r"[\u0600-\u06FF]", text) else "en"


def is_advice_query(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in ADVICE_PATTERNS)


class DrugAssistant:
    def __init__(self, index_path: str = INDEX_PATH):
        with open(index_path, encoding="utf-8") as f:
            self.index = json.load(f)
        self.products = self.index["products"]
        self.by_ingredient = self.index["by_ingredient"]
        self.name_keys = {p["name_key"]: i for i, p in enumerate(self.products)}
        self.search_terms = [
            t for t in list(self.name_keys) + list(self.by_ingredient) if len(t) >= 3
        ]

    def _exact_match(self, query: str) -> dict | None:
        """Exact (post-alias) brand or ingredient hit only; else None."""
        q = normalize(query)
        if not q:
            return None
        q = ALIASES.get(q, ALIASES.get(query.strip().lower(), q))
        q = normalize(q)
        if q in self.by_ingredient:
            return {"type": "ingredient", "ingredient": q}
        if q in self.name_keys:
            return {"type": "product", "product": self.products[self.name_keys[q]]}
        return None

    def match(self, query: str, min_score: int = 82) -> dict | None:
        """
        Resolve a free-text query (any language, possibly misspelled) to
        either a product or an ingredient. Returns:
            {"type": "product", "product": {...}}            or
            {"type": "ingredient", "ingredient": "..."}      or None
        """
        q = normalize(query)
        if not q:
            return None
        q = ALIASES.get(q, ALIASES.get(query.strip().lower(), q))
        q = normalize(q)

        in_ing = q in self.by_ingredient
        in_brand = q in self.name_keys
        if in_ing:
            return {"type": "ingredient", "ingredient": q}
        if in_brand:
            return {"type": "product", "product": self.products[self.name_keys[q]]}

        if len(q) >= 3:
            pref = [
                t
                for t in self.search_terms
                if len(t) >= 4
                and (t.startswith(q) or (q.startswith(t) and len(t) >= 0.6 * len(q)))
            ]
            if pref:
                return self._term_result(min(pref, key=len))

        window = max(4, int(0.5 * len(q)))
        candidates = [
            t
            for t in self.search_terms
            if len(t) >= 4 and abs(len(t) - len(q)) <= window
        ]
        hit = process.extractOne(
            q, candidates, scorer=fuzz.ratio, score_cutoff=min_score
        )
        if hit:
            return self._term_result(hit[0])
        return None

    def _term_result(self, term: str) -> dict:
        if term in self.name_keys:
            return {"type": "product", "product": self.products[self.name_keys[term]]}
        return {"type": "ingredient", "ingredient": term}

    def alternatives(self, match: dict) -> dict:
        """
        Return the matched product (if any) plus all registered products
        sharing its primary active ingredient, sorted cheapest first.
        Products with identical combo_key are flagged as closest equivalents.
        """
        if match["type"] == "product":
            prod = match["product"]
            ingredient = prod["primary_ingredient"]
        else:
            prod = None
            ingredient = match["ingredient"]

        idxs = self.by_ingredient.get(ingredient, [])
        alts = [self.products[i] for i in idxs]
        if prod:
            alts = [a for a in alts if a["moph_id"] != prod["moph_id"]]
            for a in alts:
                a = a
        alts.sort(key=lambda a: (a["price_lbp"] is None, a["price_lbp"] or 0))
        return {
            "query_product": prod,
            "ingredient": ingredient,
            "alternatives": alts,
            "closest": [
                a
                for a in alts
                if prod
                and a["combo_key"] == prod["combo_key"]
                and a["dosage"] == prod["dosage"]
            ]
            if prod
            else [],
        }

    def answer(self, query: str, use_llm: bool = True) -> dict:
        """
        Full pipeline. Returns a dict with keys:
            lang, refused, found, text, result (retrieval payload or None)
        """
        lang = detect_language(query)

        exact = self._exact_match(query)
        if exact is not None:
            match = exact
        elif is_advice_query(query):
            return {
                "lang": lang,
                "refused": True,
                "found": False,
                "text": REFUSAL[lang],
                "result": None,
            }
        else:
            match = self.match(query)
        if match is None:
            return {
                "lang": lang,
                "refused": False,
                "found": False,
                "text": NOT_FOUND[lang],
                "result": None,
            }

        result = self.alternatives(match)
        if use_llm and llm_available():
            text = llm_explain(query, result, lang)
        else:
            text = template_explain(result, lang)
        return {
            "lang": lang,
            "refused": False,
            "found": True,
            "text": text,
            "result": result,
        }


def fmt_price(p: int | None, lang: str) -> str:
    if p is None:
        return "price not listed" if lang == "en" else "السعر غير مدرج"
    return f"{p:,} LBP" if lang == "en" else f"{p:,} ل.ل"


def template_explain(result: dict, lang: str) -> str:
    prod, alts = result["query_product"], result["alternatives"]
    lines = []
    if lang == "ar":
        if prod:
            lines.append(
                f"الدواء: {prod['brand_name']} — المادة الفعالة: "
                f"{prod['ingredients_raw']} — {fmt_price(prod['price_lbp'], lang)}"
            )
        lines.append(
            f"البدائل المسجّلة التي تحتوي على {result['ingredient']}:"
            if alts
            else "لا توجد بدائل مسجّلة أخرى بنفس المادة الفعالة."
        )
        for a in alts:
            lines.append(
                f"• {a['brand_name']} ({a['dosage']}, {a['form']}) — "
                f"{fmt_price(a['price_lbp'], lang)}"
            )
    else:
        if prod:
            lines.append(
                f"Found: {prod['brand_name']} — active ingredient: "
                f"{prod['ingredients_raw']} — {fmt_price(prod['price_lbp'], lang)}"
            )
        lines.append(
            f"Registered products containing {result['ingredient']}:"
            if alts
            else "No other registered product shares this active ingredient."
        )
        for a in alts:
            lines.append(
                f"• {a['brand_name']} ({a['dosage']}, {a['form']}) — "
                f"{fmt_price(a['price_lbp'], lang)}"
            )
    lines.append("")
    lines.append(DISCLAIMER[lang])
    return "\n".join(lines)


LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_API_KEY = os.environ.get("LLM_API_KEY") or os.environ.get("GROQ_API_KEY")

SYSTEM_PROMPT = """You are DawaLeb, an assistant that explains official drug
registration data from the Lebanese Ministry of Public Health.

STRICT RULES:
1. Use ONLY the facts in the RETRIEVED DATA block. Never add drug names,
   prices, uses, or any fact not present there.
2. Never give dosing instructions, medical advice, safety judgments,
   interactions, or recommendations to take/avoid a medication.
3. If asked anything outside the retrieved data, say you can only share
   official registration information and to ask a pharmacist or doctor.
4. Answer in the same language as the user (Lebanese Arabic if Arabic).
5. Keep it short and clear for ordinary people. Mention cheaper options
   factually (by listed price) without recommending them.
6. ALWAYS end with exactly this disclaimer, translated to the user's
   language: "This is registered-product information from the Lebanese
   Ministry of Public Health, not medical advice. Always confirm any
   substitution with your pharmacist or doctor."
"""


def llm_available() -> bool:
    return bool(LLM_API_KEY)


def llm_explain(query: str, result: dict, lang: str) -> str:
    """Ask the LLM to explain the retrieved records. Falls back to the
    deterministic template on any error — the app never breaks."""
    try:
        from openai import OpenAI

        client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
        payload = {
            "query_product": result["query_product"],
            "shared_ingredient": result["ingredient"],
            "registered_alternatives": [
                {
                    "brand": a["brand_name"],
                    "ingredients": a["ingredients_raw"],
                    "dosage": a["dosage"],
                    "form": a["form"],
                    "price_lbp": a["price_lbp"],
                    "type": a["b_g"],
                }
                for a in result["alternatives"]
            ],
        }
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0.2,
            max_tokens=600,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"USER QUESTION: {query}\n\nRETRIEVED DATA:\n"
                    f"{json.dumps(payload, ensure_ascii=False, indent=1)}",
                },
            ],
        )
        text = resp.choices[0].message.content.strip()
        if "pharmacist" not in text.lower() and "الصيدلي" not in text:
            text += "\n\n" + DISCLAIMER[lang]
        return text
    except Exception as e:
        print(f"[LLM fallback] {e}")
        return template_explain(result, lang)


if __name__ == "__main__":
    a = DrugAssistant()
    for q in [
        "abilify",
        "ABIPREX",
        "abiprx 10",
        "اسبرين",
        "how much abiprex should I take?",
        "xyzzy",
    ]:
        r = a.answer(q, use_llm=False)
        print(f"\n>>> {q}\n{r['text']}")
