# 💊 DawaLeb — AI Medication Alternatives Assistant for Lebanon

**LebNet Tech Fellows — Final Project (Option 2: AI for Lebanon)**
Youssef El Achy · youssef.n.elachy@gmail.com

Lebanon's medicine shortages and price inflation leave patients unsure which
registered generic equivalents exist, what they cost, and whether they are
subsidized. That information is public — the Ministry of Public Health (MoPH)
publishes the National Drugs Database and, under Law No. 91/2010, an official
substitution framework — but it lives in hard-to-search government tables.

**DawaLeb** turns that data into a free, bilingual (Arabic/English) chatbot:
type a medication name (even misspelled, in Arabic script, or a foreign brand
like "Panadol") and get the officially registered products that share its
active ingredient, with current prices — always ending with a reminder to
confirm with a pharmacist.

Built on the complete registry: **5,925 registered products, 1,444 distinct
active ingredients.**

## How it works (RAG, safety-by-design)

```
user query
   ├─► exact match?  ──────────────────────────► lookup (bypasses advice filter)
   ├─► medical-advice pattern? ────────────────► refuse + redirect to pharmacist
   └─► normalize + alias map (AR/EN/FR)
          ──► prefix match / length-windowed fuzzy match (RapidFuzz)
          ──► retrieve every registered product sharing the active moiety
          ──► free-tier LLM explains ONLY the retrieved records
              (deterministic template if no key / API down)
          ──► answer + price table + mandatory disclaimer
```

Safety rules are enforced **in code, before any LLM call**: dosing, interaction,
pregnancy, side-effect, and diagnosis questions are refused with a redirect to a pharmacist/doctor (and the Red Cross 140 line for emergencies). The LLM never
adds facts; if it is unavailable the deterministic template answers instead, so
the app never breaks — and runs at **$0**, with or without an API key.

## Quickstart

```bash
pip install -r requirements.txt
```

**1. Get the data** (I already ran it before pushing to github)

```bash
python build_lookup.py            # instant and uses the bundled 22-product sample
python scrape_moph_drugs.py       # ~30 min and scrapes the full official database
python build_lookup.py            # re-indexes after the full scrape
```

> The scraper writes `data/moph_drugs.csv`; `build_lookup.py` picks it up
> automatically and stops printing the "using bundled sample" notice.


```bash
export GROQ_API_KEY=gsk_...              # macOS / Linux
$env:GROQ_API_KEY="gsk_..."              # Windows PowerShell
```

Any OpenAI-compatible provider works — swap `LLM_BASE_URL`, `LLM_MODEL`, and
`LLM_API_KEY` to change vendors without touching the code.

**3. Run**

In the first terminal, run:

```bash
streamlit run app.py     # the web app
```

Try: `abilify` · `اسبرين` · `aripiprazole` · `abiprx 10` (typo demo) ·
`how much should I take?` (refusal demo) or anything else !

## Evaluation

In another terminal, run:

```bash
python evaluate.py       # the evaluation suite
```

Measured on the **full MoPH registry (5,925 products, 1,444 ingredients)**.
Evaluation runs with the LLM disabled so results are deterministic and
reproducible: it measures the layers that decide *what information is shown*.

| Metric | What it checks | Result |
|---|---|---|
| Retrieval accuracy | exact-name queries return the complete registered-alternatives set | **99.9%** (5,921/5,925) |
| Misspelling robustness | same, with programmatic typos (deletion, swap, doubling) | **98.7%** (5,837/5,914) |
| Ingredient lookup | ingredient queries return every product containing it | **100%** (1,444/1,444) |
| Safety refusals | 12 EN/AR medical-advice questions refused; 6 lookups not falsely refused | **100%**, 0 false refusals |

Residual misspelling failures are almost entirely 4–5 letter brand names
(ADOL, TARKA, ZOMAX, RISEK) whose scrambled forms are ambiguous non-words. At
that length a single transposition destroys nearly all signal, so the system
returns nothing rather than risk suggesting the wrong medication.

### Bugs the test suite caught (that manual demoing never would)

- **Salt fragmentation** — diclofenac sodium / potassium / diethylamine were
  indexed as three unrelated drugs. Fixed with active-moiety normalization,
  guarded by a mineral list so magnesium sulfate and sodium chloride, where the
  salt *is* the drug, stay intact.
- **Synonym fragmentation** — the registry independently uses acetaminophen and
  paracetamol, simeticone and simethicone, sulphate and sulfate.
- **Parser poisoning** — ingredient cells like `Vitamin A - 1000IU, C - 100mg`
  produced a junk ingredient `c` which, under substring matching, silently
  absorbed hundreds of unrelated queries.
- **Guardrail false positive** — the real product *SINOCORT METERED DOSE* was
  refused because its name contains "DOSE". Exact registered names now bypass
  the advice filter; genuine advice questions are still refused 100% of the time.

## Repository layout

| File | Purpose |
|---|---|
| `scrape_moph_drugs.py` | scraper for the MoPH National Drugs Database |
| `build_lookup.py` | ingredient parsing, active-moiety normalization, alternatives index |
| `assistant.py` | matching, retrieval, safety guardrails and LLM layer |
| `app.py` | polyglot Streamlit chat interface |
| `evaluate.py` | retrieval / robustness / safety test suite |
| `data/moph_drugs_sample.csv` | 22 real MoPH records for a demo |

## Data sources (official, public)

- [Lebanon National Drugs Database](https://www.moph.gov.lb/en/Drugs/index/3/4848/lebanon-national-drugs-database)
- [Lebanon Substitution Drug List](https://moph.gov.lb/en/DynamicPages/index/3/4841/lebanon-substitution-drug-list)
- [Drugs Public Price List](https://moph.gov.lb/en/Pages/3/3101/drugs-public-price-list-)


## Please Note

DawaLeb is an information tool, not a medical device and not medical advice.
It presents products sharing an active ingredient and doesn't claim that they areinterchangeable. Always confirm any substitution with a licensed pharmacist or doctor. As for the scrape_moph_drugs.py and the build_lookup.py there is no need to re-run them as they were alreday ran before pushing to github.
Source data © Lebanese Ministry of Public Health.
