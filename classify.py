import os
import json
import requests
import time
import re
from PyPDF2 import PdfReader

#################################
# CONFIG
#################################
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-20b"

#################################
# ALLOWED CATEGORIES (STRICT)
#################################
METHODS = ["Econometrics", "Empirical", "Macro", "Theory"]

FIELDS = [
    "Behavioral", "Development", "Econometrics", "Experimental",
    "Finance", "Industrial Organization", "Labor",
    "Macro", "Public", "Theory", "Trade"
]

APPROACHES = [
    "Descriptive/Observational", "Event Study", "Lab Experiment",
    "RCT", "Regression Discontinuity", "Structural Model Estimation",
    "Synthetic Control", "Other"
]

ALLOWED = {
    "methodology": METHODS,
    "field": FIELDS,
    "empirical_approach": APPROACHES
}

#################################
# KEYWORDS FOR EMPIRICAL APPROACH
#################################
APPROACH_KEYWORDS = {
    "Regression Discontinuity": ["regression discontinuity", "cutoff", "threshold", "border"],
    "Event Study": ["event study", "policy shock", "announcement"],
    "RCT": ["randomized", "random assignment", "field experiment"],
    "Synthetic Control": ["synthetic control", "donor pool"],
    "Structural Model Estimation": ["structural model", "calibration"],
    "Lab Experiment": ["lab experiment", "laboratory"],
    "Descriptive/Observational": ["panel data", "survey", "observational"]
}

#################################
# PDF HELPERS
#################################
def download_pdf(url):
    fname = "tmp.pdf"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    with open(fname, "wb") as f:
        f.write(r.content)
    return fname


def extract_relevant_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)

    lines = text.splitlines()

    def section(name):
        for i, l in enumerate(lines):
            if name in l.lower():
                return i
        return None

    abs_i = section("abstract")
    intro_i = section("introduction")
    concl_i = section("conclusion")

    parts = []
    if abs_i is not None and intro_i:
        parts.extend(lines[abs_i:intro_i])
    if intro_i:
        parts.extend(lines[intro_i:intro_i + 200])
    if concl_i:
        parts.extend(lines[concl_i:concl_i + 150])

    return "\n".join(parts[:3000])  # hard cap


#################################
# LLM CALL
#################################
def call_groq(prompt):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 800
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    while True:
        r = requests.post(GROQ_API_URL, headers=headers, json=payload)
        if r.status_code == 429:
            print("Rate limited. Waiting...")
            time.sleep(10)
            continue
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


#################################
# CLASSIFICATION LOGIC
#################################
def classify_chunk(text, paper):
    prompt = f"""
You are an expert economist.

Choose ONLY from the lists below.
Always choose AT LEAST ONE methodology and field.

Methodology options: {METHODS}
Field options: {FIELDS}
Empirical approach options: {APPROACHES}

Paper:
Title: {paper['title']}
Authors: {', '.join(paper['authors'])}
Journal: {paper['journal']}
Date: {paper['date']}

Text:
\"\"\"{text}\"\"\"

Return STRICT JSON:
{{"methodology": [...], "field": [...], "empirical_approach": [...]}}
"""

    out = call_groq(prompt)

    out = re.sub(r"```.*?```", "", out, flags=re.S).strip()
    try:
        return json.loads(out)
    except:
        return {"methodology": [], "field": [], "empirical_approach": []}


def merge_classifications(chunks):
    merged = {k: set() for k in ALLOWED}

    for c in chunks:
        for k in ALLOWED:
            merged[k].update(x for x in c.get(k, []) if x in ALLOWED[k])

    # 🔒 NEVER EMPTY FALLBACKS
    if not merged["methodology"]:
        merged["methodology"].add("Empirical")

    if not merged["field"]:
        merged["field"].add("Public")

    if not merged["empirical_approach"]:
        merged["empirical_approach"].add("Descriptive/Observational")

    return {k: sorted(v) for k, v in merged.items()}


#################################
# MAIN CLASSIFIER
#################################
def classify_paper(paper):
    pdf_path = download_pdf(paper["pdf"])
    text = extract_relevant_text(pdf_path)

    chunks = [text[i:i+1500] for i in range(0, len(text), 1500)]
    results = [classify_chunk(c, paper) for c in chunks]

    return merge_classifications(results)


#################################
# EXAMPLE PAPERS
#################################
papers = [
    {
        "title": "Tipping and the Dynamics of Segregation",
        "authors": ["David Card"],
        "journal": "QJE",
        "date": "2007",
        "pdf": "https://davidcard.berkeley.edu/papers/tipping-dynamics.pdf"
    },
    {
        "title": "The Brazilian Amazon’s Double Reversal of Fortune",
        "authors": ["Robin Burgess", "Francisco Costa", "Ben Olken"],
        "journal": "LSE WP",
        "date": "2023",
        "pdf": "https://static1.squarespace.com/static/5f806416079eeb68f5e277b1/t/670fc00367a9507e903abeef/1729085455619/241015amazon_border.pdf"
    }
]

#################################
# RUN
#################################
output = []
for p in papers:
    output.append({
        **p,
        "classification": classify_paper(p)
    })

print(json.dumps(output, indent=2))
