#################################
# IMPORTS
#################################
import os          # for reading API key
import json        # for JSON output
import re          # clean LLM output
from PyPDF2 import PdfReader  # extract text from PDF
import requests    # call Groq API

#################################
# CONFIG
#################################
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-20b"

#################################
# STRICT CATEGORIES
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

APPROACH_KEYWORDS = {
    "Regression Discontinuity": ["regression discontinuity", "cutoff", "threshold"],
    "Event Study": ["event study", "policy shock"],
    "RCT": ["randomized", "random assignment"],
    "Synthetic Control": ["synthetic control"],
    "Structural Model Estimation": ["structural model", "calibration"],
    "Lab Experiment": ["lab experiment", "laboratory"],
    "Descriptive/Observational": ["panel data", "survey", "observational"],
    "Other": ["IV", "instrument", "VAR"]
}

#################################
# PDF TEXT EXTRACTION
#################################
def extract_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

#################################
# API CALL
#################################
def classify_paper_local(pdf_path, metadata):
    text = extract_pdf_text(pdf_path)[:3000]  # first 3000 chars only

    prompt = f"""
You are an expert economist. Classify this paper using ONLY the options below.
Always select at least one methodology and field.

Methodology options: {METHODS}
Field options: {FIELDS}
Empirical approach options: {APPROACHES}

Paper metadata:
Title: {metadata['title']}
Authors: {', '.join(metadata['authors'])}
Journal: {metadata['journal']}
Date: {metadata['date']}

Text (abstract + intro + conclusion):
\"\"\"{text}\"\"\"

Return STRICT JSON:
{{"methodology": [...], "field": [...], "empirical_approach": [...]}}
"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 800
    }

    r = requests.post(GROQ_API_URL, headers=headers, json=payload)
    r.raise_for_status()
    out = r.json()["choices"][0]["message"]["content"]
    out = re.sub(r"```.*?```", "", out, flags=re.S).strip()

    try:
        return json.loads(out)
    except json.JSONDecodeError:
        # fallback if parsing fails
        return {
            "methodology": ["Empirical"],
            "field": ["Public"],
            "empirical_approach": ["Descriptive/Observational"]
        }

#################################
# TEST
#################################
if __name__ == "__main__":
    papers = [
    { # example of an IV paper
        "title": "The Effect of Rural Credit on Deforestation: Evidence from the Brazilian Amazon",
        "authors": ["Juliano Assuncao", "Others"],
        "journal": "The Economic Journal",
        "date": "2020",
        "pdf": "pdfs/assuncao.pdf"
    },
    { # example of an RD paper
        "title": "Tipping and the Dynamics of Segregation",
        "authors": ["David Card", "Alexandre Mas", "Jesse Rothstein"],
        "journal": "QJE",
        "date": "2008",
        "pdf": "pdfs/card.pdf"
    },
    { # example of an RCT paper
        "title": "Worms: Identifying Impacts on Education and Health in the Presence of Treatment Externalities",
        "authors": ["Ted Miguel", "Michael Kremer"],
        "journal": "Econometrica",
        "date": "2003",
        "pdf": "pdfs/miguel.pdf"
    },
    { # example of an (odd) event study paper
        "title": "Knocking It Down and Mixing It Up: The Impact of Public Housing Regenerations",
        "authors": ["Hector Blanco", "Lorenzo Neri"],
        "journal": "ReStat",
        "date": "2025",
        "pdf": "pdfs/blanco.pdf"
    },
    { # example of a VAR/structural paper
        "title": "U.S. Monetary Policy and the Global Financial Cycle",
        "authors": ["Silvia Miranda-Agrippino", "Helene Rey"],
        "journal": "ReStud",
        "date": "2020",
        "pdf": "pdfs/miranda.pdf"
    },
    { # example of a metrics paper
        "title": "Harvesting Differences-in-Differences and Event-Study Evidence",
        "authors": ["Alberto Abadie", "Joshua Angrist", "Brigham Frandsen", "Jörn-Steffen Pischke"],
        "journal": "NBER",
        "date": "2025",
        "pdf": "pdfs/abadie.pdf"
    },
    ]

# Run classification for all papers
output = []
for p in papers:
    classification = classify_paper_local(p["pdf"], p)
    print(json.dumps(classification, indent=2))
