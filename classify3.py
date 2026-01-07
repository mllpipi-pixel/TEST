import os
import json
import re
import time
import csv
import requests
from PyPDF2 import PdfReader

#################################
# CONFIG
#################################
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-20b"

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

#################################
# PDF TEXT EXTRACTION
#################################
def extract_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)[:3000]  # cap 3k chars

#################################
# GROQ LLM CALL
#################################
def call_groq(prompt):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"model": MODEL, "messages":[{"role":"user","content":prompt}],"max_tokens":800}

    while True:
        try:
            r = requests.post(GROQ_API_URL, headers=headers, json=payload)
            if r.status_code == 429:
                print("Rate limited. Waiting 10s...")
                time.sleep(10)
                continue
            r.raise_for_status()
            out = r.json()["choices"][0]["message"]["content"]
            out = re.sub(r"```.*?```", "", out, flags=re.S).strip()
            return json.loads(out)
        except json.JSONDecodeError:
            return {
                "methodology": ["Empirical"],
                "field": ["Public"],
                "empirical_approach": ["Descriptive/Observational"]
            }
        except requests.RequestException as e:
            print(f"Error calling Groq: {e}")
            return {
                "methodology": ["Empirical"],
                "field": ["Public"],
                "empirical_approach": ["Descriptive/Observational"]
            }

#################################
# PAPER CLASSIFICATION
#################################
def classify_paper(pdf_path, metadata):
    text = extract_pdf_text(pdf_path)
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
    return call_groq(prompt)

#################################
# RUN AND EXPORT CSV
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
    { # example of survey/lit review paper (hi!)
        "title": "Psychology and Economics: Evidence from the Field",
        "authors": ["Stefano DellaVigna"],
        "journal": "JEL",
        "date": "2009",
        "pdf": "pdfs/dellavigna.pdf"
    },
    ]

    output = []
    for p in papers:
        classification = classify_paper(p["pdf"], p)
        output.append({
            "title": p["title"],
            "authors": "; ".join(p["authors"]),
            "journal": p["journal"],
            "date": p["date"],
            "methodology": "; ".join(classification["methodology"]),
            "field": "; ".join(classification["field"]),
            "empirical_approach": "; ".join(classification["empirical_approach"])
        })

    csv_file = "classified_papers.csv"
    with open(csv_file,"w",newline="",encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["title","authors","journal","date","methodology","field","empirical_approach"])
        writer.writeheader()
        for row in output:
            writer.writerow(row)

    print(f"CSV saved to {csv_file}")
