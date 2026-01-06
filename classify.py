import os
import json
import requests
import time
from PyPDF2 import PdfReader  # pip install PyPDF2

#################################
# Configuration
#################################
groq_api_key = os.getenv("GROQ_API_KEY")
groq_api_url = "https://api.groq.com/openai/v1/chat/completions"

methods = ["Econometrics", "Empirical", "Macro", "Theory"]
fields = ["Behavioral", "Development", "Econometrics", "Experimental", "Finance",
          "Industrial Organization", "Labor", "Macro", "Public", "Theory", "Trade"]

approach_keywords = {
    "Descriptive/Observational": ["survey", "observational", "panel data", "cross-section", "statistics", "data analysis"],
    "Event Study": ["event study", "policy shock", "announcement", "financial market reaction"],
    "Lab Experiment": ["lab experiment", "controlled experiment", "participant study"],
    "RCT": ["randomized controlled trial", "RCT", "random assignment", "field experiment"],
    "Regression Discontinuity": ["regression discontinuity", "cutoff", "threshold", "running variable", "border"],
    "Structural Model Estimation": ["structural model", "calibration", "dynamic model", "estimation"],
    "Synthetic Control": ["synthetic control", "counterfactual", "donor pool", "treatment unit"],
    "Other": ["instrumental variable" "IV" "case study", "simulation", "qualitative"],
    "None": []
}

#################################
# Helper: download PDF
#################################
def download_pdf(url, local_path):
    r = requests.get(url)
    r.raise_for_status()
    with open(local_path, "wb") as f:
        f.write(r.content)
    return local_path

#################################
# Helper: extract PDF sections
#################################
def extract_pdf_sections(pdf_path):
    """Extract Abstract, Introduction, Conclusion sections"""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    lines = text.splitlines()

    # Extract Abstract
    try:
        abstract_start = next(i for i, l in enumerate(lines) if "abstract" in l.lower())
        intro_start = next(i for i, l in enumerate(lines) if "introduction" in l.lower())
        abstract_lines = lines[abstract_start+1:intro_start]
    except StopIteration:
        abstract_lines = lines[:50]  # fallback

    # Extract Introduction (first 2 pages)
    intro_lines = lines[:200]

    # Extract Conclusion
    try:
        conclusion_start = next(i for i, l in enumerate(lines) if "conclusion" in l.lower())
        conclusion_lines = lines[conclusion_start:]
    except StopIteration:
        conclusion_lines = lines[-50:]  # fallback

    extracted_text = "\n".join(abstract_lines + intro_lines + conclusion_lines)
    return extracted_text

#################################
# Helper: chunk text
#################################
def chunk_text(text, chunk_size=1500):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i+chunk_size])

#################################
# Helper: merge classifications
#################################
def merge_classifications(class_list):
    merged = {"methodology": set(), "field": set(), "empirical_approach": set()}
    for c in class_list:
        merged["methodology"].update(c.get("methodology", []))
        merged["field"].update(c.get("field", []))
        merged["empirical_approach"].update(c.get("empirical_approach", []))
    return {k: list(v) for k, v in merged.items()}

#################################
# Classify one text chunk
#################################
def classify_chunk(chunk_text, paper):
    keyword_text = "\n".join([f"{k}: {', '.join(v)}" for k, v in approach_keywords.items()])

    prompt = f"""
You are an expert economist. Classify the following paper based on its content.

Paper metadata:
Title: {paper['title']}
Authors: {', '.join(paper['authors'])}
Journal: {paper['journal']}
Date: {paper['date']}

Use the following keywords to help identify the empirical approach:
{keyword_text}

Paper text:
\"\"\"{chunk_text}\"\"\"

Return a JSON object with keys: methodology, field, empirical_approach, each containing a list of categories.
If no category is applicable, return an empty list for that key.
"""

    payload = {
        "model": "openai/gpt-oss-20b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2000
    }

    headers = {"Authorization": f"Bearer {groq_api_key}"}

    try:
        response = requests.post(groq_api_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        output_text = result["choices"][0]["message"]["content"].strip()

        # Remove code fences if present
        if output_text.startswith("```") and output_text.endswith("```"):
            lines = output_text.split("\n")[1:-1]
            output_text = "\n".join(lines).strip()
            if output_text.lower().startswith("json"):
                output_text = output_text[4:].strip()

        return json.loads(output_text)
    except Exception as e:
        print(f"Failed to classify chunk: {e}")
        return {"methodology": [], "field": [], "empirical_approach": []}

#################################
# Classify full paper
#################################
def classify_paper(paper, retries=3, wait_sec=2):
    # Handle PDF URL
    pdf_path = paper["pdf"]
    if pdf_path.startswith("http://") or pdf_path.startswith("https://"):
        pdf_filename = f"tmp_{os.path.basename(pdf_path)}"
        try:
            download_pdf(pdf_path, pdf_filename)
            pdf_path = pdf_filename
        except Exception as e:
            print(f"Failed to download PDF {pdf_path}: {e}")
            return {"methodology": [], "field": [], "empirical_approach": []}

    pdf_text = extract_pdf_sections(pdf_path)
    classifications = []

    # Chunk and classify
    for chunk in chunk_text(pdf_text):
        classifications.append(classify_chunk(chunk, paper))
        time.sleep(1)  # avoid rate limits

    # Merge results
    final_class = merge_classifications(classifications)
    return final_class

#################################
# Example papers list
#################################
papers = [
    {
        "title": "Tipping and the Dynamics of Segregation",
        "authors": ["David Card", "Others"],
        "journal": "The Quarterly Journal of Economics",
        "date": "2007",
        "pdf": "https://davidcard.berkeley.edu/papers/tipping-dynamics.pdf"
    },
    {
        "title": "The Brazilian Amazon’s Double Reversal of Fortune",
        "authors": ["Robin Burgess", "Francisco Costa", "Ben Olken"],
        "journal": "LSE Working Paper",
        "date": "2023-05-01",
        "pdf": "https://static1.squarespace.com/static/5f806416079eeb68f5e277b1/t/670fc00367a9507e903abeef/1729085455619/241015amazon_border.pdf"
    },
    {
        "title": "Do Rural Banks Matter? Evidence from the Indian Social Banking Experiment",
        "authors": ["Abhijit Banerjee", "Esther Duflo", "Rachel Glennerster"],
        "journal": "American Economic Review",
        "date": "2003-06-01",
        "pdf": "https://www.povertyactionlab.org/sites/default/files/publications/Do%20Rural%20Banks%20Matter.pdf"
    },
    {
        "title": "Field Experiments on Corruption",
        "authors": ["Benjamin Olken"],
        "journal": "Handbook of Field Experiments",
        "date": "2017-01-01",
        "pdf": "https://www.nber.org/system/files/chapters/c13891/c13891.pdf"
    },
    {
        "title": "The Aggregate Economic Effects of Large Infrastructure Projects",
        "authors": ["David Donaldson"],
        "journal": "Quarterly Journal of Economics",
        "date": "2018-02-01",
        "pdf": "https://economics.mit.edu/files/15353"
    },
    {
        "title": "Information, Externalities, and Welfare: Evidence from a Randomized Experiment on Mosquito Nets in Kenya",
        "authors": ["Jessica Cohen", "Aldo Moscona", "Others"],
        "journal": "American Economic Review",
        "date": "2019-05-01",
        "pdf": "https://www.aeaweb.org/articles?id=10.1257/aer.20171220"
    },
    {
        "title": "Economic Growth and the Costs of Land Reallocation in the Brazilian Amazon",
        "authors": ["Robin Burgess", "Francisco Costa"],
        "journal": "LSE Working Paper",
        "date": "2020-03-01",
        "pdf": "https://eprints.lse.ac.uk/104435/1/Burgess_Costa_Amazon_Land.pdf"
    },
    {
        "title": "Microcredit in Theory and Practice: Using Randomized Credit Scoring for Impact Evaluation",
        "authors": ["David McKenzie", "Christopher Woodruff"],
        "journal": "American Economic Review",
        "date": "2013-03-01",
        "pdf": "https://www.aeaweb.org/articles?id=10.1257/aer.103.3.741"
    }
]


#################################
# Classify all papers
#################################
dataset = []
for p in papers:
    classification = classify_paper(p)
    dataset.append({
        "title": p["title"],
        "authors": p["authors"],
        "journal": p["journal"],
        "date": p["date"],
        "classification": classification
    })

#################################
# Output
#################################
print(json.dumps(dataset, indent=2))
