#################################
# SETUP                         #
#################################

# LIBRARY IMPORTS
import os # Used to read API key
import json # Used to parse and format JSON responses
import requests # Used for HTTP requests (PDF download, API calls)
import time # Used to sleep when rate-limited
import re # Used to clean model output with regular expressions
from PyPDF2 import PdfReader # Used to extract text from PDF

# READ API KEY FROM ENVIRONMENT, PROVIDE API URL, PROVIDE MODEL TO USE
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-20b"


################################################
# DEFINE ALLOWED CATEGORIES & PROVIDE KEYWORDS
################################################

# DEFINE CATEGORIES (PROVIDED)
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

# DEFINE ALLOWED LABELS
ALLOWED = {
    "methodology": METHODS,
    "field": FIELDS,
    "empirical_approach": APPROACHES
}

# PROVIDE (SOME) KEYWORDS FOR EMPIRICAL APPROACH
APPROACH_KEYWORDS = {
    "Regression Discontinuity": ["regression discontinuity", "cutoff", "threshold", "border"],
    "Event Study": ["event study", "policy shock", "announcement"],
    "RCT": ["randomized", "random assignment", "field experiment"],
    "Synthetic Control": ["synthetic control", "donor pool"],
    "Structural Model Estimation": ["structural model", "calibration"],
    "Lab Experiment": ["lab experiment", "laboratory"],
    "Descriptive/Observational": ["panel data", "survey", "observational"],
    "Other": ["IV", "instrument", "cointegration", "autoregressive", "VAR"]
}


#################################
# PDF DOWNLOAD
#################################
# DOWNLOAD PDF
def download_pdf(url):
    """
    Downloads a PDF from a URL and saves it locally as tmp.pdf.
    Raises an error if the download fails.
    """
    fname = "tmp.pdf"                          # Temporary filename
    r = requests.get(url, timeout=30)          # HTTP GET with timeout
    r.raise_for_status()                       # Fail loudly on HTTP errors
    with open(fname, "wb") as f:               # Write binary PDF to disk
        f.write(r.content)
    return fname                               # Return local file path


def extract_relevant_text(pdf_path):
    """
    Extracts only the abstract, introduction, and conclusion
    from a PDF to reduce token usage and noise.
    """
    reader = PdfReader(pdf_path)               # Load PDF
    text = "\n".join(
        page.extract_text() or ""              # Extract text page by page
        for page in reader.pages
    )

    lines = text.splitlines()                  # Split text into individual lines

    def section(name):
        """
        Find the line index where a given section header appears.
        Returns None if not found.
        """
        for i, l in enumerate(lines):
            if name in l.lower():
                return i
        return None

    # Locate key sections
    abs_i = section("abstract")
    intro_i = section("introduction")
    concl_i = section("conclusion")

    parts = []

    # Extract abstract (up to introduction)
    if abs_i is not None and intro_i:
        parts.extend(lines[abs_i:intro_i])

    # Extract first ~200 lines of introduction
    if intro_i:
        parts.extend(lines[intro_i:intro_i + 200])

    # Extract first ~150 lines of conclusion
    if concl_i:
        parts.extend(lines[concl_i:concl_i + 150])

    # Join extracted parts and cap total length
    return "\n".join(parts[:3000])  # hard cap to control token usage

#################################
# LLM CALL
#################################

def call_groq(prompt):
    """
    Sends a prompt to the Groq API and returns the model's text response.
    Automatically handles rate limits.
    """
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 800
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",  # API authentication
        "Content-Type": "application/json"
    }

    while True:
        r = requests.post(
            GROQ_API_URL,
            headers=headers,
            json=payload
        )

        # Handle rate limiting explicitly
        if r.status_code == 429:
            print("Rate limited. Waiting...")
            time.sleep(10)
            continue

        r.raise_for_status()  # Fail on other HTTP errors
        return r.json()["choices"][0]["message"]["content"]

#################################
# CLASSIFICATION LOGIC
#################################

def classify_chunk(text, paper):
    """
    Classifies a single chunk of text using the LLM.
    Returns a dictionary with methodology, field, and empirical approach.
    """
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

    out = call_groq(prompt)                     # Call the LLM

    # Remove Markdown code fences if the model adds them
    out = re.sub(r"```.*?```", "", out, flags=re.S).strip()

    try:
        return json.loads(out)                  # Parse JSON output
    except:
        # Fail gracefully if parsing fails
        return {"methodology": [], "field": [], "empirical_approach": []}


def merge_classifications(chunks):
    """
    Merges classifications from multiple chunks.
    Enforces allowed labels and guarantees non-empty outputs.
    """
    merged = {k: set() for k in ALLOWED}         # Use sets to avoid duplicates

    for c in chunks:
        for k in ALLOWED:
            merged[k].update(
                x for x in c.get(k, []) if x in ALLOWED[k]
            )

    # 🔒 NEVER EMPTY FALLBACKS (guarantees valid output)
    if not merged["methodology"]:
        merged["methodology"].add("Empirical")

    if not merged["field"]:
        merged["field"].add("Public")

    if not merged["empirical_approach"]:
        merged["empirical_approach"].add("Descriptive/Observational")

    # Convert sets to sorted lists for JSON output
    return {k: sorted(v) for k, v in merged.items()}

#################################
# MAIN CLASSIFIER
#################################

def classify_paper(paper):
    """
    End-to-end classification for a single paper:
    download → extract → chunk → classify → merge
    """
    pdf_path = download_pdf(paper["pdf"])       # Download PDF
    text = extract_relevant_text(pdf_path)      # Extract relevant sections

    # Split text into manageable chunks for the LLM
    chunks = [text[i:i+1500] for i in range(0, len(text), 1500)]

    # Classify each chunk independently
    results = [classify_chunk(c, paper) for c in chunks]

    # Merge chunk-level classifications
    return merge_classifications(results)

#################################
# EXAMPLE PAPERS
#################################

# List of papers to classify
papers = [
    {
        "title": "The Effect of Rural Credit on Deforestation: Evidence from the Brazilian Amazon",
        "authors": ["Juliano Assuncao", "Others"],
        "journal": "The Economic Journal",
        "date": "2020",
        "pdf": "https://watermark02.silverchair.com/uez060.pdf?token=..."
    }





    
]

#################################
# RUN
#################################

# Run classification for all papers
output = []
for p in papers:
    output.append({
        **p,                                      # Original metadata
        "classification": classify_paper(p)      # LLM-derived classification
    })

# Pretty-print final JSON output
print(json.dumps(output, indent=2))