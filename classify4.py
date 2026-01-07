# IMPORTS
import os          # For reading API key from environment variables
import json        # For parsing and writing JSON
import re          # For cleaning model output
import time        # For sleep/delay in case of rate-limiting
import requests    # For HTTP requests (Groq API)
from PyPDF2 import PdfReader  # For extracting text from PDF files

#################################
# CONFIG
#################################
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Read Groq API key from environment
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"  # Groq API endpoint
MODEL = "openai/gpt-oss-20b"  # Model name to use

# STRICTLY ALLOWED CLASSIFICATION OPTIONS
METHODS = ["Econometrics", "Empirical", "Macro", "Theory"]  # Allowed methodology labels
FIELDS = [
    "Behavioral", "Development", "Econometrics", "Experimental",
    "Finance", "Industrial Organization", "Labor",
    "Macro", "Public", "Theory", "Trade"
]  # Allowed field labels
APPROACHES = [
    "Descriptive/Observational", "Event Study", "Lab Experiment",
    "RCT", "Regression Discontinuity", "Structural Model Estimation",
    "Synthetic Control", "Other"
]  # Allowed empirical approach labels


#################################
# PDF TEXT EXTRACTION
#################################
def extract_pdf_text(pdf_path):
    """
    Reads a PDF from local path and extracts all text.
    Returns first 3000 characters only (to limit prompt size).
    """
    reader = PdfReader(pdf_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)[:3000]  # cap 3k chars


#################################
# GROQ API CALL
#################################
def call_groq(prompt):
    """
    Sends a classification prompt to the Groq API and returns the JSON output.
    Handles rate-limiting and fallback if parsing fails.
    """
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}", # Bearer token authentication
        "Content-Type": "application/json" # JSON request
    }
    payload = {"model": MODEL, # Model to use
               "messages":[{"role":"user","content":prompt}], # Prompt content
               "max_tokens":800} # Limit response length

    while True: # Keep retrying if rate-limited
        try:
            r = requests.post(GROQ_API_URL, headers=headers, json=payload) # Call API
            if r.status_code == 429: # Rate-limited
                print("Rate limited. Waiting 10s...") 
                time.sleep(10)
                continue # Retry after delay
            r.raise_for_status() # Raise error for other HTTP issues

            # Extract model's content
            out = r.json()["choices"][0]["message"]["content"]

            # Remove Markdown/code fences from output
            out = re.sub(r"```.*?```", "", out, flags=re.S).strip()
            return json.loads(out) # Parse JSON and return
        except json.JSONDecodeError: # If JSON parsing fails
            return {
                "methodology": ["Error"],
                "field": ["Error"],
                "empirical_approach": ["Error"]
            }
        except requests.RequestException as e: # Network or other request errors
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
    """
    High-level function to classify a paper.
    1. Extract PDF text
    2. Build prompt
    3. Call Groq
    4. Return classification JSON
    """
    text = extract_pdf_text(pdf_path) # Extract first 3000 chars from PDF
    # Build prompt including metadata and allowed categories
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
    return call_groq(prompt) # Call Groq API and return classification