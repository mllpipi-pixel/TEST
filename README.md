# Coding-Test-for-Prof.-DellaVigna
This project classifies papers using the Groq API. For each paper, it extracts text from PDFs, queries Groq to determine methodology, field, and empirical approach, and outputs the results as a CSV.

## Setup Instructions

1. Open GitHub Codespaces

2. Install Dependencies
- PyPDF2 → Extract text from PDFs
- requests → Call Groq API
- pandas → Save results as CS

3. Set Groq API Key
- export GROQ_API_KEY="your_api_key_here"

4. If desired, add your PDFs to folder pdfs and update file papers.py. Each entry must include:
- title
- authors
- journal
- date
- pdf (local path).

5. Run main.py. This script will:
- Read papers.py.
- Extract text from PDFs.
- Query the Groq API for classification.
- Merge chunk-level classifications.
- Output results in CSV.

6. Results found in CSV file with all papers and their classifications: classified_papers.csv