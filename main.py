import csv  # For writing CSV output
from classify4 import classify_paper  # Import classification functions
from papers import papers  # Import list of papers

output = []  # List to store processed rows

# Loop over each paper
for p in papers:
    classification = classify_paper(p["pdf"], p)  # Classify PDF via Groq
    output.append({
        "title": p["title"],  # Paper title
        "authors": "; ".join(p["authors"]),  # Authors separated by semicolons
        "journal": p["journal"],  # Journal
        "date": p["date"],  # Year/date
        "methodology": "; ".join(classification["methodology"]),  # Classified methodology
        "field": "; ".join(classification["field"]),  # Classified field
        "empirical_approach": "; ".join(classification["empirical_approach"])  # Classified approach
    })

# CSV filename
csv_file = "classified_papers.csv"

# Write output to CSV
with open(csv_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "title","authors","journal","date","methodology","field","empirical_approach"
    ])
    writer.writeheader()  # Write header row
    for row in output:
        writer.writerow(row)  # Write each row

print(f"CSV saved to {csv_file}")  # Inform user
