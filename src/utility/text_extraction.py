import fitz
from pathlib import Path
import json

pdf_folder = Path("/home/austin/agentic/skin_rag/dermnetfiles")
output_file = Path("/home/austin/agentic/skin_rag/data/text.json")

data = []

for pdf_file in pdf_folder.glob("*.pdf"):

    disease = pdf_file.stem

    doc = fitz.open(pdf_file)

    text = ""

    for page in doc:
        text += page.get_text() + "\n"

    doc.close()

    data.append({
        "disease": disease,
        "source_pdf": pdf_file.name,
        "text": text.strip()
    })

output_file.parent.mkdir(parents=True, exist_ok=True)

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f"Processed {len(data)} PDFs")
print(f"Saved to: {output_file}")