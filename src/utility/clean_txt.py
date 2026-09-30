import json
import re
from pathlib import Path

input_file = Path("/home/austin/agentic/skin_rag/data/text.json")
output_file = Path("/home/austin/agentic/skin_rag/data/clean_text.json")

# Load extracted text
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

cleaned_data = []

for item in data:

    text = item["text"]

    # Remove spaces/tabs at the beginning and end of lines
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    # Remove leading/trailing whitespace
    text = text.strip()

    cleaned_data.append({
        "disease": item["disease"],
        "source_pdf": item["source_pdf"],
        "text": text
    })

# Save cleaned data
output_file.parent.mkdir(parents=True, exist_ok=True)

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(cleaned_data, f, indent=4, ensure_ascii=False)

print(f"Processed {len(cleaned_data)} documents")
print(f"Saved to: {output_file}")