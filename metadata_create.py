from pathlib import Path
import json

image_folder = Path("/home/austin/agentic/skin_rag/images")
metadata_file = Path("/home/austin/agentic/skin_rag/meta_data.json")

metadata = []

image_id = 1

for disease_folder in image_folder.iterdir():

    if not disease_folder.is_dir():
        continue

    disease = disease_folder.name

    for image_file in disease_folder.iterdir():

        if image_file.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
            continue

        metadata.append({
            "image_id": image_id,
            "image_path": str(image_file),
            "disease": disease
        })

        image_id += 1


with open(metadata_file, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4, ensure_ascii=False)

print(f"Created metadata for {len(metadata)} images")