import fitz
from pathlib import Path

pdf_folder = Path("/home/austin/agentic/skin_rag/dermnetfiles")
output_folder = Path("/home/austin/agentic/skin_rag/images")

output_folder.mkdir(exist_ok=True)

for pdf_file in pdf_folder.glob("*.pdf"):

    # Disease name from PDF filename
    disease_name = pdf_file.stem

    disease_folder = output_folder / disease_name
    disease_folder.mkdir(parents=True, exist_ok=True)

    pdf = fitz.open(pdf_file)

    image_count = 0

    for page in pdf:
        images = page.get_images(full=True)

        for image in images:
            image_count += 1

            image_data = pdf.extract_image(image[0])

            extension = image_data["ext"]
            image_bytes = image_data["image"]

            image_path = disease_folder / f"image_{image_count:03d}.{extension}"

            with open(image_path, "wb") as f:
                f.write(image_bytes)

    pdf.close()

    print(f"{disease_name}: {image_count} images")