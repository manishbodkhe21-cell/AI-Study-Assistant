from pdf2image import convert_from_path
import easyocr
import numpy as np
import json
import os

# POPPLER PATH
POPPLER_PATH = r"C:\Users\Manish Bodkhe\Downloads\Release-25.12.0-0\poppler-25.12.0\Library\bin"

# EasyOCR Reader
reader = easyocr.Reader(['en'], gpu=False)


def extract_text_from_pdf(pdf_path):

    images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)

    results = []
    filename = os.path.basename(pdf_path)

    for i, image in enumerate(images):

        image_np = np.array(image)

        text = reader.readtext(image_np, detail=0)

        page_text = " ".join(text)

        results.append({
            "file": filename,
            "page": i + 1,
            "text": page_text
        })

    # Save OCR result
    json_name = filename.replace(".pdf", ".json")

    os.makedirs("data/ocr_text", exist_ok=True)

    with open(f"data/ocr_text/{json_name}", "w") as f:
        json.dump(results, f, indent=4)

    return results