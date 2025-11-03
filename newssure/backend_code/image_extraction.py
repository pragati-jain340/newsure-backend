import os
import cv2
import numpy as np
from paddleocr import PaddleOCR
# ----------------------------
# OCR Text Extraction

ocr = PaddleOCR(
    lang='en',
    use_textline_orientation=False,
    text_det_unclip_ratio=1.5,
    text_recognition_batch_size=4,
    textline_orientation_batch_size=1
)

def run_ocr_extraction(image_path: str, visualize: bool = False) -> str:
    """Extract text from an image using PaddleOCR, compatible with all versions."""
    print("🚀 Initializing PaddleOCR...")

    ocr = PaddleOCR(lang='en')

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ Image not found: {image_path}")

    try:
        # ✅ For PaddleOCR v4.x (newer)
        img = cv2.imread(image_path)
        results = ocr.predict(img)
    except Exception:
        # ✅ Fallback for older versions
        results = ocr.ocr(image_path)

    all_text = []
    extracted_data = []

    # Newer dict-based output
    if isinstance(results, list) and results and isinstance(results[0], dict):
        for res in results:
            texts = res.get("rec_texts", [])
            scores = res.get("rec_scores", [])
            boxes = res.get("rec_polys", [])
            for i in range(len(texts)):
                extracted_data.append({
                    "text": texts[i],
                    "confidence": float(scores[i]),
                    "bbox": np.array(boxes[i]).astype(int).tolist()
                })
                all_text.append(texts[i])
    else:
        # Legacy nested list format
        for line in results:
            for item in line:
                txt = item[1][0]
                conf = float(item[1][1])
                all_text.append(txt)
                extracted_data.append({"text": txt, "confidence": conf})

    extracted_text = " ".join(all_text).strip()
    print(f"🧾 Extracted text: {extracted_text or '⚠️ No text detected.'}")
    return extracted_text


    

if __name__ == "__main__":
    test_image = r"C:\Users\praga\OneDrive\Desktop\news_dataset\NewsSure\Backend\app\assets\Screenshot 2025-10-12 212503.jpg"  # Replace with your test image path
    extracted_text = run_ocr_extraction(test_image, visualize=False)
    print(f"\nFinal Extracted Text:\n{extracted_text}\n")