# python -m newssure.demo_input     
"""
============================================================
TruthScope - AI + Community-Driven News Verification System
============================================================
Unified Workflow:
    1 - Input Handling (Text / Image)
    2 - image_verification (if image)
    3 - image_extraction (if image)
    4 - serp_searching (Related Article Retrieval)
    5 - finding_credibility (Domain Credibility Check)
    6 - embedding_filtering (Semantic Similarity Analysis)  
    7 - scrapping_content (Article Content Extraction)
    8 - summarising_content (Creating Summary)
    9 - anaylize_summary (Final Truth + Verdict)
============================================================"""

import os
from dotenv import load_dotenv
import time
import tempfile
from rest_framework.response import Response
import json
from newssure.backend_code.image_verfication import simulate_image_verification
from newssure.backend_code.image_extraction import run_ocr_extraction
from newssure.backend_code.serp_searching import finding_related_article
from newssure.backend_code.finding_credibilty import simulate_domain_check
from newssure.backend_code.embedding_filtering import find_semantic_matches
from newssure.backend_code.scrapping_content import extract_article
from newssure.backend_code.summarising_content import summarize_all_articles
from newssure.backend_code.anaylize_summary import verify_claim_from_text

# Go one directory up (from newssure → Backend)
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env.example")
load_dotenv(dotenv_path)

# ------------------------------------------------------------

def combine_results(anaylising_content):
    """Combine all verification layers into unified output."""

    reliable_sources = [src for src in anaylising_content.get("reliable_sources", [])]

    return {
        "claim": anaylising_content.get("claim", "Unknown claim"),
        "truthScore": anaylising_content.get("truthScore", 0),
        "finalVerdict": anaylising_content.get("final_verdict", "Uncertain"),
        "aiGenerated": False,
        "averageConfidence": anaylising_content.get("average_confidence", 0),
        
        "weightedStanceScore": anaylising_content.get("weighted_stance_score", 0),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sources": reliable_sources
}

def verify_claim(request):
    
    input_type = request.data.get('inputType')

    # STAGE 1 & 2 – IMAGE VERIFICATION + OCR EXTRACTION

    if input_type == 'image':
        image_file = request.FILES.get('file')

        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, image_file.name)

        # ✅ Safe copy (no corruption)
        if hasattr(image_file, "chunks"):  # Django UploadedFile case
            with open(temp_path, "wb+") as dest:
                for chunk in image_file.chunks():
                    dest.write(chunk)
        else:  # Local testing mode
            with open(image_file.path, "rb") as src, open(temp_path, "wb+") as dest:
                dest.write(src.read())
    

        # --- Stage 1: Check if AI generated ---
        ai_check = simulate_image_verification(image_file)
        if ai_check.get("aiGenerated"):
            return{
                "verdict": "AI-generated image",
                "truthScore": 0,
                "aiGenerated": True,
                "explanation": "Detected as AI-generated visual. No further processing required."
            }


        # --- Stage 2: OCR Extraction ---
        extracted_text = run_ocr_extraction(temp_path, visualize=True)
        claim = extracted_text 


    #  STAGE 3 – TEXT INPUT HANDLING
    
    else:
        claim = request.data.get('input', '').strip()

    if not claim:
        return {"error": "No valid text or image content found.", "status": 400}

    print(f"\n CLAIM: {claim}\n{'='*80}")


    # STAGE 4 – SERP API ARTICLE RETRIEVAL
    serp_results = finding_related_article(claim)
    print(f"[INFO] Retrieved {serp_results['article_count']} related articles.")

    # Extract only the list of articles for next stage
    retrieved_articles = serp_results.get("articles", [])


    # STAGE 5 – DOMAIN CREDIBILITY CHECK
    domain_results = simulate_domain_check(retrieved_articles)
    print(f"[INFO] Domain credibility check complete with {len(domain_results.get('filtered_articles', []))} usable URLs.")
    

    # STAGE 6 – embedding_filtering (Semantic Similarity Analysis)
    semantic_matches = find_semantic_matches(claim, domain_results.get("filtered_articles", []))
    print(f"[INFO] Semantic filtering selected {len(semantic_matches)} high-similarity URLs.")
    

    # STAGE 7 – scrapping_content (Article Content Extraction)
    summary_extract = extract_article(claim, articles=semantic_matches)
    print(f"[INFO] Extracted {summary_extract['total_articles']} full articles.")


    # STAGE 8 – summarising_content (Creating Summary)
    summarized_articles = summarize_all_articles(claim, summary_extract)


    # STAGE 9 – anaylize_summary (Final Truth + Verdict)
    anaylising_content = verify_claim_from_text(claim, summarized_articles)


    # STAGE 8 – FINAL RESULT COMBINATION
    result = combine_results(anaylising_content)

    return result


if __name__ == "__main__":
    class DummyRequest:
        def __init__(self, input_type, input_data):
            self.data = {"inputType": input_type, "input": input_data}
            self.FILES = {}

    class DummyFile:
        """Simulates a Django UploadedFile object for local image testing."""

        def __init__(self, path):
            self.name = os.path.basename(path)
            self.path = path

        def read(self):
            with open(self.path, "rb") as f:
                return f.read()

        def chunks(self, chunk_size=4096):
            with open(self.path, "rb") as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk

    # Local image test
    test_request_img = DummyRequest("image", "")
    image_path = r"C:\pragati\codes\ignite_hackathon\test\Screenshot 2025-10-12 212503.jpg"

    test_request_img.FILES["file"] = DummyFile(image_path)

    # Run full pipeline
    response_img = verify_claim(test_request_img)

    # ✅ Display output nicely
    print("\nFINAL OUTPUT:")
    print(json.dumps(response_img, indent=4, ensure_ascii=False))


