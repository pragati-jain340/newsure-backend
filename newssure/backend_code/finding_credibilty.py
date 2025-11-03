# TruthScope Stage 5 – Domain Credibility Scoring + Weight Assignment
# ---------------------------------------------------------------
# Uses MBFC dataset to compute credibility scores for retrieved articles
# Combines bias, factual reporting, and credibility into a weighted score (0–100)
# Outputs labeled articles ready for semantic verification

import json
import numpy as np
import tldextract
from .serp_searching import finding_related_article
from dotenv import load_dotenv
import os


# Go two directories up (from backend_code → newssure → Backend)
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(dotenv_path)

# ----------------------------
# MBFC API Config (if using API instead of local JSON)
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
RAPIDAPI_HOST = os.getenv("RAPIDAPI_HOST")

# === CONFIG ===
CACHE_FILE = "mbfc_data.json"
CACHE_EXPIRY_HOURS = 24*15  # optional - re-fetch after 15 day

# ----------------------------
# Extract domain
# ----------------------------
def extract_domain(url):
    ext = tldextract.extract(url)
    return f"{ext.domain}.{ext.suffix}"


# ----------------------------
# Compute hybrid credibility score
# ----------------------------
def compute_credibility_score(entry):
    """Compute a weighted credibility score (0-100) using bias, factuality, and reliability."""

    # Weight lookup tables
    factual_weights = {
        "very high": 1.0, "high": 0.85, "mostly factual": 0.70,
        "mixed": 0.6, "low": 0.35, "very low": 0.1, "n/a": 0.5, "unknown": 0.5
    }
    bias_weights = {
        "extreme left": 0.3, "left": 0.5, "left-center": 0.7,
        "center": 1.0, "right-center": 0.7, "right": 0.5,
        "extreme right": 0.3, "least biased": 0.9,
        "pro-science": 1.0, "conspiracy-pseudoscience": 0.0, "unknown": 0.5
    }
    credibility_weights = {
        "high": 0.9, "medium": 0.65, "low": 0.4,
        "very low": 0.3, "n/a": 0.5, "unknown": 0.5
    }

    # Extract fields safely
    factual = entry.get("Factual Reporting", "unknown").strip().lower()
    bias = entry.get("Bias", "unknown").strip().lower()
    cred = entry.get("Credibility", "unknown").strip().lower()

    # Map each component
    factual_score = factual_weights.get(factual, 0.5)
    bias_score = bias_weights.get(bias, 0.5)
    cred_score = credibility_weights.get(cred, 0.5)

    # Weighted combination (factual = 50%, bias = 30%, credibility = 20%)
    final_score = (factual_score * 0.5) + (bias_score * 0.3) + (cred_score * 0.2)

    return round(final_score * 100, 2)  # Normalize to 0–100 scale


# ----------------------------
# Domain credibility check
# ----------------------------
def simulate_domain_check(retrieved_articles):
    """
    Reads MBFC dataset, extracts domains, computes credibility, and assigns trust labels.
    Returns filtered credible articles + all results with trust weights.
    """

    if not retrieved_articles:
        return {"filtered_articles": [], "avg_score": 0}

    try:
        mbfc_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),  # go up to Backend/
            "app",
            "assets",
            "mbfc_data.json"
        )
        with open(mbfc_path, "r", encoding="utf-8") as f:
            mbfc_data = json.load(f)["data"]
    except Exception as e:
        print(f"⚠️ Could not load MBFC dataset: {e}")
        return {"filtered_articles": [], "avg_score": 0}

    results, credible_articles = [], []

    for article in retrieved_articles:
        url = article.get("url")
        title = article.get("title", "Untitled")
        snippet = article.get("snippet", "")
        if not url:
            continue

        domain = extract_domain(url)
        found_entry = None

        for site in mbfc_data:
            source_url = site.get("Source URL", "").lower()
            if domain in source_url:
                found_entry = site
                break

        if found_entry:
            score = compute_credibility_score(found_entry)
            bias = found_entry.get("Bias", "Unknown")
            factuality = found_entry.get("Factual Reporting", "Unknown")
            credibility_label = found_entry.get("Credibility", "Unknown")
        else:
            score = 50  # Neutral default
            bias, factuality, credibility_label = "Unknown", "Unknown", "Not Rated"

        # --------------------------
        # Assign trust labels + weights
        # --------------------------
        if score >= 80:
            label = "Trusted"
            trust_label = "Trustworthy"
            weight = 1.0
        elif score >= 60:
            label = "Mostly Reliable"
            trust_label = "Can Be Considered"
            weight = 0.7
        elif score >= 40:
            label = "Questionable"
            trust_label = "Low Credibility"
            weight = 0.5
        else:
            label = "Unreliable"
            trust_label = "Untrustworthy"
            weight = 0.2

        # Store full result
        results.append({
            "title": title,
            "snippet": snippet,
            "url": url,
            "domain": domain,
            "credibility_score": score,
            "bias": bias,
            "factuality": factuality,
            "credibility_label": credibility_label,
            "label": label,
            "trust_label": trust_label,
            "weight": weight
        })

        # Keep articles ≥ 40 credibility for semantic stage
        if score >= 40:
            credible_articles.append({
                "title": title,
                "url": url,
                "snippet": snippet,
                "credibility": score,
                "trust_label": trust_label,
                "weight": weight
            })
    

    avg_score = round(np.mean([r["credibility_score"] for r in results]), 2)
    print(f"[INFO] Domain credibility check complete - {len(credible_articles)} usable URLs selected.")

    for art in credible_articles:
        print(f" - [{art['trust_label']}] {art['title']} ({art['credibility']})")

    return {"avg_score": avg_score, "filtered_articles": credible_articles, "results": results}


# ----------------------------
# Full pipeline (testing)
# ----------------------------
if __name__ == "__main__":
    claim = "Meta announced an AI model named LLaMA in 2023."

    print("\n[STEP 1] 🔍 Searching related articles...")
    serp_output = finding_related_article(claim)

    print(f"\n[STEP 2] 🌐 Found {serp_output['total_results']} articles, checking domain credibility...")
    credibility_results = simulate_domain_check(serp_output["articles"])

    # ✅ Clean Summary Output
    print("\n✅ Final Summary:")
    print(f"Average Credibility Score: {credibility_results['avg_score']}")

    for art in credibility_results['filtered_articles']:
        print(f" - [{art['trust_label']}] {art['title']} ({art['credibility']}%)")

    # Optional: save full results to file
    with open("credibility_results.json", "w", encoding="utf-8") as f:
        json.dump(credibility_results, f, indent=2)
        print("\n📁 Saved detailed results to credibility_results.json")
