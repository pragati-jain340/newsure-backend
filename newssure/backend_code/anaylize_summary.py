from ..model_loader import get_classifier_model
import numpy as np

# ------------------------------------------------------------
# ⚙️ Setup
classifier = get_classifier_model()

# ------------------------------------------------------------
# 🧩 Helper Functions
# ------------------------------------------------------------
def classify_claim_evidence(claim, evidence):
    result = classifier(f"Premise: {evidence} Hypothesis: {claim}", top_k=None)
    if isinstance(result, list):
        result = result[0]
    label = result["label"].lower()
    score = result["score"]

    if label == "entailment":
        meaning = "supports"
    elif label == "contradiction":
        meaning = "refutes"
    else:
        meaning = "neutral"

    return meaning, round(score * 100, 2)


def classify_claim_evidence(claim, evidence):
    """
    Determines whether the article evidence supports, refutes, or is neutral to the claim.
    """
    # Flip order: hypothesis first improves clarity in negation detection
    result = classifier(f"Hypothesis: {claim} Premise: {evidence}", top_k=None)

    if isinstance(result, list):
        result = result[0]
    label = result["label"].lower()
    score = result["score"]

    if label == "entailment":
        meaning = "supports"
    elif label == "contradiction":
        meaning = "refutes"
    else:
        meaning = "neutral"

    return meaning, round(score * 100, 2)


def analyze_claim_vs_summary(claim, summary):
    """
    Adds rule-based enhancement to fix NLI misclassification when
    the article clearly denies, contradicts, or disproves a claim.
    """
    text = summary.lower()

    # Explicit cues of refutation or denial
    refute_keywords = [
        "refutes","fake", "false", "denied", "refuted", "clarified", "no such",
        "fabricated", "not true", "incorrect", "contradict", "disprove",
        "debunk", "denies", "myth", "hoax", "isn't", "is not", "wasn't"
    ]
    support_keywords = [
        "confirmed", "agreed", "verified", "approved", "affirmed",
        "announced", "declared", "supports", "proves", "true", "confirmed that"
    ]

    # Pre-label biasing before NLI
    if any(k in text for k in refute_keywords):
        rule_relation = "refutes"
    elif any(k in text for k in support_keywords):
        rule_relation = "supports"
    else:
        rule_relation = None

    # Run NLI model
    model_relation, conf = classify_claim_evidence(claim, summary)

    # Combine model + rule logic
    if rule_relation and model_relation != rule_relation:
        # Override only if explicit contradiction terms are present
        if rule_relation == "refutes":
            model_relation, conf = "refutes", min(conf + 20, 100)
        elif rule_relation == "supports":
            model_relation, conf = "supports", min(conf + 10, 100)

    # Ensure high certainty for obvious contradictions
    if model_relation == "neutral" and rule_relation:
        model_relation = rule_relation
        conf = min(conf + 25, 100)

    return {
        "relation": model_relation,
        "confidence": conf,
        "summary": summary
    }


# ------------------------------------------------------------
# 🧮 Aggregation & Final Verdict
# ------------------------------------------------------------
def aggregate_results(results):
    """
    Aggregate individual claim-summaries into one overall truth verdict.
    Now considers:
      - Relation (supports/refutes/neutral)
      - Model confidence
      - Source credibility
      - Semantic similarity
      - Assigned weight
    """
    if not results:
        return "NEUTRAL", 0.0, 0.0

    mapping = {"supports": 1, "neutral": 0, "refutes": -1}

    weighted_scores = []
    total_weight = 0

    for r in results:
        stance = mapping.get(r["relation"], 0)
        confidence = r.get("confidence", 50) / 100
        credibility = r.get("credibility", 50) / 100
        similarity = r.get("similarity", 0.5)
        user_weight = r.get("weight", 0.5)

        # Weighted composite score
        # Gives higher importance to high-credibility + high-confidence sources
        composite_weight = (0.4 * credibility) + (0.3 * confidence) + (0.2 * similarity) + (0.1 * user_weight)
        total_weight += composite_weight
        weighted_scores.append(stance * composite_weight)

    # Compute normalized weighted average stance
    stance_score = sum(weighted_scores) / total_weight if total_weight else 0

    # Weighted average confidence across all results
    avg_conf = round(np.average([r["confidence"] for r in results],
                                weights=[r.get("credibility", 50) for r in results]), 2)

    # Convert stance score to final verdict
    if stance_score > 0.25:
        final = "SUPPORTS"
    elif stance_score < -0.25:
        final = "REFUTES"
    else:
        final = "NEUTRAL"

    return final, avg_conf, round(stance_score, 2)



# ------------------------------------------------------------
# 🔗 Main Integration – Multi-URL Summary Handling
# ------------------------------------------------------------
def verify_claim_from_text(claim_text: str, summarized_output: dict):
    """
    Takes the claim and multi-URL summaries (from summarize_all_articles),
    analyzes each summary independently, and aggregates the overall verdict.
    """
    if not summarized_output or not summarized_output.get("summaries"):
        return {"error": "No summaries to analyze."}

    results = []

    for art in summarized_output["summaries"]:
        url = art.get("url")
        summary = art.get("summary", "")
        credibility = art.get("credibility", 50)
        trust_label = art.get("trust_label", "Unknown")
        similarity = art.get("similarity", 0.0)
        weight = art.get("weight", 0.5)

        if not summary.strip():
            continue

        analysis = analyze_claim_vs_summary(claim_text, summary)
        analysis.update({
            "url": url,
            "credibility": credibility,
            "trust_label": trust_label,
            "similarity": similarity,
            "weight": weight
        })
        results.append(analysis)

    final_verdict, avg_conf, stance_score = aggregate_results(results)
    avg_conf = float(avg_conf)


    # Weighted truth score considering confidence, credibility, similarity, and weight
    truth_score = np.mean([
        (
            0.5 * r["confidence"] +                # AI confidence (main driver)
            0.2 * r["credibility"] +               # Source credibility
            0.2 * (r["similarity"] * 100) +        # Semantic similarity (scaled to 0–100)
            0.1 * (r.get("weight", 0.5) * 100)     # User-defined or system weight
        )
        for r in results
    ]) if results else 0
    
    truth_score = float(round(min(truth_score, 100), 2))

    return {
        "claim": claim_text,
        "final_verdict": final_verdict,
        "truthScore": truth_score,
        "average_confidence": avg_conf,
        "weighted_stance_score": stance_score,
        "reliable_sources": results
    }


# ------------------------------------------------------------
# 🧪 Example Test
# ------------------------------------------------------------
if __name__ == "__main__":
    claim_example = "The Eiffel Tower is located in Berlin."
    summaries_example = {
        "summaries": [
            {
                "url": "https://bbc.com/news/abc",
                "summary": "The article clarifies that the Eiffel Tower is in Paris, not Berlin.",
                "credibility": 90,
                "trust_label": "Trusted",
                "weight": 1.0,
                "similarity": 0.85
            },
            {
                "url": "https://randomblog.net/eiffel",
                "summary": "Some sources falsely claimed it was in Berlin, but official reports confirm it is in Paris.",
                "credibility": 60,
                "trust_label": "Mostly Reliable",
                "weight": 0.7,
                "similarity": 0.8
            }
        ]
    }

    result = verify_claim_from_text(claim_example, summaries_example)
    print("\n🧠 Final Claim Analysis:")
    print(result)
