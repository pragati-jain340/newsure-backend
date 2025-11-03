# TruthScope Stage 6 – Semantic Embedding & Verification
# ------------------------------------------------------
# Takes credible, pre-filtered articles (from Stage 5)
# and performs semantic similarity between claim and
# article content (title + snippet) using embeddings.
# Returns similarity scores and optionally helps compute TruthScore.

from sentence_transformers import util
from ..model_loader import get_embedding_model


# from finding_credibilty import simulate_domain_check
# from serp_searching import finding_related_article


# ----------------------------
# Load or download model
# ----------------------------
model = get_embedding_model()

# ----------------------------
# Find semantic matches
# ----------------------------
def find_semantic_matches(claim, filtered_articles, threshold=0.75, top_k=5, model=None):
    """
    Computes semantic similarity between a claim and pre-filtered articles.
    Each article combines title + snippet for richer understanding.
    """
    if not filtered_articles:
        print("⚠️ No credible articles provided for semantic comparison.")
        return []

    if model is None:
        model = get_embedding_model()

    combined_texts = [f"{a['title']} {a.get('snippet', '')}" for a in filtered_articles]
    claim_emb = model.encode(claim, convert_to_tensor=True)
    article_embs = model.encode(combined_texts, convert_to_tensor=True)

    similarities = util.cos_sim(claim_emb, article_embs)[0].cpu().numpy()
    
    matches = []
    for art, sim in zip(filtered_articles, similarities):
        sim_score = float(sim)
        if sim_score >= threshold:
            credibility = art.get("credibility", 50)
            weight = art.get("weight", 0.5)
            trust_label = art.get("trust_label", "Unknown")

            # Composite ranking score
            final_score = sim_score * (credibility / 100) * weight

            matches.append({
                "title": art["title"],
                "url": art["url"],
                "snippet": art.get("snippet", ""),
                "credibility": credibility,
                "weight": weight,
                "trust_label": trust_label,
                "similarity": round(sim_score, 3),
                "final_score": round(final_score, 3)
            })

    # Sort by composite ranking
    matches.sort(key=lambda x: x["final_score"], reverse=True)

    # Keep only top_k results 
    top_matches = matches[:top_k]

    # Sort by (similarity × credibility weight)
    # matches.sort(key=lambda x: x["similarity"] * x["weight"], reverse=True)
    return top_matches


# ----------------------------
# Example Run (for local testing)
# ----------------------------
# if __name__ == "__main__":
#     claim_example = "The Eiffel Tower is located in Berlin."

#     # Stage 4 – SERP Search
#     print("\n[STEP 1] 🔍 Searching related articles...")
#     serp_results = finding_related_article(claim_example)
#     print(f"[INFO] Found {serp_results['total_results']} related articles.")

#     # Stage 5 – Domain Credibility
#     print("\n[STEP 2] 🌐 Evaluating domain credibility...")
#     credibility_results = simulate_domain_check(serp_results["retrieved_articles"])
#     credible_articles = credibility_results["filtered_articles"]

#     # Stage 6 – Semantic Embedding
#     print("\n[STEP 3] 🧠 Performing semantic similarity analysis...")
#     model = load_or_download_model()
#     semantic_matches = find_semantic_matches(claim_example, credible_articles, threshold=0.7, model=model)

#     # Prepare output summary
#     output = {
#         "claim": claim_example,
#         "num_matches": len(semantic_matches),
#         "avg_similarity": round(np.mean([m["similarity"] for m in semantic_matches]), 3) if semantic_matches else 0,
#         "matches": semantic_matches
#     }

#     # Print JSON-like structured output
#     print("\n✅ Final Semantic Verification Results:")
#     print(json.dumps(output, indent=2))

#     # Save results
#     with open("semantic_verification_results.json", "w", encoding="utf-8") as f:
#         json.dump(output, f, indent=2)
#         print("\n📁 Saved detailed results → semantic_verification_results.json")
