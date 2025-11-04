# ⚙️ Lazy import version — avoids crashing on Render due to high memory at startup

import numpy as np

# Do NOT import SentenceTransformer or transformers here
# Import them only when the function runs

def find_semantic_matches(claim, filtered_articles, threshold=0.75, top_k=5, model=None):
    """
    Computes semantic similarity between a claim and pre-filtered articles.
    Each article combines title + snippet for richer understanding.
    """
    if not filtered_articles:
        print("⚠️ No credible articles provided for semantic comparison.")
        return []

    # 🧠 Lazy import inside the function
    from ..model_loader import get_embedding_model
    from sentence_transformers import util

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

    matches.sort(key=lambda x: x["final_score"], reverse=True)
    return matches[:top_k]
