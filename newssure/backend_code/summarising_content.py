from ..model_loader import get_gemini_model
import nltk
import google.generativeai as genai
from deep_translator import GoogleTranslator
from langdetect import detect

gemini_model = get_gemini_model()  # ✅ Gemini only

# ------------------------------------------------------------
# 🌍 Language Handling
# ------------------------------------------------------------
def ensure_english(text):
    try:
        lang = detect(text)
        if lang != "en":
            text = GoogleTranslator(source="auto", target="en").translate(text)
        return text
    except Exception:
        return text


# ------------------------------------------------------------
# 🧩 Claim Relevance Filter
# ------------------------------------------------------------
def filter_relevant_sentences(claim, text, top_k=5):
    if not text.strip():
        return ""
    claim_keywords = [w.lower() for w in claim.split() if len(w) > 3]
    sentences = nltk.sent_tokenize(text)
    relevant = [s for s in sentences if any(k in s.lower() for k in claim_keywords)]
    if len(relevant) < top_k:
        relevant = sentences[:top_k]
    return " ".join(relevant)


# ------------------------------------------------------------
# 🤖 Gemini Summarization Only
# ------------------------------------------------------------
def summarize_with_gemini(claim: str, text: str) -> str:
    """
    Uses Gemini to create a claim-focused summary — summarizes only
    the parts of the article relevant to the given claim.
    If the text is short or unclear, Gemini still analyzes whether the article
    overall supports, refutes, or is neutral toward the claim.
    """
    try:
        if not text.strip():
            return ""

        # Limit text length for efficiency (~1000–1200 words)
        text = " ".join(text.split())[:7000]

        prompt = f"""
        You are an expert fact-checking assistant.

        Claim: "{claim}"

        Your task:
        1. Summarize the article **only in relation to this claim** — focus on parts that directly agree, deny, refute, or contradict the claim.
        2. Even if the article is short or vague, you must still determine whether the article overall:
           - Supports the claim
           - Refutes (contradicts/disputes) the claim
           - Or is Neutral / Unclear
        3. At the end, clearly include one sentence like:
           "Overall, the article supports the claim."
           OR
           "Overall, the article refutes the claim."
           OR
           "Overall, the article is neutral toward the claim."
        4. Write the summary in 3-6 concise sentences, keeping key facts, entities, and stance indicators
           (words like “denied”, “refuted”, “agreed”, “confirmed”, “disputed”, “claimed”, etc.).

        Article:
        {text}
        """

        response = gemini_model.generate_content(prompt)
        if not response or not getattr(response, "text", "").strip():
            raise ValueError("Empty Gemini response")

        return response.text.strip()

    except Exception as e:
        print(f"⚠️ Gemini summarization failed: {e}")
        return "Summary could not be generated due to an error."


# ------------------------------------------------------------
# 📄 Summarize Single Article
# ------------------------------------------------------------
def summarize_article(claim, text):
    text = ensure_english(text)
    relevant_text = filter_relevant_sentences(claim, text, top_k=5)
    if not relevant_text.strip():
        return "No relevant content found to summarize."
    summary = summarize_with_gemini(claim, relevant_text)
    return summary


# ------------------------------------------------------------
# 📚 Summarize Multiple Articles
# ------------------------------------------------------------
def summarize_all_articles(claim, extracted_data):
    """
    Summarizes each extracted article separately.
    Input:
        claim (str): user-submitted claim
        extracted_data (dict): output from extract_article() containing articles
    Output:
        dict: list of summarized articles, one per URL
    """
    summarized_articles = []

    for art in extracted_data.get("articles", []):
        url = art.get("url")
        title = art.get("title", "Untitled")
        text = art.get("text", "")
        credibility = art.get("credibility", 50)
        trust_label = art.get("trust_label", "Unknown")
        weight = art.get("weight", 0.5)
        similarity = art.get("similarity", 0.0)

        if not text.strip():
            print(f"⚠️ Skipping empty article: {url}")
            continue

        summary_text = summarize_article(claim, text)

        summarized_articles.append({
            "url": url,
            "title": title,
            "summary": summary_text,
            "length": len(text),
            "credibility": credibility,
            "trust_label": trust_label,
            "weight": weight,
            "similarity": similarity
        })

    print(f"[INFO] ✅ Generated {len(summarized_articles)} summaries.")

    return {
        "stage": "summarization",
        "total_summaries": len(summarized_articles),
        "summaries": summarized_articles
    }


# ------------------------------------------------------------
# 🧪 Example Run
# ------------------------------------------------------------
if __name__ == "__main__":
    claim_example = "The Eiffel Tower is located in Berlin."
    extracted_example = {
        "articles": [
            {
                "url": "http://example.com/article1",
                "title": "Facts about the Eiffel Tower",
                "text": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower.",
                "method": "scraping"
            },
            {
                "url": "http://example.com/article2",
                "title": "Tourist Attractions in Berlin",
                "text": "Berlin is known for its art scene and modern landmarks like the Berliner Philharmonie, but it does not have the Eiffel Tower.",
                "method": "api"
            }
        ]
    }
    summary_results = summarize_all_articles(claim_example, extracted_example)
    for summary in summary_results["summaries"]:
        print(f"URL: {summary['url']}\nSummary: {summary['summary']}\n")
