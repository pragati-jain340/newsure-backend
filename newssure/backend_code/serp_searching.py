import requests
import spacy
import re
import json

SERP_API_KEY = "ea89ee70c31f83aade9cf6d1bfb04cff3f2046976e8cc323515719f801a91ae8"
nlp = spacy.load("en_core_web_sm")

def finding_related_article(claim):
    """Fetch top 10 related articles from SERP API using the full claim text."""
    
    def extract_keywords(text):
        doc = nlp(text)
        important_labels = {"PERSON", "ORG", "GPE", "MONEY", "PERCENT", "QUANTITY", "CARDINAL", "DATE"}
        
        # Extract entities and key words
        ents = [ent.text for ent in doc.ents if ent.label_ in important_labels]
        nouns_verbs = [t.text for t in doc if t.pos_ in ["NOUN", "PROPN", "VERB"] and len(t.text) > 2]
        
        # Combine, remove duplicates, and keep order
        keywords = list(dict.fromkeys(ents + nouns_verbs))
        
        # Clean and remove near-duplicates (like "Donald Trump" + "Trump")
        cleaned = []
        seen = set()
        for kw in keywords:
            kw = re.sub(r"[^\w\s$€£-]", "", kw).strip()
            lower_kw = kw.lower()
            if kw and lower_kw not in seen:
                seen.add(lower_kw)
                cleaned.append(kw)
        
        return " ".join(cleaned)


    # Keep the full claim as query to ensure exact results
    optimized_keywords = extract_keywords(claim)
    query = claim.strip()  # use full text for actual search
    print(f"[INFO] Full-text search query: {query}")
    print(f"[INFO] Extracted keywords (for reference): {optimized_keywords}")

    params = {
        "engine": "google",
        "q": query,  # ✅ Use the full claim here
        "num": 10,
        "api_key": SERP_API_KEY
    }

    articles = []
    try:
        res = requests.get("https://serpapi.com/search", params=params, timeout=10)
        data = res.json()
        for item in data.get("organic_results", []):
            articles.append({
                "title": item.get("title"),
                "url": item.get("link"),
                "snippet": item.get("snippet"),
            })

        if not articles:
            print("[WARN] No results found for query.")

    except Exception as e:
        print(f"[Error] SERP API failed: {e}")

    return{
        "stage": "serp_search",
        "status": "success" if articles else "no_results",
        "input_claim": claim,
        "search_query": query,
        "keywords_used": optimized_keywords,
        "articles": articles,  # renamed to simpler key
        "article_count": len(articles)
    }


# Example
if __name__ == "__main__":
    claim_example = "@nation_first21 3 PATANJALL PATANJALD Patanjali launch 6G Smartphone,big step in Indian manufacturing with 250 MP camera ,200 hundren watt fast Charging With price range 25000 - 33000 Swadeshi tech innovation"
    result = finding_related_article(claim_example)
    # print(result["search_query"])
    # print(result["retrieved_articles"])

    print(result.get("articles", []))
    # print(json.dumps(result, indent=2))
