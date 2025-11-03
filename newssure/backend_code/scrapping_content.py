from ..model_loader import get_gemini_model
import requests
from bs4 import BeautifulSoup
from newspaper import Article
import trafilatura
import logging

model = get_gemini_model()


# ------------------------------------------------------------
# ⚙️ Setup
# ------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (compatible; TruthScopeBot/1.0; +https://truthscope.ai)"
})

# ------------------------------------------------------------
# Extraction methods
# ------------------------------------------------------------
def try_newspaper(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        if len(article.text.strip()) > 50:
            clean_text = " ".join(article.text.split())
            return article.title, clean_text, "newspaper3k"
    except Exception:
        pass
    return None


def try_trafilatura(url):
    try:
        text = trafilatura.extract(trafilatura.fetch_url(url))
        if text and len(text.strip()) > 50:
            clean_text = " ".join(text.split())
            return "Extracted via Trafilatura", clean_text, "trafilatura"
    except Exception:
        pass
    return None


def extract_with_gemini(url):
    try:
        logging.info("🔄 Gemini fallback extraction…")
        html = session.get(url, timeout=10).text
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup([
            "script", "style", "nav", "footer", "header", "aside",
            "form", "button", "noscript", "iframe", "section"
        ]):
            tag.extract()
        text = soup.get_text(separator="\n", strip=True)
        if len(text) < 200:
            return None
        model = model
        prompt = f"Extract only the main article body from this text:\n{text[:6000]}"
        response = model.generate_content(prompt)
        clean_text = " ".join(response.text.split())
        return "Extracted via Gemini", clean_text, "gemini"
    except Exception as e:
        logging.warning(f"⚠️ Gemini extraction failed: {e}")
    return None


# ------------------------------------------------------------
# ✅ FIXED: Handle multiple article objects (with credibility metadata)
# ------------------------------------------------------------
def extract_article(claim, articles):
    """
    Extracts article content while preserving credibility metadata.
    Input:
        articles (list): list of dicts from semantic stage, each with
                         url, credibility, trust_label, weight, etc.
    Output:
        dict: structured content extraction result for summarization
    """

    all_results = []

    for art in articles:
        url = art.get("url")
        if not url:
            continue

        credibility = art.get("credibility", 50)
        trust_label = art.get("trust_label", "Unknown")
        weight = art.get("weight", 0.5)
        similarity = art.get("similarity", 0.0)


        logging.info(f"📰 Extracting content from: {url}")

        extracted = None
        for fn in (try_newspaper, try_trafilatura, extract_with_gemini):
            result = fn(url)
            if result:
                clean_text = " ".join(result[1].split())
                extracted = {
                    "claim": claim,
                    "url": url,
                    "title": result[0],
                    "text": clean_text,
                    "method": result[2],
                    "length": len(clean_text),
                    "credibility": credibility,
                    "trust_label": trust_label,
                    "weight": weight,
                    "similarity": similarity
                }
                logging.info(
                    f"✅ Extracted via {result[2]} | {trust_label} ({credibility}%) | {len(clean_text)} chars"
                )
                break

        if not extracted:
            logging.warning(f"🚫 Failed to extract article: {url}")
        else:
            all_results.append(extracted)

    return {
        "stage": "content_extraction",
        "status": "success" if all_results else "no_content",
        "total_articles": len(all_results),
        "articles": all_results
    }


# ------------------------------------------------------------
# Example Usage
# ------------------------------------------------------------
if __name__ == "__main__":
    test_articles = [
        {
            "url": "https://www.bbc.com/news/world-us-canada-66823405",
            "credibility": 90,
            "trust_label": "Trusted",
            "weight": 1.0
        },
        {
            "url": "https://edition.cnn.com/2023/09/19/tech/ai-generated-images-fake-news/index.html",
            "credibility": 80,
            "trust_label": "Mostly Reliable",
            "weight": 0.8
        }
    ]

    claim = "AI-generated images are becoming more prevalent in news media."
    extraction_results = extract_article(claim, test_articles)

    print("\n=== Extraction Summary ===")
    print(f"Total Articles Extracted: {extraction_results['total_articles']}")
    for article in extraction_results["articles"]:
        print(
            f"\nURL: {article['url']}\n"
            f"Credibility: {article['credibility']} ({article['trust_label']})\n"
            f"Method: {article['method']}\n"
            f"Length: {article['length']} chars\n"
        )
