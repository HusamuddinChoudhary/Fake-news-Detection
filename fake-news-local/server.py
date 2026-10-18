"""
Fake News Detector — No API key required.
Uses RSS feeds + NLTK VADER sentiment + heuristic NLP scoring.
Run: python server.py  →  http://localhost:5000
"""

import os, re, json, math, time, threading
from datetime import datetime, timezone
from collections import Counter

import feedparser
import requests
from flask import Flask, request, jsonify, send_from_directory

# ── NLTK bootstrap ──────────────────────────────────────────────────────────
import nltk
for pkg in ("vader_lexicon", "punkt", "stopwords", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{pkg}" if "punkt" in pkg else f"sentiment/{pkg}" if "vader" in pkg else f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

app = Flask(__name__, static_folder="static")
sia = SentimentIntensityAnalyzer()
STOP_WORDS = set(stopwords.words("english"))

# ── RSS Feeds (trusted sources, no key needed) ──────────────────────────────
RSS_FEEDS = {
    "BBC News":      "https://feeds.bbci.co.uk/news/rss.xml",
    "Reuters":       "https://feeds.reuters.com/reuters/topNews",
    "AP News":       "https://rsshub.app/apnews/topics/ap-top-news",
    "NPR":           "https://feeds.npr.org/1001/rss.xml",
    "The Guardian":  "https://www.theguardian.com/world/rss",
    "Al Jazeera":    "https://www.aljazeera.com/xml/rss/all.xml",
    "CNN":           "http://rss.cnn.com/rss/edition.rss",
    "NBC News":      "https://feeds.nbcnews.com/nbsnews/public/news",
    "ABC News":      "https://abcnews.go.com/abcnews/topstories",
    "Sky News":      "https://feeds.skynews.com/feeds/rss/world.xml",
}

# ── Pattern lists ────────────────────────────────────────────────────────────
CLICKBAIT_PATTERNS = [
    r"\byou won'?t believe\b", r"\bshocking(ly)?\b", r"\bunbelievable\b",
    r"\bmind.?blow(ing)?\b", r"\bexplosive\b", r"\bbombshell\b",
    r"\bGOING VIRAL\b", r"\bSHARE BEFORE (DELETED|REMOVED)\b",
    r"\bwake up\b", r"\bsheeple\b", r"\bhoax\b", r"\bconspiracy\b",
    r"\bdeep state\b", r"\bcrisis actor\b", r"\bcensored\b",
    r"\bthey don'?t want you to know\b", r"\bmainstream media\b",
    r"\bone weird trick\b", r"\bdoctors hate\b", r"\bmiracle cure\b",
    r"\b100%\s*(proven|confirmed|guaranteed)\b", r"\bsecret (they|the government)\b",
    r"\bnew world order\b", r"\bSTOP SHARING\b", r"\bURGENT\b",
    r"\bBREAKING\s*[!:]\s*[A-Z]", r"\b(MUST|NEED TO)\s*READ\b",
]

CREDIBILITY_MARKERS = [
    r"\baccording to\b", r"\breported by\b", r"\bconfirmed by\b",
    r"\bstudy (shows|finds|suggests)\b", r"\bresearch (indicates|shows)\b",
    r"\bofficials? (say|said|announced)\b", r"\bspokesperson (said|stated)\b",
    r"\bpublished in\b", r"\bper\b", r"\bstated that\b",
    r"\bin a statement\b", r"\bpress release\b", r"\banalysis (shows|finds)\b",
    r"\bdata (shows|suggests|indicates)\b", r"\bexpert(s)?\b",
    r"\bprofessor\b", r"\bscientist(s)?\b", r"\bjournalist\b",
]

SATIRE_MARKERS = [
    r"\bsatire\b", r"\bparody\b", r"\bthe onion\b", r"\bbabylon bee\b",
    r"\bnot real\b", r"\bfictional\b", r"\bopinion\b", r"\bsatirically\b",
]

# ── Live news cache ──────────────────────────────────────────────────────────
_news_cache = {"articles": [], "ts": 0}
_cache_lock = threading.Lock()
CACHE_TTL = 300  # 5 minutes


def fetch_live_news(force=False):
    """Fetch and cache articles from all RSS feeds."""
    with _cache_lock:
        if not force and time.time() - _news_cache["ts"] < CACHE_TTL:
            return _news_cache["articles"]
        articles = []
        for source, url in RSS_FEEDS.items():
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:8]:
                    title = entry.get("title", "").strip()
                    summary = re.sub(r"<[^>]+>", "", entry.get("summary", "")).strip()
                    link = entry.get("link", "")
                    pub = entry.get("published", "")
                    if title:
                        articles.append({
                            "source": source,
                            "title": title,
                            "summary": summary[:300] if summary else "",
                            "url": link,
                            "published": pub,
                        })
            except Exception:
                pass
        _news_cache["articles"] = articles
        _news_cache["ts"] = time.time()
        return articles


def get_keywords(text, n=12):
    """Extract top non-stop keywords from text."""
    words = [w.lower() for w in word_tokenize(text) if w.isalpha() and len(w) > 2]
    keywords = [w for w in words if w not in STOP_WORDS]
    return [w for w, _ in Counter(keywords).most_common(n)]


def cross_reference(text, articles):
    """Check how many trusted articles share keywords with the input text."""
    kws = set(get_keywords(text, 12))
    if not kws:
        return 0, []
    matches = []
    for art in articles:
        art_kws = set(get_keywords(art["title"] + " " + art["summary"], 12))
        overlap = kws & art_kws
        if len(overlap) >= 2:
            matches.append({"source": art["source"], "title": art["title"],
                            "url": art["url"], "overlap": len(overlap)})
    matches.sort(key=lambda x: x["overlap"], reverse=True)
    return len(matches), matches[:6]


def count_patterns(text, patterns):
    count = 0
    found = []
    for p in patterns:
        m = re.findall(p, text, re.IGNORECASE)
        if m:
            count += len(m)
            found.append(re.sub(r"\\b|\\s\*|\\.\\?", " ", p).strip())
    return count, found


def score_text(text):
    """
    Heuristic fake-news scoring. Returns real_probability (0-100).
    """
    score = 50  # neutral start
    flags = []
    positives = []

    # ── 1. Sentiment (genuine news is moderately negative or neutral) ────────
    vs = sia.polarity_scores(text)
    compound = vs["compound"]
    if abs(compound) > 0.75:
        score -= 12
        flags.append(f"Extremely emotional language (sentiment: {compound:+.2f})")
    elif abs(compound) > 0.5:
        score -= 5
    else:
        score += 5
        positives.append("Measured, neutral language tone")

    # ── 2. Clickbait / fake patterns ────────────────────────────────────────
    cb_count, cb_found = count_patterns(text, CLICKBAIT_PATTERNS)
    if cb_count >= 4:
        score -= 22
        flags.append(f"Heavy clickbait language detected ({cb_count} triggers)")
    elif cb_count >= 2:
        score -= 14
        flags.append(f"Multiple clickbait phrases ({cb_count} triggers)")
    elif cb_count == 1:
        score -= 6
        flags.append(f"Clickbait phrase detected")

    # ── 3. Credibility markers ───────────────────────────────────────────────
    cred_count, _ = count_patterns(text, CREDIBILITY_MARKERS)
    if cred_count >= 3:
        score += 15
        positives.append(f"Multiple credibility markers ({cred_count} found)")
    elif cred_count >= 1:
        score += 7
        positives.append(f"Credibility marker present ({cred_count})")
    else:
        score -= 5
        flags.append("No attribution or credibility markers found")

    # ── 4. ALL CAPS ratio ────────────────────────────────────────────────────
    words = text.split()
    caps_ratio = sum(1 for w in words if w.isupper() and len(w) > 2) / max(len(words), 1)
    if caps_ratio > 0.2:
        score -= 15
        flags.append(f"Excessive capitalization ({caps_ratio:.0%} of words in CAPS)")
    elif caps_ratio > 0.1:
        score -= 7
        flags.append("Elevated capitalization")

    # ── 5. Excessive punctuation ─────────────────────────────────────────────
    excl = text.count("!!")
    if excl >= 3:
        score -= 10
        flags.append(f"Excessive exclamation marks ({excl} double!!)")
    elif excl >= 1:
        score -= 4

    # ── 6. Text length (very short texts are less reliable) ──────────────────
    word_count = len(words)
    if word_count < 10:
        score -= 8
        flags.append("Very short text — hard to verify")
    elif word_count > 80:
        score += 5
        positives.append("Substantial article length")

    # ── 7. Satire markers ────────────────────────────────────────────────────
    sat_count, _ = count_patterns(text, SATIRE_MARKERS)
    if sat_count:
        score -= 20
        flags.append("Satire/parody markers detected")

    # ── 8. URL-like references (sourced claims) ───────────────────────────────
    urls = re.findall(r"https?://\S+", text)
    if urls:
        score += 5
        positives.append(f"Contains source URL(s)")

    # ── Clamp ────────────────────────────────────────────────────────────────
    score = max(5, min(95, int(round(score))))
    return score, flags, positives


def analyze(text):
    """Full analysis pipeline."""
    articles = fetch_live_news()

    # Heuristic scoring
    real_prob, flags, positives = score_text(text)

    # Cross-reference with live trusted news
    match_count, matched_articles = cross_reference(text, articles)
    if match_count >= 5:
        real_prob = min(95, real_prob + 20)
        positives.append(f"Topic corroborated by {match_count} trusted news sources")
    elif match_count >= 2:
        real_prob = min(92, real_prob + 12)
        positives.append(f"Topic found in {match_count} trusted news sources")
    elif match_count == 1:
        real_prob = min(88, real_prob + 6)
        positives.append("Topic partially corroborated by 1 trusted source")
    else:
        real_prob = max(5, real_prob - 8)
        flags.append("No matching story found in trusted news sources")

    real_prob = max(5, min(95, real_prob))
    verdict = "REAL" if real_prob >= 50 else "FAKE"

    # Confidence label
    if real_prob >= 75:
        confidence = "High confidence"
    elif real_prob >= 55:
        confidence = "Moderate confidence"
    elif real_prob >= 45:
        confidence = "Low confidence — unable to clearly verify"
    elif real_prob >= 25:
        confidence = "Likely false"
    else:
        confidence = "High confidence this is false/misleading"

    return {
        "verdict": verdict,
        "real_probability": real_prob,
        "fake_probability": 100 - real_prob,
        "confidence": confidence,
        "red_flags": flags,
        "positives": positives,
        "matched_sources": matched_articles,
        "total_live_articles": len(articles),
    }


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.post("/api/check")
def check():
    text = ((request.get_json(silent=True) or {}).get("text") or "").strip()
    if len(text) < 10:
        return jsonify(error="Please enter a longer headline or article (min 10 chars)."), 400
    try:
        return jsonify(analyze(text))
    except Exception as e:
        return jsonify(error=f"Analysis failed: {e}"), 500


@app.get("/api/live-news")
def live_news():
    try:
        articles = fetch_live_news()
        return jsonify({"articles": articles[:40], "count": len(articles)})
    except Exception as e:
        return jsonify(error=str(e)), 500


if __name__ == "__main__":
    print("⚡  Pre-fetching live news...")
    fetch_live_news(force=True)
    print(f"✅  Loaded {len(_news_cache['articles'])} articles from trusted sources")
    app.run(host="127.0.0.1", port=5000, debug=False)
