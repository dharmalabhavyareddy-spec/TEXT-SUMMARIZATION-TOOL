# TEXT-SUMMARIZATION-TOOL
# summarizer.py
# A lightweight NLP summarizer: TF-IDF TextRank-style extractive core + optional rewrite.

import re
import math
from typing import List, Tuple
from dataclasses import dataclass

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure punkt is available for sentence tokenization
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# ---------------------------
# Utilities
# ---------------------------

_sentsplitter = nltk.data.load('tokenizers/punkt/english.pickle')

def sent_tokenize(text: str) -> List[str]:
    # Robust to newlines and spacing
    text = re.sub(r'\s+', ' ', text).strip()
    return [s.strip() for s in _sentsplitter.tokenize(text) if s.strip()]

def normalize_spaces(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

def clean_text(text: str) -> str:
    # Preserve sentence punctuation but normalize whitespace
    text = normalize_spaces(text)
    return text

def build_similarity_matrix(sentences: List[str]) -> List[List[float]]:
    # TF-IDF vectors for sentences
    tfidf = TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        ngram_range=(1, 2),
        max_df=0.9,
        min_df=1,
        norm='l2'
    )
    X = tfidf.fit_transform(sentences)
    sim = cosine_similarity(X)
    # Zero diagonal (no self-similarity for ranking)
    for i in range(sim.shape[0]):
        sim[i, i] = 0.0
    return sim

def textrank_scores(sim_mat, damping: float = 0.85, max_iter: int = 100, tol: float = 1e-6) -> List[float]:
    import numpy as np
    n = sim_mat.shape[0]
    if n == 0:
        return []
    # Row-normalize to stochastic matrix
    S = sim_mat.copy()
    row_sums = S.sum(axis=1, keepdims=True)
    # Handle rows with zero sum
    S = np.divide(S, row_sums, out=np.zeros_like(S), where=row_sums != 0)
    # Power iteration
    pr = np.ones(n) / n
    teleport = (1 - damping) / n
    for _ in range(max_iter):
        prev = pr.copy()
        pr = teleport + damping * S.T.dot(pr)
        if np.linalg.norm(pr - prev, 1) < tol:
            break
    return pr.tolist()

def select_top_sentences(sentences: List[str], scores: List[float], max_sentences: int = 3) -> List[Tuple[int, str, float]]:
    # Keep original order but select by score
    ranked = sorted(list(enumerate(zip(sentences, scores))), key=lambda x: x[1][1], reverse=True)
    top_idx = sorted([i for i, _ in ranked[:max_sentences]])
    return [(i, sentences[i], scores[i]) for i in top_idx]

def limit_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip(",;:") + "…"

def keyword_weights(corpus: List[str], top_k: int = 12) -> List[str]:
    # Simple corpus-level TF-IDF to estimate topic words
    vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 1))
    X = vec.fit_transform(corpus)
    means = X.mean(axis=0).A1
    vocab = vec.get_feature_names_out()
    idx = means.argsort()[::-1][:top_k]
    return [vocab[i] for i in idx]

def compress_sentence(sentence: str, keep_terms: List[str]) -> str:
    # Lightweight compression: prefer clauses with key terms and remove filler
    # 1) Drop some boilerplate phrases
    s = re.sub(r'\b(in order to|as well as|due to the fact that)\b', 'to', sentence, flags=re.I)
    s = re.sub(r'\b(it is important to note that|it should be noted that)\b', '', s, flags=re.I)
    s = re.sub(r'\s+', ' ', s).strip(",;:. ").strip()

    # 2) If sentence is long, keep clause around key terms
    if len(s.split()) > 28 and keep_terms:
        # Split on commas and "that/which" to find salient chunk
        clauses = re.split(r',| which | that ', s)
        # Score clauses by presence of keep terms
        def score(cl):
            w = cl.lower().split()
            return sum(1 for t in keep_terms if t in w)
        clauses_sorted = sorted(clauses, key=score, reverse=True)
        if clauses_sorted:
            s = clauses_sorted[0].strip().rstrip(",;:.")
    # 3) Trim to ~24 words
    s = limit_words(s, 24)
    return s

@dataclass
class SummaryResult:
    sentences: List[str]
    extractive_summary: str
    abstractive_summary: str

class NLPSummarizer:
    def __init__(self, max_sentences: int = 3, max_words: int = 80):
        self.max_sentences = max_sentences
        self.max_words = max_words

    def summarize(self, text: str) -> SummaryResult:
        text = clean_text(text)
        sentences = sent_tokenize(text)
        if not sentences:
            return SummaryResult([], "", "")

        sim = build_similarity_matrix(sentences)
        scores = textrank_scores(sim)
        top = select_top_sentences(sentences, scores, max_sentences=self.max_sentences)
        extractive = " ".join([s for _, s, _ in top])
        extractive = limit_words(extractive, self.max_words)

        # Lightweight abstractive pass: keyword-guided compression of each extracted sentence
        keeps = keyword_weights(sentences, top_k=12)
        compressed = [compress_sentence(s, keeps) for _, s, _ in top]
        abstractive = " ".join(compressed)
        abstractive = limit_words(abstractive, self.max_words - 10)  # slightly shorter

        return SummaryResult(sentences, extractive, abstractive)

# ---------------------------
# Demo
# ---------------------------

if __name__ == "__main__":
    demo_article = """
    Artificial intelligence (AI) is transforming a wide range of industries by automating tasks,
    enhancing decision-making, and enabling new products and services. In healthcare, AI models
    help analyze medical images, predict patient risks, and streamline operations. Financial
    institutions use AI to detect fraud, assess creditworthiness, and personalize customer
    experiences. Manufacturers apply AI for predictive maintenance and quality control, reducing
    downtime and waste. While the benefits are significant, organizations face challenges such as
    data privacy concerns, model bias, regulatory uncertainty, and the need for specialized talent.
    Best practices include establishing clear governance, investing in high-quality labeled data,
    selecting interpretable models when possible, and continuously monitoring model performance
    in production. Looking ahead, advances in foundation models, multimodal learning, and edge AI
    are expected to expand capabilities and lower adoption barriers across sectors.
    """

    tool = NLPSummarizer(max_sentences=3, max_words=70)
    result = tool.summarize(demo_article)

    print("=== INPUT (truncated) ===")
    print(limit_words(clean_text(demo_article), 80))
    print("\n=== EXTRACTIVE SUMMARY ===")
    print(result.extractive_summary)
    print("\n=== CONCISE (LIGHT ABSTRACTION) ===")
    print(result.abstractive_summary)
