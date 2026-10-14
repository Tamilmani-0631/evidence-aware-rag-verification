import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

question = "Can I return Thali jewellery?"
retrieved_docs = [
    {"id": "doc1_p1", "text": "Policy: Thali is sacred for married women and cannot be returned under any condition."},
    {"id": "doc1_p2", "text": "Bangles can be returned within 7 days if not worn and with bill."},
]
answer_good = "No, Thali cannot be returned because it is considered sacred."
answer_bad = "Yes, Thali can be returned within 30 days for a full refund."

def extract_claims(a):
    return [s.strip() for s in re.split(r'[.!?]', a) if len(s.strip())>10]

def verify_pipeline(q, docs, ans):
    print(f"\nQ: {q}\nAns: {ans}\n")
    vectorizer = TfidfVectorizer()
    doc_texts = [d["text"] for d in docs]

    for claim in extract_claims(ans):
        all_texts = [claim] + doc_texts
        tfidf = vectorizer.fit_transform(all_texts)
        scores = cosine_similarity(tfidf[0:1], tfidf[1:])[0]
        best_idx = int(scores.argmax())
        score = float(scores[best_idx])
        verdict = "SUPPORTED" if score > 0.5 else "NOT_SUPPORTED"
        print(f"Claim: {claim}\n -> Evidence: {docs[best_idx]['text']}\n -> Score: {score:.3f} -> {verdict}\n")

print("=== TEST 1: Correct Answer ===")
verify_pipeline(question, retrieved_docs, answer_good)

print("\n=== TEST 2: Hallucinated Answer ===")
verify_pipeline(question, retrieved_docs, answer_bad)