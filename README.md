# Evidence-Aware RAG Answer Verification

Detects LLM hallucinations in RAG system using evidence grounding.

## Pipeline
1. Claim Extraction
2. Evidence Retrieval (TF-IDF + Cosine Similarity)
3. Verification (Threshold 0.5)

## Results
- Correct Answer "Thali cannot be returned": Score 0.75+ -> SUPPORTED ✅
- Hallucinated Answer "Thali can be returned in 30 days": Score 0.294 -> NOT_SUPPORTED ❌ (Hallucination Detected)

## How to Run
pip install scikit-learn
python rag_dynamic_chunking.py

## Tech
Python 3.14 compatible, scikit-learn (Production would use sentence-transformers)
