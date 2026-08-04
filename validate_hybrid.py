from hybrid_search import HybridSearchEngine

def run_validation():
    engine = HybridSearchEngine()

    test_queries = [
        # Group 1: Exact Technical Terms / Parameters (BM25 Strengths)
        "Adam optimizer beta1 0.9",
        "BLEU scores English-to-German",
        "Modern Prometheus Wollstonecraft",
        
        # Group 2: Broad / Conceptual Queries (Vector Strengths)
        "How does the model calculate attention between words?",
        "Why did the creator feel guilty about his monster?",
        "What training hardware and GPUs were used?"
    ]

    for q in test_queries:
        engine.hybrid_search_explained(q, top_k=2)

if __name__ == "__main__":
    run_validation()