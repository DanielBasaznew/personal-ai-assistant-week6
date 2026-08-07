from hybrid_search import HybridSearchEngine
from query_router import parse_query_intent

def main():
    engine = HybridSearchEngine()
    available_sources = ["my_notes", "resnet_paper", "frankenstein_book", "transformer_paper"]

    queries = [
        "In my notes, what did I write about RAG?",
        "What does resnet_paper say about residual learning?",
        "Search everything for Frankenstein"
    ]

    for user_q in queries:
        # Step 1: LLM extracts intent and filters
        intent = parse_query_intent(user_q, available_sources)
        
        # Step 2: Pass cleaned query and extracted metadata filters to hybrid search
        engine.hybrid_search_explained(
            query=intent.clean_query,
            top_k=2,
            source_filter=intent.source_filter,
            document_type=intent.document_type
        )

if __name__ == "__main__":
    main()