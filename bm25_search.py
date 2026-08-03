import json
import os
import re
from rank_bm25 import BM25Okapi


class BM25Index:
    def __init__(self, storage_path: str = "bm25_index.json"):
        self.storage_path = storage_path
        self.documents: list[dict] = []  # Stores dicts with 'id', 'text', 'metadata'
        self.corpus_tokens: list[list[str]] = []
        self.bm25: BM25Okapi | None = None
        
        self._load_if_exists()

    def _tokenize(self, text: str) -> list[str]:
        """Convert raw text to lowercased alphanumeric tokens."""
        text = text.lower()
        tokens = re.findall(r"\b\w+\b", text)
        return tokens

    def _save(self) -> None:
        """Persist documents and tokens to disk."""
        data = {
            "documents": self.documents,
            "corpus_tokens": self.corpus_tokens,
        }
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _load_if_exists(self) -> None:
        """Load index from disk if storage file exists."""
        if os.path.exists(self.storage_path):
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.documents = data.get("documents", [])
                self.corpus_tokens = data.get("corpus_tokens", [])
                if self.corpus_tokens:
                    self.bm25 = BM25Okapi(self.corpus_tokens)

    def add_documents(self, docs: list[dict]) -> None:
        """
        Add new documents (each dict must have 'id', 'text', and optional 'metadata').
        Silently skips duplicates based on 'id'.
        """
        existing_ids = {doc["id"] for doc in self.documents}
        new_docs_added = False

        for doc in docs:
            doc_id = doc["id"]
            if doc_id in existing_ids:
                continue
            
            tokens = self._tokenize(doc["text"])
            self.documents.append({
                "id": doc_id,
                "text": doc["text"],
                "metadata": doc.get("metadata", {})
            })
            self.corpus_tokens.append(tokens)
            existing_ids.add(doc_id)
            new_docs_added = True

        if new_docs_added:
            self.bm25 = BM25Okapi(self.corpus_tokens)
            self._save()

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search the index for a given query and return non-zero matching docs."""
        if not self.bm25 or not self.documents:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        # Get scores for all documents
        scores = self.bm25.get_scores(query_tokens)

        # Pair document indexes with scores and filter out zero-score matches
        scored_results = [
            (idx, score) for idx, score in enumerate(scores) if score > 0
        ]
        
        # Sort by score in descending order
        scored_results.sort(key=lambda x: x[1], reverse=True)

        # Return top_k formatted results
        results = []
        for idx, score in scored_results[:top_k]:
            doc = self.documents[idx]
            results.append({
                "id": doc["id"],
                "text": doc["text"],
                "metadata": doc["metadata"],
                "score": float(score)
            })

        return results


if __name__ == "__main__":
    # Isolated sanity check
    index = BM25Index(storage_path="test_bm25.json")
    sample_docs = [
        {"id": "doc1", "text": "HNSW is a graph-based vector index algorithm for fast ANN search."},
        {"id": "doc2", "text": "BM25 Okapi is a classic probabilistic keyword search algorithm."},
        {"id": "doc3", "text": "Python automation uses scripts to streamline repetitive workflows."}
    ]
    index.add_documents(sample_docs)

    print("--- Test Search 1: Exact Technical Term ('HNSW') ---")
    results = index.search("HNSW")
    for r in results:
        print(f"[Score: {r['score']:.4f}] Doc ID: {r['id']} | Text: {r['text']}")

    print("\n--- Test Search 2: Non-matching semantic term ('car') ---")
    results = index.search("car")
    print(f"Results count: {len(results)}")

    # Clean up test artifact
    if os.path.exists("test_bm25.json"):
        os.remove("test_bm25.json")