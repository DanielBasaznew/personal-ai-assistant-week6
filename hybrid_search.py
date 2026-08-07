import os
import chromadb
from chromadb.utils import embedding_functions
from rich.console import Console
from rich.table import Table

from bm25_search import BM25Index

console = Console()


def reciprocal_rank_fusion(
    bm25_results: list[dict], 
    semantic_results: list[dict], 
    k: int = 60,
    bm25_weight: float = 0.5,
    semantic_weight: float = 0.5
) -> list[dict]:
    """
    Merges two ranked lists using Reciprocal Rank Fusion (RRF).
    Calculates score = weight * (1 / (rank + k)) for each document.
    """
    rrf_scores: dict[str, float] = {}
    doc_lookup: dict[str, dict] = {}
    sources: dict[str, list[str]] = {}

    # Process BM25 results
    for rank, doc in enumerate(bm25_results, start=1):
        doc_id = doc["id"]
        score = bm25_weight * (1.0 / (rank + k))
        
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + score
        doc_lookup[doc_id] = doc
        
        if doc_id not in sources:
            sources[doc_id] = []
        sources[doc_id].append("BM25")

    # Process Semantic results
    for rank, doc in enumerate(semantic_results, start=1):
        doc_id = doc["id"]
        score = semantic_weight * (1.0 / (rank + k))
        
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + score
        if doc_id not in doc_lookup:
            doc_lookup[doc_id] = doc
            
        if doc_id not in sources:
            sources[doc_id] = []
        sources[doc_id].append("Semantic")

    # Combine into ranked list
    fused_results = []
    for doc_id, score in rrf_scores.items():
        doc = doc_lookup[doc_id].copy()
        doc["rrf_score"] = score
        doc["sources"] = sources[doc_id]
        fused_results.append(doc)

    # Sort descending by fused RRF score
    fused_results.sort(key=lambda x: x["rrf_score"], reverse=True)
    return fused_results


class HybridSearchEngine:
    def __init__(self, bm25_path: str = "bm25_index.json", chroma_path: str = "../private-knowledge-assistant-week5/data/chroma_db"):
        # 1. Initialize BM25
        self.bm25_index = BM25Index(storage_path=bm25_path)

        # 2. Initialize ChromaDB safely
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.chroma_collection = self.chroma_client.get_collection(
            name="private_knowledge_base",
            embedding_function=self.emb_fn
        )

    def search(
        self, 
        query: str, 
        top_k: int = 5,
        source_filter: str | None = None,
        document_type: str | None = None
    ) -> list[dict]:
        """
        Performs hybrid search with optional metadata pre-filtering.
        """
        fetch_candidate_count = top_k * 3

        # 1. Fetch BM25 candidates
        bm25_candidates = self.bm25_index.search(query, top_k=fetch_candidate_count)
        
        # Filter BM25 candidates in-memory if metadata filters exist
        if source_filter:
            bm25_candidates = [
                doc for doc in bm25_candidates 
                if doc.get("metadata", {}).get("document_name") == source_filter
            ]
        if document_type:
            bm25_candidates = [
                doc for doc in bm25_candidates 
                if doc.get("metadata", {}).get("document_type") == document_type
            ]

        # 2. Build ChromaDB 'where' filter clause
        where_clause = None
        if source_filter and document_type:
            where_clause = {
                "$and": [
                    {"document_name": source_filter},
                    {"document_type": document_type}
                ]
            }
        elif source_filter:
            where_clause = {"document_name": source_filter}
        elif document_type:
            where_clause = {"document_type": document_type}

        # Fetch ChromaDB vector candidates with metadata filter applied
        chroma_kwargs = {
            "query_texts": [query],
            "n_results": fetch_candidate_count
        }
        if where_clause:
            chroma_kwargs["where"] = where_clause

        chroma_raw = self.chroma_collection.query(**chroma_kwargs)

        semantic_candidates = []
        if chroma_raw and chroma_raw.get("documents") and chroma_raw["documents"][0]:
            docs = chroma_raw["documents"][0]
            ids = chroma_raw["ids"][0]
            metas = chroma_raw["metadatas"][0] if chroma_raw.get("metadatas") else [{}] * len(docs)

            for doc_id, text, meta in zip(ids, docs, metas):
                semantic_candidates.append({
                    "id": doc_id,
                    "text": text,
                    "metadata": meta or {}
                })

        # 3. Fuse filtered candidates using RRF
        fused_results = reciprocal_rank_fusion(
            bm25_results=bm25_candidates,
            semantic_results=semantic_candidates
        )

        return fused_results[:top_k]

    def hybrid_search_explained(
        self, 
        query: str, 
        top_k: int = 5,
        source_filter: str | None = None,
        document_type: str | None = None
    ) -> None:
        """
        Runs hybrid search with metadata filters and prints debug table.
        """
        results = self.search(
            query, 
            top_k=top_k, 
            source_filter=source_filter, 
            document_type=document_type
        )

        filter_info = []
        if source_filter:
            filter_info.append(f"source='{source_filter}'")
        if document_type:
            filter_info.append(f"doc_type='{document_type}'")
        filter_str = f" [Filters: {', '.join(filter_info)}]" if filter_info else " [Filters: None]"

        console.rule(f"[bold cyan]Hybrid Search Debug: '{query}'{filter_str}[/bold cyan]")

        table = Table(show_lines=True)
        table.add_column("Rank", style="bold yellow", width=6)
        table.add_column("RRF Score", style="magenta", width=12)
        table.add_column("Source(s)", style="bold cyan", width=18)
        table.add_column("Document Snippet", style="green")

        if not results:
            table.add_row("-", "0.0000", "None", "[red]No matching documents found with active metadata filters.[/red]")
        else:
            for rank, r in enumerate(results, start=1):
                sources_str = " + ".join(r["sources"])
                sources_fmt = f"[bold green]{sources_str}[/bold green]" if len(r["sources"]) > 1 else f"[yellow]{sources_str}[/yellow]"
                snippet = r["text"].replace("\n", " ")[:120] + "..."
                table.add_row(str(rank), f"{r['rrf_score']:.5f}", sources_fmt, snippet)

        console.print(table)
        console.print("\n")

if __name__ == "__main__":
    engine = HybridSearchEngine()
    engine.hybrid_search_explained("Adam optimizer beta1 0.9", top_k=3)
