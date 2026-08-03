import os
import chromadb
from chromadb.utils import embedding_functions
from rich.console import Console
from rich.table import Table

from bm25_search import BM25Index

console = Console()

def run_comparison():
    # 1. Initialize BM25
    bm25 = BM25Index(storage_path="bm25_index.json")

    # 2. Initialize ChromaDB safely with explicit embedding function
    chroma_db_path = "../private-knowledge-assistant-week5/data/chroma_db"
    client = chromadb.PersistentClient(path=chroma_db_path)
    
    # Explicitly use the default sentence-transformer model to avoid TorchCodec DLL crashes
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    chroma_collection = client.get_collection(
        name="private_knowledge_base",
        embedding_function=emb_fn
    )

    # Test queries designed to highlight strengths and weaknesses of each approach
    queries = [
            # 1. Exact Technical Terms (BM25 should win clearly here)
            "Adam optimizer beta1 0.9",
            "BLEU scores English-to-German",
            "Modern Prometheus Wollstonecraft",
            # 2. Broad / Paraphrased Meaning (Semantic Search should win clearly here)
            "How does the model calculate attention between words?",
            "Why did the creator feel guilty about his monster?",
            "What training hardware and GPUs were used?"
        ]

    for q in queries:
        console.rule(f"[bold cyan]Query: '{q}'[/bold cyan]")

        # Run BM25 Search
        bm25_results = bm25.search(q, top_k=2)

        # Run Semantic Search
        chroma_results = chroma_collection.query(
            query_texts=[q],
            n_results=2
        )

        table = Table(title=f"Results for: {q}", show_lines=True)
        table.add_column("Engine", style="bold yellow", width=12)
        table.add_column("Score / Distance", style="magenta", width=16)
        table.add_column("Snippet", style="green")

        # Add BM25 Rows
        if bm25_results:
            for r in bm25_results:
                snippet = r['text'].replace('\n', ' ')[:120] + "..."
                table.add_row("BM25", f"Score: {r['score']:.4f}", snippet)
        else:
            table.add_row("BM25", "No Match (0.0)", "[red]No matching tokens found[/red]")

        # Add Semantic Rows
        if chroma_results and chroma_results.get("documents"):
            docs = chroma_results["documents"][0]
            distances = chroma_results["distances"][0] if chroma_results.get("distances") else ["N/A"] * len(docs)
            for doc, dist in zip(docs, distances):
                snippet = doc.replace('\n', ' ')[:120] + "..."
                dist_str = f"Dist: {dist:.4f}" if isinstance(dist, float) else str(dist)
                table.add_row("Semantic", dist_str, snippet)

        console.print(table)
        console.print("\n")

if __name__ == "__main__":
    run_comparison()