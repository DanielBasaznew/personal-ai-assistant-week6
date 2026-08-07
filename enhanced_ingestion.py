import datetime
import os
import chromadb
from chromadb.utils import embedding_functions

from bm25_search import BM25Index

def ingest_document_with_metadata(
    doc_id: str,
    text: str,
    doc_name: str,
    doc_type: str,
    tags: str = "",
    chroma_path: str = "../private-knowledge-assistant-week5/data/chroma_db",
    bm25_path: str = "bm25_index.json"
) -> None:
    """
    Ingests a document chunk into both ChromaDB and BM25Index with enriched metadata.
    """
    metadata = {
        "document_name": doc_name,
        "document_type": doc_type,
        "tags": tags,
        "ingested_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }

    # 1. Add to BM25 Index
    bm25 = BM25Index(storage_path=bm25_path)
    bm25.add_documents([{
        "id": doc_id,
        "text": text,
        "metadata": metadata
    }])

    # 2. Add to ChromaDB Vector Store
    client = chromadb.PersistentClient(path=chroma_path)
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.get_collection(
        name="private_knowledge_base",
        embedding_function=emb_fn
    )

    # Upsert chunk with rich metadata
    collection.add(
        ids=[doc_id],
        documents=[text],
        metadatas=[metadata]
    )

    print(f"Successfully ingested '{doc_id}' into both BM25 and ChromaDB with metadata: {metadata}")


if __name__ == "__main__":
    # Test enhanced ingestion with sample chunks
    sample_text_1 = "RAG pipelines combine vector retrieval with large language models."
    ingest_document_with_metadata(
        doc_id="test_chunk_001",
        text=sample_text_1,
        doc_name="my_notes",
        doc_type="notes",
        tags="rag,llm,ai"
    )

    sample_text_2 = "Deep residual learning for image recognition uses shortcut connections."
    ingest_document_with_metadata(
        doc_id="test_chunk_002",
        text=sample_text_2,
        doc_name="resnet_paper",
        doc_type="paper",
        tags="cv,deep_learning"
    )