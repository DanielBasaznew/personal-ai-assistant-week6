import os
import chromadb
from bm25_search import BM25Index

def populate_bm25_from_chroma():
    # Relative path pointing to your Week 5 data directory
    chroma_db_path = "../private-knowledge-assistant-week5/data/chroma_db"
    
    if not os.path.exists(chroma_db_path):
        print(f"Error: Could not find ChromaDB at path: {chroma_db_path}")
        return

    client = chromadb.PersistentClient(path=chroma_db_path)
    
    # Using your exact collection name from Week 5
    collection_name = "private_knowledge_base"
    collection = client.get_collection(name=collection_name)
    
    # Retrieve all stored chunks
    all_data = collection.get()
    
    ids = all_data.get("ids", [])
    documents = all_data.get("documents", [])
    metadatas = all_data.get("metadatas", [])
    
    print(f"Found {len(ids)} chunks in ChromaDB ('{collection_name}').")
    
    # Format for BM25Index
    bm25_docs = []
    for doc_id, doc_text, meta in zip(ids, documents, metadatas):
        bm25_docs.append({
            "id": doc_id,
            "text": doc_text,
            "metadata": meta or {}
        })

    # Add to BM25 Index
    bm25 = BM25Index(storage_path="bm25_index.json")
    bm25.add_documents(bm25_docs)
    print(f"Successfully indexed {len(bm25_docs)} documents into BM25 ('bm25_index.json').")

if __name__ == "__main__":
    populate_bm25_from_chroma()