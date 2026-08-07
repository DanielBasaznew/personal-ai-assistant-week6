import json
import os
import re
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()

class QueryIntent(BaseModel):
    clean_query: str = Field(
        description="The core search query with operational phrases stripped out."
    )
    source_filter: str | None = Field(
        default=None,
        description="Exact document name referenced by the user (e.g., 'resnet_paper'), or None."
    )
    document_type: str | None = Field(
        default=None,
        description="Type of document requested (e.g., 'notes', 'paper', 'book'), or None."
    )
    requires_documents: bool = Field(
        default=True,
        description="False if the query is purely conversational (e.g., 'hello', 'thanks')."
    )
    requires_memory: bool = Field(
        default=False,
        description="True if asking about personal user history or past conversations."
    )


SYSTEM_PROMPT = """You are an expert Query Intent Router for a RAG search engine.
Analyze the user's input question and extract a structured QueryIntent JSON object.

Available Document Sources: {available_sources}
Available Document Types: ['notes', 'paper', 'book', 'pdf']

Rules:
1. Extract 'clean_query' by stripping operational phrases like "search in my notes for", "what does the paper say about", "search everything for", etc.
2. If the user mentions a specific source filename from the available sources list, set 'source_filter' to that exact document name.
3. If the user mentions a document category (e.g., 'notes', 'paper', 'book'), set 'document_type' to that type.
4. Set 'requires_documents' to False for simple greetings or non-retrieval chat (e.g., "hi", "thank you").
5. CRITICAL: If the user explicitly says "search everything", "all documents", or "search across everything", set BOTH 'source_filter' and 'document_type' to None, regardless of any keywords in the query.
"""


def parse_query_intent(user_query: str, available_sources: list[str]) -> QueryIntent:
    """
    Parses user question using Gemini API to generate a structured QueryIntent object.
    Falls back to heuristic parsing if API call fails or key is missing.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("[Warning] GEMINI_API_KEY not found in .env. Using fallback heuristic parser.")
        q_lower = user_query.lower()
        doc_type = None
        if "notes" in q_lower:
            doc_type = "notes"
        elif "paper" in q_lower:
            doc_type = "paper"
        elif "book" in q_lower:
            doc_type = "book"
            
        is_chat = q_lower in ["hi", "hello", "thanks", "thank you", "hey"]
        
        return QueryIntent(
            clean_query=user_query,
            document_type=doc_type,
            requires_documents=not is_chat
        )

    try:
        client = genai.Client(api_key=api_key)
        prompt = SYSTEM_PROMPT.format(available_sources=available_sources)
        
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=f"{prompt}\n\nUser Question: {user_query}",
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=QueryIntent,
                temperature=0.0
            )
        )
        
        return QueryIntent.model_validate_json(response.text)

    except Exception as e:
        print(f"[Warning] Gemini API call failed ({e}). Falling back to original query.")
        return QueryIntent(clean_query=user_query, requires_documents=True)


if __name__ == "__main__":
    sources = ["my_notes", "resnet_paper", "frankenstein_book", "transformer_paper"]
    
    test_queries = [
        "In my notes, what did I write about RAG?",
        "What does resnet_paper say about residual learning?",
        "Search everything for Frankenstein",
        "Hello, how are you today?"
    ]

    print("--- Testing Query Router Intent Parsing (Gemini API) ---")
    for q in test_queries:
        intent = parse_query_intent(q, available_sources=sources)
        print(f"\nUser Query: '{q}'")
        print(f"Parsed Intent -> {intent.model_dump()}")