import os
import uuid
from dotenv import load_dotenv
from google import genai
from rich.console import Console

from memory import PersistentMemory
from memory_extractor import extract_facts_from_conversation
from hybrid_search import HybridSearchEngine
from query_router import parse_query_intent

load_dotenv()
console = Console()

class PersonalAIAssistant:
    def __init__(self):
        # 1. Initialize core systems
        self.memory = PersistentMemory("assistant_memory.db")
        self.search_engine = HybridSearchEngine()
        
        # 2. Setup Gemini client
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=self.api_key)
        
        # 3. Session state (In-Context Memory)
        self.session_id = f"session_{uuid.uuid4().hex[:8]}"
        self.chat_history = []

    def _get_available_sources(self) -> list[str]:
        """Pulls distinct document names from ChromaDB to pass to the Query Router."""
        try:
            data = self.search_engine.chroma_collection.get(include=["metadatas"])
            if data and data.get("metadatas"):
                return list(set(meta.get("document_name") for meta in data["metadatas"] if meta and "document_name" in meta))
        except Exception:
            pass
        return []

    def _build_system_prompt(self) -> str:
        """Injects persistent facts into the prompt with natural personality instructions."""
        facts_block = self.memory.format_for_prompt()
        
        return f"""You are a highly capable, personalized AI assistant.
        
{facts_block}

CRITICAL INSTRUCTIONS:
1. When answering questions, use the personal context above to tailor your response.
2. DO NOT explicitly reference "my records", "the database", or "my memory". Respond naturally as if you simply remember the person.
3. Answer clearly and concisely.
"""

    def _update_memory(self, user_message: str, assistant_message: str):
        """Runs in the background after generating a response to update semantic and episodic memory."""
        # Log episodic memory
        self.memory.log_conversation(self.session_id, "user", user_message)
        self.memory.log_conversation(self.session_id, "assistant", assistant_message)
        
        # Extract and update semantic facts
        current_facts = self.memory.format_for_prompt()
        extracted_facts = extract_facts_from_conversation(user_message, current_facts)
        
        for fact in extracted_facts:
            if fact.action in ["store", "update"]:
                self.memory.store_fact(fact.key, fact.value, category=fact.category)
                console.print(f"  [dim cyan]💾 Remembered: {fact.key} -> {fact.value}[/dim cyan]")
            elif fact.action == "delete":
                self.memory.delete_fact(fact.key)
                console.print(f"  [dim red]🗑️ Forgot: {fact.key}[/dim red]")

    def chat(self, user_input: str) -> str:
        """The main orchestration pipeline for a single conversation turn."""
        # 1. Route the query
        available_sources = self._get_available_sources()
        intent = parse_query_intent(user_input, available_sources)
        
        # 2. Retrieve document context if required
        context_block = ""
        if intent.requires_documents:
            results = self.search_engine.search(
                query=intent.clean_query, 
                top_k=3,
                source_filter=intent.source_filter,
                document_type=intent.document_type
            )
            if results:
                context_block = "\n\nRetrieved Document Context:\n"
                for i, r in enumerate(results, 1):
                    context_block += f"--- Chunk {i} ---\n{r['text']}\n"
        
        # 3. Build prompts and history
        sys_prompt = self._build_system_prompt()
        
        # Format conversation history for Gemini API
        contents = [{"role": "user", "parts": [{"text": sys_prompt}]}] # Inject system prompt as first user message context
        contents.append({"role": "model", "parts": [{"text": "Understood. I am ready to assist."}]})
        
        for msg in self.chat_history:
            contents.append({"role": msg["role"], "parts": [{"text": msg["content"]}]})
            
        # Append current user message (with context if applicable)
        full_user_prompt = user_input + context_block
        contents.append({"role": "user", "parts": [{"text": full_user_prompt}]})
        
        # 4. Generate LLM response
        try:
            response = self.client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=contents
            )
            answer = response.text
        except Exception as e:
            return f"Error connecting to LLM: {e}"
            
        # 5. Update session history
        self.chat_history.append({"role": "user", "content": user_input})
        self.chat_history.append({"role": "model", "content": answer})
        
        # 6. Extract persistent memory (fire and forget)
        self._update_memory(user_input, answer)
        
        return answer

    def run(self):
        """Interactive terminal loop."""
        console.print("[bold green]Personal AI Assistant Initialized.[/bold green]")
        console.print("Commands: 'quit' to exit, 'memory' to view facts, '/forget <key>' to delete a fact.")
        console.print("-" * 50)
        
        while True:
            user_input = console.input("\n[bold blue]You:[/bold blue] ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                break
                
            if user_input.lower() == 'memory':
                console.print(self.memory.format_for_prompt())
                continue
                
            if user_input.lower().startswith("/forget "):
                key = user_input.split(" ", 1)[1]
                if self.memory.delete_fact(key):
                    console.print(f"[green]Deleted '{key}' from memory.[/green]")
                else:
                    console.print(f"[yellow]Fact '{key}' not found.[/yellow]")
                continue

            if not user_input:
                continue

            response = self.chat(user_input)
            console.print(f"\n[bold magenta]Assistant:[/bold magenta] {response}")


if __name__ == "__main__":
    app = PersonalAIAssistant()
    app.run()