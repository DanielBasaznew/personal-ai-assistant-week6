from memory import PersistentMemory
from memory_extractor import extract_facts_from_conversation


def process_turn_and_update_memory(
    mem: PersistentMemory, session_id: str, user_message: str
) -> None:
  """1. Logs conversation turn (Episodic Memory)

  2. Extracts facts via LLM
  3. Updates facts table (Semantic Memory)
  """
  # 1. Log turn
  mem.log_conversation(session_id, "user", user_message)

  # 2. Get current formatted facts for LLM context
  current_facts_formatted = mem.format_for_prompt()

  # 3. Extract facts
  extracted_facts = extract_facts_from_conversation(
      user_message, current_facts_formatted
  )

  # 4. Apply database updates
  for fact in extracted_facts:
    if fact.action in ["store", "update"]:
      mem.store_fact(fact.key, fact.value, category=fact.category)
      print(
          f"  [DB UPDATE] Stored/Updated '{fact.key}' -> '{fact.value}'"
          f" ({fact.category})"
      )
    elif fact.action == "delete":
      deleted = mem.delete_fact(fact.key)
      if deleted:
        print(f"  [DB UPDATE] Deleted fact key '{fact.key}'")


def main():
  # Use a fresh test database for this end-to-end run
  mem = PersistentMemory("integrated_memory.db")
  session_id = "session_001"

  turns = [
      "Hi, my favorite language is Python and I prefer concise answers.",
      "I also love Rust now and want to build AI agents.",
      "Forget my favorite language.",
      "How do Transformer attention mechanisms work?",
  ]

  print("=== End-to-End Memory Integration Test ===")

  for turn_idx, user_msg in enumerate(turns, start=1):
    print(f"\n--- Turn {turn_idx}: '{user_msg}' ---")
    process_turn_and_update_memory(mem, session_id, user_msg)
    print("\n[Current Stored Memory Prompt Block]:")
    print(mem.format_for_prompt())


if __name__ == "__main__":
  main()